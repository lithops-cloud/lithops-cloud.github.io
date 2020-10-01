# Lithops Moments in Time dataset example
## Video/image prediction
In [this notebook](https://github.com/cloudbutton/examples/blob/master/momentsintime/example_mit.ipynb) we will process video clips from the MiT dataset at scale with Lithops
by predicting the actions with a pretrained ResNet50 model and then counting how many
occurrences of each category have been predicted.



```python
import time
import builtins
import torch.optim
import torch.nn.parallel
from torch import save, load
from torch.nn import functional as F

from utils import extract_frames
from models import load_model, load_transform, load_categories

from lithops.multiprocessing import Pool, Queue
from lithops.multiprocessing.util import get_uuid
```

### Backends
The same program can be run in a local environtment with processes or executed by
functions in the cloud. After we choose a backend, only a few file locations must
be changed. In this example we will be using the cloud functions backend.

We will be using a custom runtime for our functions which has torch, torchvision,
ffmpeg and opencv-python modules already installed.
We will store the pretrained weights in the cloud so that functions can access it.
Then, after functions get the models weights they will start preprocessing input
videos and inferring them one by one.
  
Later in this notebook, we will see a little improvement detail to this process.  



```python
LOCAL_EXEC = False
```


```python
INPUT_DATA_DIR = 'momentsintime/input_data'

if LOCAL_EXEC:
    import os
    from builtins import open
    initargs = {
        'backend': 'localhost',
        'storage_backend': 'localhost'
        }
    weights_location = '/dev/shm/model_weights'
    INPUT_DATA_DIR = os.path.abspath(INPUT_DATA_DIR)

else:
    from lithops.cloud_proxy import os, open
    initargs = {
        'backend': 'ibm_cf',
        'storage_backend': 'ibm_cos',
        'runtime': 'dhak/pywren-runtime-pytorch:3.6',
        'runtime_memory': 2048
        }
    weights_location = 'momentsintime/models/model_weights'
    
```


```python
video_locations = [os.path.join(INPUT_DATA_DIR, name) for name in os.listdir(INPUT_DATA_DIR)]
```

As you can see, we have masked the `open` function and `os` module with a proxy
to manage files from the cloud transparently.  
We will use `builtins.open` from now on to explicitly access a local file as some accesses have to occur in the very same machine.

### Download pretrained ResNet50 model weights and save them in a directory accessible by all functions (`weights_location`)


```python
ROOT_URL = 'http://moments.csail.mit.edu/moments_models'
WEIGHTS_FILE = 'moments_RGB_resnet50_imagenetpretrained.pth.tar'

if not os.access(WEIGHTS_FILE, os.R_OK):
    os.system('wget ' + '/'.join([ROOT_URL, WEIGHTS_FILE]))

with builtins.open(WEIGHTS_FILE, 'rb') as f_in:
    weights = f_in.read()
with open(weights_location, 'wb') as f_out:
    f_out.write(weights)
```

### Video prediction and reduce function code



```python
NUM_SEGMENTS = 16

# Get dataset categories
categories = load_categories()

# Load the video frame transform
transform = load_transform()

def predict_videos(queue, video_locations):
    with open(weights_location, 'rb') as f:
        model = load_model(f)
    model.eval()

    results = []
    local_video_loc = 'video_to_predict_{}.mp4'.format(get_uuid())

    for video_loc in video_locations:
        start = time.time()
        with open(video_loc, 'rb') as f_in:
            with builtins.open(local_video_loc, 'wb') as f_out:
                f_out.write(f_in.read())

        # Obtain video frames
        frames = extract_frames(local_video_loc, NUM_SEGMENTS)

        # Prepare input tensor [num_frames, 3, 224, 224]
        input_v = torch.stack([transform(frame) for frame in frames])

        # Make video prediction
        with torch.no_grad():
            logits = model(input_v)
            h_x = F.softmax(logits, 1).mean(dim=0)
            probs, idx = h_x.sort(0, True)

        # Output the prediction
        result = dict(key=video_loc)
        result['prediction'] = (idx[0], round(float(probs[0]), 5))
        result['iter_duration'] = time.time() - start
        results.append(result)
    queue.put(results)

# Counts how many predictions of each category have been made
def reduce(queue, n):
    pred_x_categ = {}
    for categ in categories:
        pred_x_categ[categ] = 0

    checkpoint = 0.2
    res_count = 0

    for i in range(n):
        results = queue.get()
        res_count += len(results)
        for res in results:
            idx, prob = res['prediction']
            pred_x_categ[categories[idx]] += 1

        # print progress
        if i >= (N * checkpoint):
            print('Processed {} results.'.format(res_count))
            checkpoint += 0.2

    return pred_x_categ
```

### Map functions
Similar to the `multiprocessing` module API, we use a Pool to map the video keys
across n workers (concurrency). However, we do not have to instantiate a Pool of
n workers *specificly*, it is the map function that will invoke as many workers according
to the length of the list.


```python
CONCURRENCY = 1000
```


```python
queue = Queue()
pool = Pool(initargs=initargs)

# Slice data keys
N = min(CONCURRENCY, len(video_locations))
iterable = [(queue, video_locations[n::CONCURRENCY]) 
            for n in range(N)]

# Map and reduce on the go
start = time.time()
pool.map_async(func=predict_videos, iterable=iterable)
pred_x_categ = reduce(queue, N)
end = time.time()
    
print('\nDone.')
print('Videos processed:', len(video_locations))
print('Total duration:', round(end - start, 2), 'sec\n')

for categ, count in pred_x_categ.items():
    if count != 0:
        print('{}: {}'.format(categ, count))
```

---------------

## Performance improvement
Now, since we know every function will have to pull the model weights from
the cloud storage, we can actually pack these weights with the runtime image
and reduce the start-up cost substantially.


```python
initargs['runtime'] = 'dhak/pywren-runtime-resnet'
weights_location = '/momentsintime/model_weights'
```


```python
def predict_videos(queue, video_locations):
    # force local file access on new weights_location
    with builtins.open(weights_location, 'rb') as f:
        model = load_model(f)
    model.eval()

    results = []
    local_video_loc = 'video_to_predict_{}.mp4'.format(get_uuid())

    for video_loc in video_locations:
        start = time.time()
        with open(video_loc, 'rb') as f_in:
            with builtins.open(local_video_loc, 'wb') as f_out:
                f_out.write(f_in.read())

        # Obtain video frames
        frames = extract_frames(local_video_loc, NUM_SEGMENTS)

        # Prepare input tensor [num_frames, 3, 224, 224]
        input_v = torch.stack([transform(frame) for frame in frames])

        # Make video prediction
        with torch.no_grad():
            logits = model(input_v)
            h_x = F.softmax(logits, 1).mean(dim=0)
            probs, idx = h_x.sort(0, True)

        # Output the prediction
        result = dict(key=video_loc)
        result['prediction'] = (idx[0], round(float(probs[0]), 5))
        result['iter_duration'] = time.time() - start
        results.append(result)
    queue.put(results)
```


```python
queue = Queue()
pool = Pool(initargs=initargs)

# Slice data keys
N = min(CONCURRENCY, len(video_locations))
iterable = [(queue, video_locations[n::CONCURRENCY]) 
            for n in range(N)]

# Map and reduce on the go
start = time.time()
r = pool.map_async(func=predict_videos, iterable=iterable)
pred_x_categ = reduce(queue, N)
end = time.time()
    
print('\nDone.')
print('Videos processed:', len(video_locations))
print('Total duration:', round(end - start, 2), 'sec\n')

for categ, count in pred_x_categ.items():
    if count != 0:
        print('{}: {}'.format(categ, count))
```

### Clean


```python
try:
    os.remove(weights_location)
except FileNotFoundError:
    pass

try:
    os.remove(WEIGHTS_FILE)
except FileNotFoundError:
    pass
```

### Dockerfiles and build scripts for both runtimes can be found in the docker/ folder.

### Source code adapted from the demonstration in https://github.com/zhoubolei/moments_models

### Moments in Time article: http://moments.csail.mit.edu/#paper

