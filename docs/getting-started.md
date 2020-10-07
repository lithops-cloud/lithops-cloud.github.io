# Lithops

Lithops is a Python multicloud library for running serverless jobs. Lithops' goals are massively scaling the execution of Python code and its dependencies on serverless computing platforms and monitoring the results. Lithops delivers the user’s code into the serverless platform without requiring knowledge of how functions are invoked and run. 

It currently supports AWS, IBM Cloud, Google Cloud, Microsoft Azure, Alibaba Aliyun, and more.

### Quick start
1. Install Lithops from the PyPi repository:

```
$ pip install lithops
```

2. Follow [choose compute backend and storage](https://github.com/lithops-cloud/lithops/tree/master/config) to configure Lithops.

3. Test Lithops by simply running the next command:
  
```
$ lithops test
```

   or by running the following code:

```python
import lithops

def hello(name):
   return 'Hello {}!'.format(name)

fexec = lithops.function_executor()
fexec.call_async(hello, 'World')
print(fexec.get_result())
```
   
## Documentation
- [Website](https://cloudbutton.github.io)
- [Lithops API](https://github.com/lithops-cloud/lithops/blob/master/docs/api-details.md)
- [Examples](https://github.com/lithops-cloud/lithops/tree/master/examples)
- [Plugins](https://github.com/lithops-cloud/)
