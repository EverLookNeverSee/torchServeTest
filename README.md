# torchServeTest
## Deploying Machine Learning Model with TorchServe


### What is model serving?
Model serving is the process of situating a trained machine learning model within a system
so that it can take new inputs and return the inferences to the system.

### What is torchserve?
TorchServe is the preferred model serving solution for pytorch. It allows you to expose a
web API for your model that may be accessed directly or via your application.


### Installing the project dependencies
1. Creating a conda environment:
```shell
conda create --name pt python=3.9
```
2. Activating the environment:
```shell
conda activate pt
```
3. Installing dependencies:
```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 torchserve torch-model-archiver -c pytorch
```
