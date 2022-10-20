# Environment setup

## GPU version
### CUDA setup
* from NVIDIA download and install CUDA -> cuda version 11.3.1 cuDNN library version 8.2.1
* start by creating an account [here](https://developer.nvidia.com/) 
```
conda create -n env_name python=3.8.5
conda activate env_name
conda config --env --add channels conda-forge
conda install numpy==1.19.5
pip install pandas==1.3.0
conda install matplotlib==3.2.2
pip install tensorflow-gpu==2.4.1
pip install foolbox==3.3.1
conda install -c conda-forge scikit-learn 
```

## List Installed Packages
```sh
conda list
```

## Test
`python test.py tensorflow-gpu matplotlib`
The output should be:
```sh
libary_name succeed version
tensorflow-gpu True 2.4.1
matplotlib True 3.2.2
```

## 'cusolver64_10.dll' Issue Fix
```
Tensorflow GPU Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
```
Steps to resolve this issue:
* Move to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
* Rename file cusolver64_11.dll  To  cusolver64_10.dll 


