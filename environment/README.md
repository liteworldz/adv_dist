# Custom Environment setup

## GPU version  
```sh
conda create -n env_name python=3.8.5
conda activate env_name
conda config --env --add channels conda-forge
conda install numpy==1.19.5
conda install matplotlib==3.2.2
pip install tensorflow-gpu==2.4.1
pip install foolbox==3.3.1
```

## Test
`python test.py tensorflow-gpu matplotlib`
The output should be:
```sh
libary_name succeed version
tensorflow-gpu True 2.4.1
matplotlib True 3.2.2
```

The compatibility list of the `tensorfflow/tensorflow-gpu`, `cuda` and `python` 
versions can be found [here](https://www.tensorflow.org/install/source#tested_build_configurations).

The compatibility list of the `cuda` and `GPU driver` versions can be found 
[here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility)
