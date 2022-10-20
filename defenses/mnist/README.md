# Model training
`python train.py`

It will train a convolutional net against a PGD L_inf adversary using 40 steps and eps=0.3.  
Model checkpoints are stored in `saved_models` directory.

1 epoch takes 205s on an RTX 2080 TI gpu. 

# Model training on a multi-gpu system
Training only on `GPU 1`:  
`python train.py --gpu 1`  
Define the size of the allocated memory on the GPU in MB:  
`python train.py --gpu 1 --memory_limit 2000`

# Customization
* Define custom epoch number and batch size  
`python train.py --epoch 50 --batch_size 100`
* or Define custom steps number and eps  
`python train.py --steps 10 --eps 0.1`
* You can switch the data set by implementing the `DataSet` abstract class in [dataset.py](https://github.com/szegedai/examples/blob/master/defenses/mnist/dataset.py)
* You can switch the model by implementing the `Model` abstract class in [models.py](https://github.com/szegedai/examples/blob/master/defenses/mnist/models.py)
* You can switch the loss function or the used metrics in [train.py](https://github.com/szegedai/examples/blob/master/defenses/mnist/train.py)
 