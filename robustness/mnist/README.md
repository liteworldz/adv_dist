# Evaluating model robustness on MNIST against a PGD adversary using infinite norm
You can pass your own model or download one from [here](https://github.com/liteworld369/svd_robustness/tree/main/training/mnist/saved_models).  
`python main.py --fname path_to_model.h5`

## Adjusting evaluation parameters
Note, according to the evaluated model size you need to change some parameters.

For example, when you got `OutOfMemoryException` you might define a smaller batch size such as 100.   
 `python main.py --fname path_to_model.h5 --batch_size 100`  
Alternatively, you can increase the avaiable memory by:  
 `python main.py --fname path_to_model.h5 --memory_limit 10000`  
 Or both...  
In general, the larger the batch_size the faster the evaluation and the more memory required. However, larger models lessly fit into memory, so those needs smaller batch size. 

## Modify attack parameters
You can strengthen the `PGD` adversary by changing certain parameters.
Increasing epsilon from the default `0.1` to `0.3`:  
`python main.py --fname path_to_model.h5 --eps 0.3`

You might increase the number of steps from the default `40` to `100`:  
`python main.py --fname path_to_model.h5 --steps 100`  
Note, when you change the number of steps, you might modify the step_size as well. E.g. it is common, to increase the steps while decrease the step size.

An other way to strengthen your adversary is applying the attack with multiple random initial point, and evaluate against the worst perturbation.  
`python main.py --fname path_to_model.h5 --trials 10`

The attack used in [link](https://arxiv.org/abs/1706.06083) can be applied using the following configuration:  
`python main.py --fname path_to_model.h5 --trials 50 --eps 0.3 --steps 100`
