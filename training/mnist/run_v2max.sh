for comps in 60; do
	eval $(echo "python train.py --gpu 0 --save_dir MNIST --dataset MNIST --epoch 100 --comps $comps --method svd --v2 1 --freeze 1 --normalize1 0 --normalize2 0 --denses 0 --dense_size 256") 
done