for dataset in FMNIST; do
	for method in svd; do  
		for denses in 0 1 2 3 4; do
			for dense_size in 128 256; do 
				for comps in 10 20 30 40 50 60 70 80 90 100; do
					eval $(echo "python train.py --gpu 0 --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 0 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size")
					eval $(echo "python train.py --gpu 0 --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size")
					eval $(echo "python train.py --gpu 0 --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 1 --normalize1 1 --normalize2 0 --denses $denses --dense_size $dense_size")
					eval $(echo "python train.py --gpu 0 --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 1 --normalize1 1 --normalize2 1 --denses $denses --dense_size $dense_size")
				done  
			done
		done   
	done
done