for dataset in FMNIST; do
	for method in svd; do
		for comps in 150; do  
			for denses in 1 6; do
				for dense_size in 256; do 
					echo "python train.py --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 0 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
					echo "python train.py --save_dir $dataset --dataset $dataset --epoch 100 --comps $comps --method $method --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
				done  
			done
		done   
	done
done