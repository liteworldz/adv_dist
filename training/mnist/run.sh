for dataset in MNIST FMNIST; do
	for method in sparse; do 
		for freeze in 0 1; do 
			for normalize1 in 1 0; do
				for normalize2 in 1 0; do
					for denses in 1 2 3 4; do
						for dense_size in 128 256; do 
							for comps in 10 20 30 40 50 60 70 80 90 100; do
								eval $(echo "python train.py --gpu 0 --save_dir $dataset --epoch 100 --comps $comps --method $method --freeze $freeze --normalize1 $normalize1 --normalize2 $normalize2 --denses $denses --dense_size $dense_size") 
							done  
						done
					done 
					for denses in 0; do
						for comps in 10 20 30 40 50 60 70 80 90 100; do
								eval $(echo "python train.py --gpu 0 --save_dir $dataset --epoch 100 --comps $comps --method $method --freeze $freeze --normalize1 $normalize1 --normalize2 $normalize2 --denses $denses") 
						done  
					done
				done
			done
		done
	done
done