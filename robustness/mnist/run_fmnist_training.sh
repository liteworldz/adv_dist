INPUT=FMNIST_training.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_sparse_categorical_accuracy
do
	for trials in 10; do
		for steps in 100; do
			for eps in 10.0 50.0; do
				eval $(echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l2") | tee -a fminst-output.csv
			done
			for eps in 1.0 2.0; do
				eval $(echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l2") | tee -a fminst-output.csv
			done
			for eps in 0.1 0.15; do
				eval $(echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm linf") | tee -a fminst-output.csv
			done
		done
	done
done < FMNIST_training.csv 
IFS=$OLDIFS