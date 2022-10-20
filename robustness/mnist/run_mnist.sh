INPUT=MNIST_training.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_accuracy
do
	eval $(echo "File Loaded:$filename")
	for trials in 1 10; do
		for steps in 40 100; do
			for eps in 1.0 2.0; do
				eval $(echo "python main.py --dataset MNIST --gpu 0 --fname $filename --trials $trials --steps $steps --eps $eps --norm l2") | tee -a minst-output.csv
			done
			for eps in 0.15 0.1 0.2 0.3; do
				eval $(echo "python main.py --dataset MNIST --gpu 0 --fname $filename --trials $trials --steps $steps --eps $eps --norm linf") | tee -a minst-output.csv
			done
		done
	done
done < MNIST_training.csv 
IFS=$OLDIFS

read line