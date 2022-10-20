INPUT=FMNIST_training.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_accuracy
do
	eval $(echo "File Loaded:$filename")
	for trials in 1 10; do
		for steps in 40 100; do
			for eps in 1.0 2.0; do
				eval $(echo "python main.py --dataset FMNIST --attack br  --gpu 0 --fname $filename --trials $trials --steps $steps --eps $eps --norm l2") | tee -a fminst-v2max-output.csv
			done
		done
	done
done < FMNIST_training.csv 
IFS=$OLDIFS

read line