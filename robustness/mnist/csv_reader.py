import pandas as pd
import glob
import os
import csv


path = r'../../training/mnist/FMNIST'
all_files = glob.glob(path + "/*.csv")

li = []
header = ['filename', 'model_index', 'val_sparse_categorical_accuracy']
f = open('FMNIST_training.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(header)

for csv_filename in all_files:
    df = pd.read_csv(csv_filename, index_col=None, header=0)
    rowIndex = df['val_sparse_categorical_accuracy'].idxmax()
    rowMax = df['val_sparse_categorical_accuracy'].max()
    li.append([os.path.basename(csv_filename),[rowIndex,rowMax]] )  
    model_path = path + '/MLP' +  os.path.basename(csv_filename).replace('.csv','') + f'_{rowIndex+1:03d}.h5'
    writer.writerow([model_path, rowIndex, f'{rowMax:.2f}'])



#print([li[0][1]][0][1])