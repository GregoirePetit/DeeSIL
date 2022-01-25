import os
import sys
import csv
import numpy as np
from multiprocessing import Pool

dataset = sys.argv[1]
s10 = sys.argv[2]
root_path = sys.argv[3]
nb_classes = 1000
if dataset=='cifar100' or dataset=='additional':
    nb_classes = 100
s = int(s10[1:])
batch_size = nb_classes//s
root_path_pred = os.path.join(root_path,f"predictions/{dataset}/{s10}/deesil/")

resultats = {}
def compute_score(nb_batch):
    path_pred = os.path.join(root_path_pred,"batch"+str(nb_batch))
    y_pred = []
    y_true = []
    score_top5 = []
    for c in range(batch_size*nb_batch):
        with open(os.path.join(path_pred,str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])
    y_pred = np.asarray(y_pred)
    y_pred_top5 = np.asarray(score_top5)
    y_true = np.asarray(y_true)
    return((nb_batch,[np.mean(y_pred == y_true),np.mean(y_pred_top5)]))

with Pool() as p:
    resultats = dict(p.map(compute_score, range(1,s+1)))

top1=[]
top5=[]
for batch_number in range(1,s+1):
    print(f'batch {batch_number}, top1 = {resultats[batch_number][0]:.3f}, top5 = {resultats[batch_number][1]:.3f}')
    top1.append(resultats[batch_number][0])
    top5.append(resultats[batch_number][1])
print('=================================================')
print('===================  TOTAL  =====================')
print(f'top1 = {sum(top1[1:])/len(top1[1:]):.3f}, top5 = {sum(top5[1:])/len(top5[1:]):.3f}')