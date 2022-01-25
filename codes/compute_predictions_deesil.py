import sys
import os
import numpy as np
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool

""" list of arguments for the script """
dataset_name =  sys.argv[1]
s10 = sys.argv[2]
S = int(s10[1:])
root_path = sys.argv[3]
model_dir = os.path.join(root_path,f"svms/{dataset_name}/{s10}/deesil")
test_feats_path = os.path.join(root_path,f"features/{dataset_name}/{s10}/test/")
pred_path = os.path.join(root_path,f"predictions/{dataset_name}/{s10}/deesil/")
os.makedirs(pred_path, exist_ok=True)
nb_classes = 1000
if dataset_name == 'cifar100' or dataset_name=='additional':
   nb_classes = 100
   
def compute_feature(i):
   corresponding_batch = i//(nb_classes//S)+1
   #print('compute feature',i, 'corresponding_batch', corresponding_batch)
   test_feats = os.path.join(test_feats_path,str(i))
   print(test_feats)
   for batchs in range(corresponding_batch, S+1):
      os.makedirs(os.path.join(pred_path,"batch"+str(batchs)),exist_ok=True)
      pred_file = os.path.join(pred_path,"batch"+str(batchs),str(i))
      if not os.path.exists(pred_file): # TODO 
         with open(pred_file, "w") as f_pred:
            syns = []
            f_list_syn = list(range((nb_classes//S)*(batchs)))
            #print('class',f_list_syn[-1],'batch', batchs)
            for syn in f_list_syn:
               syn = str(syn)
               syns.append(syn)
            print("synsets:",len(syns))
            weights_list = []  
            biases_list = []
            for syn in range(0,len(syns)):
               line_cnt = 0 # counter to get the weights and bias lines
               current_C = '1'#precomputed[(dataset_name,s10)][corresponding_batch]
               target_model = os.path.join(model_dir,str(syn)+".model")
               f_model = open(target_model)
               for line in f_model:
                  line = line.rstrip()
                  if line_cnt == 0:
                     parts = line.split(" ")
                     parts_float = [] # tmp list to store the weights
                     for pp in parts:
                        parts_float.append(float(pp))
                     weights_list.append(parts_float)
                  elif line_cnt == 1:
                     biases_list.append(float(line))
                  line_cnt = line_cnt + 1
               f_model.close()
            print("list sizes - weights:",len(weights_list),"; biases:",len(biases_list))
            f_test_feat = open(test_feats, 'r')
            for vline in f_test_feat.readlines():
               vparts = vline.split(" ")
               crt_feat = [[float(vp) for vp in vparts]]
               crt_feat = Normalizer().fit_transform(crt_feat)[0]
               pred_dict = []
               for cls_cnt in range(0, len(weights_list)):
                  cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
                  pred_dict.append(cls_score)
               pred_line = ""
               predictions_idx = sorted(range(len(pred_dict)), key=lambda k: -pred_dict[k])
               for idx in predictions_idx:
                  pred_line = pred_line+" "+str(idx)+":"+str(pred_dict[idx]) 
               pred_line = pred_line.lstrip()
               f_pred.write(pred_line+"\n")
            f_test_feat.close()
      else:
         print("exists predictions file:",pred_file)
with Pool() as p:
   p.map(compute_feature, range(nb_classes))