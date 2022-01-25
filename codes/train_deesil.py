import sys
import os
import math
from configparser import ConfigParser
from os.path import isfile, join
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
import numpy as np
from numpy.linalg import norm
from multiprocessing import Pool

import utils


#provide the following arguments to facilitate paralellization
regul = sys.argv[1] #value of the regularization parameter for the SVMs
toler = sys.argv[2] #value of the tolerance parameter for the SVMs
state_id = int(sys.argv[3]) #ID of the incremental state - i.e. 0-49 for 50 states
state_size = int(sys.argv[4]) # number of classes in each incremental state
dataset = sys.argv[5]
s10 = sys.argv[6]
root_path = sys.argv[7]
if False:
    print('regul',regul)
    print('toler',toler)
    print('state_id',state_id)
    print('state_size',state_size)
    print('dataset',dataset)
    print('s10',s10)





def normalize_train_features(il_dir,norm_type,state_id,state_size):
    """
        Normalizes the features of classes associated to the current IL state.
        Parameters
        ----------
        il_dir : string
            path to directory with data needed for 
        norm_type: string 
            type of normalization applied to raw features
        state_id: int
            ID the currently processed IL state
        state_size: int
            number of classes per incremental state

        Returns
        -------
        feats_libsvm : list 
            list of features formatted in libsvm format with original class labels
    """
    feats_libsvm = [] #create the list of features for the output
    
    #get the min and max positions of classes for the current batch
    min_pos = state_id * state_size
    max_pos = (state_id+1) * state_size
    #form the paths needed for class lists and for the raw features dir
    
    feat_dir = il_dir
    
    #open the list of classes and select those which will be processed 
    crt_pos = 0
    class_list = list(range(1000))
    if ('cifar100' in il_dir) or ('additional' in il_dir):
        class_list = list(range(100))
    f_list = class_list
    for crt_id in f_list:
        #process only the classes from the current IL state
        if crt_pos >= min_pos and crt_pos < max_pos:
            #print("crt class:",crt_id)
            #read the features of each selected class, normalize them
            #and then reformat them to libsvm format
            #print(feat_dir+str(crt_id))
            class_feats = os.path.join(feat_dir,str(crt_id))
            f_feats = open(class_feats)
            for crt_feat in f_feats:
                crt_feat = crt_feat.rstrip()
                #print(crt_feat)
                np_feat = np.fromstring(crt_feat,dtype=float,sep=' ')
                if norm_type == "l2":
                    #np_feat_l2 = normalize(np_feat,norm='l2',axis=0)
                    np_feat_l2 = utils.normalize_l2(np_feat)
                    #reformat the current feature to libsvm format
                    #keep the class IDs in the reformatted vector
                    libsvm_feat = str(crt_id)
                    for cdim in range(0,np_feat_l2.shape[0]):
                        #libsvm feature dimensions start at 1
                        crt_dim = cdim + 1
                        libsvm_feat = libsvm_feat+" "+str(crt_dim)+":"+str(np_feat_l2[cdim]) 
                    #print(libsvm_feat)
                    feats_libsvm.append(libsvm_feat)
                else:
                    print("exiting - unknown normalization type:",norm_type)
                    sys.exit(0)
            f_feats.close()
        crt_pos = crt_pos+1
    
    return feats_libsvm
    
    
""" MAIN """
if __name__ == '__main__':

    #read the parameters from the configuration file
    curri = str(state_id+1)
    il_dir = os.path.join(root_path,f"features/{dataset}/{s10}/train")
    svms_dir = os.path.join(root_path,f"svms/{dataset}/{s10}/deesil")
    #print(il_dir)
    norm_type = 'l2'
    #state_id = int(cp['state_id'])
    #state_size = int(cp['state_size'])

    #normalize the features for the current IL state
    norm_feats = normalize_train_features(il_dir,norm_type,state_id,state_size)

    #initialize the SVM configuration
    clf = LinearSVC(penalty='l2', dual=False, tol=float(toler), C=float(regul), multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
    min_pos = state_id * state_size
    max_pos = (state_id+1) * state_size
    #create a root directory for SVMs of the current dataset configuration
    
    os.makedirs(svms_dir, exist_ok=True)
    #create a subdirectory for the SVMs trained with the current pair of regularization and tolerance
    pair_svms_dir = svms_dir
    os.makedirs(pair_svms_dir, exist_ok=True)

    def calc_thrd(crt_id):
        crt_id = str(crt_id)
        print("training: ",crt_id)
        #train the SVM for the current class and param configuration only if not already done
        crt_id_svm_path = os.path.join(pair_svms_dir,crt_id+".model")
        crt_id_svm_path_scratch = os.path.join(pair_svms_dir,crt_id+".model")
        if (not os.path.exists(crt_id_svm_path)) and (not os.path.exists(crt_id_svm_path_scratch)):
            f_svm = open(crt_id_svm_path,"w")
            #TODO - replace the libsvm loading by a more efficient one
            #write the libsvm features for the current class in a temporary file
            tmp_class_path = os.path.join(pair_svms_dir,crt_id+".tmp")
            #if not os.path.isdir(pair_svms_dir[:8]+pair_svms_dir[15:]):
            #    os.makedirs(pair_svms_dir[:8]+pair_svms_dir[15:])
            f_tmp_class = open(tmp_class_path,"w")
            #reformat the features to force labels to:
            # +1 for those of the current class
            # -1 for those of other classes
            for nfeat in norm_feats:
                libsvm_label = "-1"
                nparts = nfeat.split(" ")
                if nparts[0] == crt_id:
                    libsvm_label = "+1"
                #reformatted feature with binary label
                ref_feat = libsvm_label
                #add the actual features after the binary libsvm label
                for dim in range(1,len(nparts)):
                    ref_feat = ref_feat+" "+nparts[dim]
                #print(ref_feat)
                f_tmp_class.write(ref_feat+"\n")
            f_tmp_class.close()
            #train the SVM for the current class and parameters pair
            X, y = load_svmlight_file(tmp_class_path)
            #X_normed = Normalizer().fit_transform(X)
            clf.fit(X,y)
            svm_weights = clf.coef_
            svm_bias = clf.intercept_
            #print(svm_weights, svm_bias)
            #format the weights for writing on a single line in the output file
            out_weights = ""
            for it in range(0, svm_weights.size):
                out_weights = out_weights+" "+str(svm_weights.item(it))
            out_weights = out_weights.lstrip()
            out_bias = str(svm_bias.item(0))
            f_svm.write(out_weights+"\n") 
            f_svm.write(out_bias+"\n")
            f_svm.close()
            #remove the temporary class libsvm features file
            #print(svm_weights.shape[1])
            os.remove(tmp_class_path)
        else:
            print("SVM already created for:",crt_id_svm_path)
    with Pool() as p:
        p.map(calc_thrd, range(min_pos,max_pos))


    