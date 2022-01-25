from __future__ import division
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.cuda as tc
import torch.utils.data.distributed
from configparser import ConfigParser
import sys, os, warnings
import numpy as np
from MyImageFolder import ImagesListFileFolder
import gc
import shutil

def get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path):
    import re
    datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
    for line in datasets_mean_std_file:
        line = line.strip().split(':')
        dataset_name = line[0]
        dataset_stat = line[1]
        if dataset_name == normalization_dataset_name:
            dataset_stat = dataset_stat.split(';')
            dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
            dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
            return dataset_mean, dataset_std
    print('Invalid normalization dataset name')
    sys.exit(-1)


if len(sys.argv) != 2:
    print('Arguments : general_config')
    sys.exit(-1)

if not os.path.exists(sys.argv[1]):
    print('No configuration file found in the specified path')
    sys.exit(-1)

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
nb_classes_per_passe = 1+int(cp['last_class'])#int(sys.argv[2])
# reading parameters
num_workers = int(cp['num_workers'])
batch_size = int(cp['batch_size'])
first_batch_model_load_path = cp['first_batch_model_load_path']
gpu = int(cp['gpu'])
images_list_dir = cp['images_list_dir']
destination_dir = cp['destination_dir']
normalization_dataset_name = cp['normalization_dataset_name']
P = int(cp['P'])
S = int(cp['S'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
first_class = int(cp['first_class'])
last_class = int(cp['last_class'])
dataset_mean, dataset_std = get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

print('normalization dataset name = ' + str(normalization_dataset_name))
print('dataset mean = ' + str(dataset_mean))
print('dataset std = ' + str(dataset_std))

normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

print("Number of workers = " + str(num_workers))
print("Batch size = " + str(batch_size))
print("Running on gpu " + str(gpu))


num_classes = P
model = models.resnet18(pretrained=False, num_classes=num_classes)

model_load_path = first_batch_model_load_path

print('Loading saved model from ' + model_load_path)

state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
if '.pth' in model_load_path:
    state_without_model = {clef[6:]:state['state_dict'][clef] for clef in state['state_dict'].keys()}
    #model.load_state_dict(state['state_dict'])
    model.fc = nn.Linear(512, 1000)
    model.load_state_dict(state_without_model)
else:
    model.load_state_dict(state['state_dict'])

features_extractor = nn.Sequential(*list(model.children())[:-1])
model.eval()
features_extractor.eval()
features_extractor = features_extractor.cuda(gpu)
data_types = ['train', 'test']
for data_type in data_types:
    images_list = os.path.join(images_list_dir,data_type+'.lst')
    print('Loading list file from ' + images_list)

    data_type_destination_dir = os.path.join(destination_dir, data_type)
    try:
        print('cleaning',data_type_destination_dir,'...')
        shutil.rmtree(data_type_destination_dir)
    except:
        pass
    os.makedirs(data_type_destination_dir,exist_ok=True)
    print(data_type_destination_dir,'cleaned!')
    dataset = ImagesListFileFolder(
        images_list, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ]),
            return_path=True,
            range_classes=None,
            )

    print(data_type + "-set size = " + str(len(dataset)))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    
    features_names = {}
    file_names = {}
    last_class = -1
    for data in loader:
        (inputs, labels), _ = data
        inputs = inputs.cuda(gpu)
        features = features_extractor(inputs)
        lablist=list(labels.data.cpu().numpy().squeeze())
        featlist=list(features.data.cpu().numpy().squeeze())
        for i in range(len(lablist)):
            cu_class = lablist[i]
            if cu_class!=last_class:
                last_class=cu_class
                print('beginning of extraction of class',last_class)
            with open(os.path.join(data_type_destination_dir,str(cu_class)), 'a') as features_out:
                features_out.write(str(' '.join([str(e) for e in list(featlist[i])])) + '\n')

