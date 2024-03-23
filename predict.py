import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image


from torch.utils.tensorboard import SummaryWriter  

import pandas as pd

import  os
from tqdm import tqdm


BATCHSIZE=1
#EPOCHES=500
EPOCHES=1
LR=0.001
numclass = 7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Device: ", device)
writer = SummaryWriter('output')
label_map = ["1","5","6","2","0","4","3"]


def mbnet():
	model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True
	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc

	return model

def mnasnet():
	model = models.MNASNet(alpha=1)
	for param in model.parameters():
		param.requires_grad = True

	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc
	return model

def densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
                        nn.Linear(1024, 500),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(500, numclass)
                        )

    model.classifier = classifier

    return model

def densenet201():
    model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
                        nn.Linear(1920, 500),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(500, numclass)
                        )

    model.classifier = classifier

    return model


def resnet18(fc_num=256, class_num=numclass):
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True

	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)

	return model


def resnet152(fc_num=256, class_num=numclass, train_all =False):
	model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True

	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)

	if train_all:
		for param in model.parameters():
			param.requires_grad = False
		torch.load("./models/best_loss.pt")

	return model


class dataload(Dataset):
    MEAN = [0.367035294117647, 0.41083294117647057, 0.5066129411764705]  
    STD = [1, 1, 1] 


    def __init__(self, df, dataset_name, abawce=False) -> None:
        super(dataload).__init__()
        self.df = df
        self.abawce = abawce

        if dataset_name in ['train']:
            self.preprocess = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=[0.8, 1.0]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.7, saturation=0.3, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])
        elif dataset_name in ['valid','test','fake']:
            self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])  

    def __getitem__(self, index) :
        
        img_name = self.df['img_name'].iloc[index]
        name = os.path.basename(img_name)
        img_path = img_name[-7-len(name):]
        if self.abawce:
            
            p = -24-len(name)+5
            
            img_path = "data/images_aligned/"+img_name[p:]

        X = Image.open(img_path)
        X = self.preprocess(X)

        video_name = img_name[-len(name)-3:-len(name)]
        
        while len(name)<9:
            name = "0"+name
        name = video_name + name
        return X, name

    def __len__(self):
        return len(self.df)


def data_process():
	
    df_abaw = pd.read_csv('abaw.csv')
    valid_set = dataload(df_abaw,'valid',True)
    batch_size = BATCHSIZE
    valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=8)

    return  valid_data


def writeresult(pred,name):
    for i in tqdm(range(len(pred))):
        p = pred[i]
        n = name[i]+","+'\n'
        result = []
        f = open("CVPR_6th_ABAW_CE_test_set_sample.txt",'r+')
        all = f.readlines()
        for idx,video in enumerate(all):
            if video == n:
                video = video[:-1] + label_map[pred[i]]+'\n'
            result.append(video)
        f.close()
        f = open("CVPR_6th_ABAW_CE_test_set_sample.txt",'r+')
        f.writelines(result)
        f.close()


def process(mlps, testloader):

    pre = []
    mlps_pred_valid = [[] for i in range(len(mlps))]

    with torch.no_grad():
        for img, name in tqdm(testloader):
            img = img.to(device)

            for i, mlp in enumerate(mlps):
                mlp.eval()
                out = mlp(img)
                mlps_pred_valid[i].append(out.to('cpu'))

                _, prediction = torch.max(out, 1) 
                pre_num = prediction.cpu().numpy()   
                pre.append(pre_num)
            arr = np.array(pre)
            pre.clear()

            result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]

            writeresult(result,name)
                
if __name__ == '__main__':
    mlps = [mbnet().to(device),  resnet152().to(device),  densenet121().to(device), resnet18().to(device), densenet201().to(device)]

    for index, mlp in enumerate(mlps):
        print(index)
        state_saved = torch.load('vote_model_state/'+str(index)+'.pth',map_location='cuda:0')
        mlp.load_state_dict(state_saved)

    valid_data = data_process()
    process(mlps=mlps, testloader=valid_data)



