import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.tensorboard import SummaryWriter  

from pathlib import Path
import pandas as pd
from models import vggface, efficient_face
# from torchsummary import summary

import  os
from tqdm import tqdm

#定义一些超参数
BATCHSIZE=395 # batchsize必须是testdata的整数倍
EPOCHES=20
LR=0.0005
numclass = 7

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('/home/wjh/ABAW5th/202303131513')
# -------------------------------------------------------------------------------------------
#	模型定义
# -------------------------------------------------------------------------------------------
# mobilenet-v2模型
def mbnet():
	model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
	# 前面的参数保持不变
	for param in model.parameters():
		param.requires_grad = True

	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc
	# print("[INFO] Model Layer:  ", summary(model, (3, 224, 224)))
	return model

# mnasnet模型
def mnasnet():
	model = models.MNASNet(alpha=1)
	# 前面的backbone保持不变
	for param in model.parameters():
		param.requires_grad = True

	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc
	# print("[INFO] Model Layer:  ", summary(model, (3, 224, 224)))
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



#  resnet 18模型
def resnet18(fc_num=256, class_num=numclass):
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	# 前面的backbone保持不变
	for param in model.parameters():
		param.requires_grad = True

	# 只是修改输出fc层，新加层是trainable
	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)

	# print("[INFO] Model Layer:  ", summary(model, (3, 224, 224)))
	return model

#  resnet 152模型
def resnet152(fc_num=256, class_num=numclass, train_all =False):
	model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
	# 前面的backbone保持不变
	for param in model.parameters():
		param.requires_grad = True

	# 只是修改输出fc层，新加层是trainable
	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)
	#  修改所有参数层
	if train_all:
		for param in model.parameters():
			param.requires_grad = False
		torch.load("./models/best_loss.pt")
	# print("[INFO] Model Layer:  ", summary(model, (3, 224, 224)))
	return model

# -------------------------------------------------------------------------------------------
#	数据加载
# -------------------------------------------------------------------------------------------
class dataload(Dataset):
    MEAN = [0.367035294117647, 0.41083294117647057, 0.5066129411764705]  
    STD = [1, 1, 1] 

    def __init__(self, df, dataset_name, img_dir) -> None:
        super(dataload).__init__()
        self.img_path_tmp = '/data03/cvpr23_competition/cvpr23_competition_data/test data/cropped_aligned/cropped_aligned/1-30-1280x720/00001.jpg'
        self.df = df
        self.img_dir = img_dir

        if dataset_name in ['train']:
            self.preprocess = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=[0.8, 1.0]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.7, saturation=0.3, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])
        elif dataset_name in ['valid','test','test']:
            self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])  

    def __getitem__(self, index) :
        
        img_name = self.df['img_name'].iloc[index]
        lable = self.df['lable'].iloc[index]

        img_path = self.img_dir / img_name
        
        if os.path.exists(img_path):
            self.img_path_tmp = img_path
            X = Image.open(img_path)
        else:
            X = Image.open(self.img_path_tmp)
        X = self.preprocess(X)


        return X, lable

    def __len__(self):
        return len(self.df)


def predict_batch(mlps, df, img_dir):
    # img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images')
    test_set = dataload(df,'test',img_dir) 
    batch_size = BATCHSIZE
    test_data = DataLoader(test_set, batch_size, shuffle=False, num_workers=8)

    pre = []
    all_result = []
    with torch.no_grad():
        for img, label in tqdm(test_data):
            img = img.to(device)
                
            for i, mlp in enumerate(mlps):
                mlp.eval()
                out = mlp(img)

                _, prediction = torch.max(out, 1)  # 按行取最大值
                pre_num = prediction.cpu().numpy()   
                pre.append(pre_num)
            arr = np.array(pre)
            pre.clear()
                
            result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
            all_result.extend(result)
    
    return all_result

def predict_sigal_model(mlp, df, img_dir):
    test_set = dataload(df,'test',img_dir) 
    batch_size = BATCHSIZE
    test_data = DataLoader(test_set, batch_size, shuffle=False, num_workers=8)

    pre = []
    all_result = []
    with torch.no_grad():
        for img, label in tqdm(test_data):
            img = img.to(device)
                
            mlp.eval()
            out = mlp(img)

            out = F.softmax(out, 1)

            max_value, prediction = torch.max(out, 1)  # 按行取最大值

            max_value = max_value.cpu().numpy()
            pre_num = prediction.cpu().numpy()

            # pos = np.where(max_value <= 0.5)
            # pre_num[pos] = 7   
            pre.append(pre_num)

            arr = np.array(pre)
            pre.clear()
            arr = arr.squeeze().tolist()

            all_result.extend(arr)
        
    return all_result
    

def add_label(mlps,index):
    csv_dir = Path('/home/wjh/ABAW5th/data_csv')
    img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/test data/cropped_aligned/cropped_aligned')

    df_test = pd.read_csv(csv_dir / 'CVPR_5th_ABAW_EXPR_test_set_sample.csv')

    # img_labels = predict_batch(mlps,df_test,img_dir)

    mlp = mlps[0]

    img_labels = predict_sigal_model(mlp,df_test,img_dir)

    dataframe = pd.DataFrame({'img_name':df_test['img_name'],'lable':img_labels})
    dataframe.to_csv('/home/wjh/ABAW5th/data_csv/predictions{}.txt'.format(index), index=False, sep=',',header=['image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other',' '])



if __name__ == '__main__':
    mlps = [mbnet().to(device),  resnet152().to(device),  densenet121().to(device), resnet18().to(device), densenet201().to(device)]

    # suffix_list = ['202303131513','202303151833','2dice1ce']
    # for j, suffix in enumerate(suffix_list):
    # for index, mlp in enumerate(mlps):
        
    #     state_saved = torch.load('/home/wjh/ABAW5th/vote_model_state/model'+str(index)+'_202303151833.pth')
        
    #     mlp.load_state_dict(state_saved)


    mlps = [resnet18().to(device)]
    for mlp in mlps:
        state_saved = torch.load('/home/wjh/ABAW5th/Best_submodel/model3_dice.pth')
        mlp.load_state_dict(state_saved)
    add_label(mlps, 9)

