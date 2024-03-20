import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
from torchvision import models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.tensorboard import SummaryWriter  


import pandas as pd



import  os
from tqdm import tqdm


BATCHSIZE=32 # batchsize必须是testdata的整数倍
EPOCHES=500
LR=0.001
numclass = 7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Device: ", device)
writer = SummaryWriter('output')



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

    def __init__(self, df, dataset_name, img_dir) -> None:
        super(dataload).__init__()
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
        elif dataset_name in ['valid','test','fake']:
            self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])  

    def __getitem__(self, index) :
        
        img_name = self.df['img_name'].iloc[index]
        lable = self.df['label'].iloc[index]

        X = Image.open(img_name)
        X = self.preprocess(X)

        return X, lable

    def __len__(self):
        return len(self.df)

import matplotlib.pyplot as pl
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from collections import Counter
import numpy as np


def plot_matrix(y_true, y_pred, labels_name, save_dir, title=None, thresh=0.8, axis_labels=None):

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None) 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  


    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  


    if title is not None:
        pl.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45, fontsize=4)  
    pl.yticks(num_local, axis_labels, fontsize=4)  
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  

    pl.savefig(save_dir)
    pl.clf()
    pl.cla()


def data_process(batch_size=32, abawce = False):
	
    df_train = pd.read_csv('train.csv')
    df_valid = pd.read_csv('test.csv')
    df_valid = df_valid.sample(int(len(df_valid)/BATCHSIZE)*BATCHSIZE)

    label_conuts = df_train['label'].value_counts()
    print(label_conuts)
    train_set = dataload(df_train,'train')
    valid_set = dataload(df_valid,'valid')

    batch_size = BATCHSIZE
    train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=8)

    train_data_size = len(df_train)
    valid_data_size = len(df_valid)

    print("[INFO] Train data / Test data number: ", train_data_size, valid_data_size)
    return train_data, valid_data, label_conuts

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()
    def forward(self, input, target):
        nclass = input.shape[1]
        target = F.one_hot(target.long(), nclass)
        assert input.shape == target.shape, "predict & target shape do not match"
        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0
        logits = F.softmax(input, dim=1)
        C = target.shape[1]
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss
        return total_loss / C

class Bal_CE_loss(nn.Module):
    def __init__(self, cls_num, bal_tau):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(cls_num)
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = bal_tau * prior

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior.to(x.device)
        x = x + prior
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def process(mlps, trainloader, testloader, label_conuts):
    optimizer = torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, mlp.parameters())} for mlp in mlps], lr=LR)
    weight = torch.from_numpy(np.array([1/label_conuts[i] for i in range(len(label_conuts))])).float()
    cls_num = label_conuts
    Bal_CE_loss_function = Bal_CE_loss(cls_num, 1.0).to(device)
    CE_loss_function = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(device)
    dice_loss_function = MultiClassDiceLoss().to(device)
    best_f1 = -1

    for ep in range(EPOCHES):
        print("Epoch: {}/{}".format(ep + 1, EPOCHES))
        print("[INFO] Begin to train")

        mlps_pred_train = [[] for _ in range(len(mlps))]
        label_train = []
        for img, label, _ in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            label_one_hot = F.one_hot(label, num_classes=7).float()
            label_train.append(label)
            optimizer.zero_grad()
            for i, mlp in enumerate(mlps):
                mlp.train()
                out = mlp(img)
                mlps_pred_train[i].append(out.to('cpu'))
                Diceloss = dice_loss_function(out, label)
                BalCEloss = Bal_CE_loss_function(out, label_one_hot)
                loss = BalCEloss + 1.5 * Diceloss
                loss.backward()
            optimizer.step()

        label_train = torch.cat(label_train)
        label_train = label_train.cpu().detach().numpy()

        for idx, mlp_train in enumerate(mlps_pred_train):
            mlp_train = torch.cat(mlp_train)
            mlp_train = mlp_train.detach().numpy()
            mlp_train_c = mlp_train
            mlp_train = mlp_train.argmax(axis=1)
            mlp_acc = accuracy_score(mlp_train, label_train)
            mlp_f1 = f1_score(mlp_train, label_train, average='macro')
            mlp_loss = CE_loss_function(torch.from_numpy(mlp_train_c).float().to(device), torch.from_numpy(label_train).to(device))
            if ep % 20 == 0:
                plot_matrix(y_true=label_train, y_pred=mlp_train, labels_name=[0, 1, 2, 3, 4, 5, 6], save_dir='/home/wjh/ABAW5th/confusion_matrix/train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])

            writer.add_scalar('train_loss/' + str(idx), mlp_loss.item(), ep)
            writer.add_scalar('train_acc/' + str(idx), mlp_acc, ep)
            writer.add_scalar('train_f1/' + str(idx), mlp_f1, ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item()))

        pre = []
        mlps_pred_valid = [[] for _ in range(len(mlps))]
        label_valid = []
        vote_valid = []
        
        print("[INFO] Begin to valid")
        with torch.no_grad():
            for img, label, name in tqdm(testloader):
                img = img.to(device)
                label_valid.append(label)
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
                vote_valid.extend(result)

        label_valid = torch.cat(label_valid)
        label_valid = label_valid.detach().numpy()
        vote_valid = np.array(vote_valid)
        vote_acc = accuracy_score(vote_valid, label_valid)
        vote_f1 = f1_score(vote_valid, label_valid, average='macro')

        writer.add_scalar('vote_acc', vote_acc, ep)
        writer.add_scalar('vote_f1', vote_f1, ep)

        if ep % 20 == 0:
            plot_matrix(y_true=label_valid, y_pred=vote_valid, labels_name=[0, 1, 2, 3, 4, 5, 6], save_dir='output/vote-pic-{}.png'.format(ep + 1), title='vote-pic-{}.png'.format(ep + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])

        if vote_f1 > best_f1:
            for i, mlp in enumerate(mlps):
                torch.save(mlp.state_dict(), 'model/model'+str(i)+'.pth')

        print("epoch:" + str(ep + 1) + "\n集成模型的acc=" + str(vote_acc) + ", f1=" + str(vote_f1))
        for idx, mlp_valid in enumerate(mlps_pred_valid):
            mlp_valid = torch.cat(mlp_valid)
            mlp_valid = mlp_valid.detach().numpy()
            mlp_valid_c = mlp_valid
            mlp_valid = mlp_valid.argmax(axis=1)
            mlp_acc = accuracy_score(mlp_valid, label_valid)
            mlp_f1 = f1_score(mlp_valid, label_valid, average='macro')
            mlp_loss = CE_loss_function(torch.from_numpy(mlp_valid_c).float().to(device), torch.from_numpy(label_valid).to(device))

            if ep % 20 == 0:
                plot_matrix(y_true=label_valid, y_pred=mlp_valid, labels_name=[0, 1, 2, 3, 4, 5, 6], save_dir='output/valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])

            writer.add_scalar('valid_loss/' + str(idx), mlp_loss.item(), ep)
            writer.add_scalar('valid_acc/' + str(idx), mlp_acc, ep)
            writer.add_scalar('valid_f1/' + str(idx), mlp_f1, ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item()))

if __name__ == '__main__':
    mlps = [mbnet().to(device), resnet152().to(device), densenet121().to(device), resnet18().to(device), densenet201().to(device)]
    train_data, valid_data, label_conuts = data_process()
    process(mlps=mlps, trainloader=train_data, testloader=valid_data, label_conuts=label_conuts)
