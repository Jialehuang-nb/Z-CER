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
BATCHSIZE=32 # batchsize必须是testdata的整数倍
EPOCHES=500
LR=0.001
numclass = 7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Device: ", device)
writer = SummaryWriter('output')
label_map = {"0":1,"1":5,"2":6,"3":2,"4":0,"5":4,"6":3}

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

    #def __init__(self, df, dataset_name, img_dir) -> None:
    def __init__(self, df, dataset_name, abawce=False) -> None:
        super(dataload).__init__()
        self.df = df
        self.abawce = abawce
        #self.img_dir = img_dir

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
            index = -24-len(name)+5
            img_path = "data"+img_name[index:]
        lable = self.df['lable'].iloc[index]

        #img_path = img_name
        X = Image.open(img_path)
        X = self.preprocess(X)

        video_name = img_name[-len(name)-3:-len(name)]
        while len(name)<9:
             name = "0"+name
        name = video_name + name
        return X, lable,name

    def __len__(self):
        return len(self.df)

import matplotlib.pyplot as pl
from sklearn import metrics
# 相关库

def plot_matrix(y_true, y_pred, labels_name, save_dir, title=None, thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # pl.figure(figsize=(100, 100))
# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        pl.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45, fontsize=4)  # 将标签印在x轴坐标上， 并倾斜45度，字号为8
    pl.yticks(num_local, axis_labels, fontsize=4)  # 将标签印在y轴坐标上，字号为8
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    pl.savefig(save_dir)
    pl.clf()
    pl.cla()



def data_process(batch_size=32, abawce = False):
	
    #csv_dir = Path('/home/wjh/ABAW5th/data_csv')
    #img_dir = Path('')

    df_train = pd.read_csv('train.csv')
    # df_train = df_train.drop(df_train[(df_train['lable'] == -1)|((df_train['lable'] == 7))].index)
    # df_train = df_train.drop(df_train[(df_train['lable'] == -1)].index)

    # df_train = df_train[:550]

    # df_fake = pd.read_csv(csv_dir / 'fake.csv')
    # df_fake = df_fake.drop(df_fake[(df_fake['lable'] == 0)|(df_fake['lable'] == 4)|(df_fake['lable'] == 5)].index)
    # df_fake = df_fake.sample(frac=0.5)

    # df_train = pd.concat([df_train, df_fake])
    # df_train = df_train[:128]

    # sample

    # condition0 = df_train['lable'] == 0 
    # subset0 = df_train[condition0].sample(frac=0.45)
    # condition4 = df_train['lable'] == 4 
    # subset4 = df_train[condition4].sample(frac=0.65)
    # condition5 = df_train['lable'] == 5 
    # subset5 = df_train[condition5].sample(frac=0.65)

    # subset_other = df_train.drop(df_train[condition0|condition4|condition5].index)

    # # | df_train['lable'] == 4 | df_train['lable'] == 5

    # df_train = pd.concat([subset_other,subset5,subset4,subset0])
    # print(df_train)

    df_valid = pd.read_csv('test.csv')
    # df_valid = df_valid.drop(df_valid[(df_valid['lable'] == -1)|((df_valid['lable'] == 7))].index)
    df_valid = df_valid.sample(int(len(df_valid)/BATCHSIZE)*BATCHSIZE)
    # df_valid = df_valid[:128]
    
    # df_train = pd.concat([df_train, df_valid])

    label_conuts = df_train['lable'].value_counts()
    # print(label_conuts[3])
    print(label_conuts)

    #train_set = dataload(df_train,'train',img_dir)
    #valid_set = dataload(df_valid,'valid',img_dir) 
    train_set = dataload(df_train,'train')
    valid_set = dataload(df_valid,'valid')

    abawce=False
    if abawce:
         df_abaw = pd.read_csv('abaw.csv')
         train_set = dataload(df_abaw,'train',True)
         valid_set = dataload(df_abaw,'valid',True)

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
		# 获取每个批次的大小 N
		N = targets.size()[0]
		# 平滑变量
		smooth = 1
		# 将宽高 reshape 到同一纬度
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		# 计算交集
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		# 计算一个批次中平均每张图的损失
		loss = 1 - N_dice_eff.sum() / N
		return loss


class MultiClassDiceLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=None, **kwargs):
		super(MultiClassDiceLoss, self).__init__()
		self.weight = weight
		self.ignore_index = ignore_index
		self.kwargs = kwargs
	
	def forward(self, input, target):
		"""
			input tesor of shape = (N, C, H, W)
			target tensor of shape = (N, H, W)
		"""
		# 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
		nclass = input.shape[1]
		target = F.one_hot(target.long(), nclass)

		assert input.shape == target.shape, "predict & target shape do not match"
		
		binaryDiceLoss = BinaryDiceLoss()
		total_loss = 0
		
		# 归一化输出
		logits = F.softmax(input, dim=1)
		C = target.shape[1]
		
		# 遍历 channel，得到每个类别的二分类 DiceLoss
		for i in range(C):
			dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
			total_loss += dice_loss
		
		# 每个类别的平均 dice_loss
		return total_loss / C



class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class Focal_loss(nn.Module):
    def __init__(self, weight=None, gamma=0):
        super(Focal_loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = 1e-8
    def forward(self, predict, target):
        # print(predict)
        # print(target)
        if self.weight!=None:
            weights = self.weight.unsqueeze(0).unsqueeze(1).repeat(predict.shape[0], predict.shape[2], 1)
        target_onehot = F.one_hot(target.long(), predict.shape[1]) 
        if self.weight!=None:
            weights = torch.sum(target_onehot * weights, -1)
        input_soft = F.softmax(predict, dim=1)
        # print(input_soft.transpose(1, 0))
        # print(input_soft.unsqueeze(0))
        # print(target_onehot)
        probs = torch.sum(input_soft.transpose(2, 1) * target_onehot, -1).clamp(min=0.001, max=0.999)#此处一定要限制范围，否则会出现loss为Nan的现象。
        focal_weight = (1 + self.eps - probs) ** self.gamma
        if self.weight!=None:
            return torch.sum(-torch.log(probs) * weights * focal_weight) / torch.sum(weights)
        else:
            return torch.mean(-torch.log(probs) * focal_weight)
        
        
class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
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

def writeresult(pred,name):
    for i in range(len(pred)):
        p = pred[i]
        n = name[i]+","
        result = []
        f = open("CVPR_6th_ABAW_CE_test_set_sample.txt",'r+')
        all = f.readlines()
        for idx,video in enumerate(all):
            if video == n:
                video = video + str(label_map[pred])+'\n'
            result.append(video)
        f.writelines(result)
        f.close()
               
# -------------------------------------------------------------------------------------------
#	集成表决
# -------------------------------------------------------------------------------------------
def process(mlps, trainloader, testloader, label_conuts):
    # optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)
    optimizer = torch.optim.Adam([{"params":filter(lambda p: p.requires_grad, mlp.parameters())} for mlp in mlps], lr=LR)

    weight=torch.from_numpy(np.array([1/label_conuts[0], 1/label_conuts[1], 1/label_conuts[2], 1/label_conuts[3], 1/label_conuts[4], 1/label_conuts[5], 1/label_conuts[6]])).float() 
    cls_num = [label_conuts[0], label_conuts[1], label_conuts[2], label_conuts[3], label_conuts[4], label_conuts[5], label_conuts[6]]
    # loss_function = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(device)
    Bal_CE_loss_function = Bal_CE_loss(cls_num,1.0).to(device)
    CE_loss_function = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(device)
    dice_loss_function = MultiClassDiceLoss().to(device)
    # FL_loss_function = Focal_loss(gamma=0.5).to(device)
    best_f1 = -1
    
    for ep in range(EPOCHES):
        print("Epoch: {}/{}".format(ep + 1, EPOCHES))
        print("[INFO] Begin to train")
        '''
        mlps_pred_train = [[] for i in range(len(mlps))]
        label_train=[]
        for img, label, _ in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            label_one_hot = F.one_hot(label,num_classes = 7).float()
            label_train.append(label)
            optimizer.zero_grad()  # 网络清除梯度
            for i, mlp in enumerate(mlps):
                mlp.train()
                out = mlp(img)
                mlps_pred_train[i].append(out.to('cpu'))
                
                # FLloss = FL_loss_function(out.unsqueeze(0), F.one_hot(label,num_classes = 7))
                # CEloss = CE_loss_function(out, label)
                Diceloss = dice_loss_function(out, label)

                BalCEloss = Bal_CE_loss_function(out, label_one_hot)
                # print('CEloss = %f, Diceloss = %f ' %(CEloss.item(), Diceloss.item()))

                # loss = CEloss + 1.5 * Diceloss
                loss = BalCEloss + 1.5 * Diceloss
                # loss = FLloss + Diceloss
                # loss = CEloss
                loss.backward()  # 网络们获得梯度
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
            # print(torch.from_numpy(mlp_train_c).float())
            mlp_loss = CE_loss_function(torch.from_numpy(mlp_train_c).float().to(device),torch.from_numpy(label_train).to(device)) + dice_loss_function(torch.from_numpy(mlp_train_c).float(),torch.from_numpy(label_train))
            # mlp_loss = CE_loss_function(torch.from_numpy(mlp_train).float(),torch.from_numpy(label_train).float()) 
            if ep%20 == 0:
                plot_matrix(y_true=label_train, y_pred=mlp_train, labels_name=[0,1,2,3,4,5,6], save_dir='/home/wjh/ABAW5th/confusion_matrix/train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])
            
            writer.add_scalar('train_loss/'+str(idx),mlp_loss.item(),ep)
            writer.add_scalar('train_acc/'+str(idx),mlp_acc,ep)
            writer.add_scalar('train_f1/'+str(idx),mlp_f1,ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item())) 
        '''  
        # for i, mlp in enumerate(mlps):
        #     torch.save(mlp.state_dict(), '/home/wjh/ABAW5th/vote_model_state/model'+str(i)+'_202303151833.pth')
        pre = []
        mlps_pred_valid = [[] for i in range(len(mlps))]
        label_valid = []
        vote_valid = []
        
        print("[INFO] Begin to valid")
        with torch.no_grad():
            for img, label,name in tqdm(testloader):
                img = img.to(device)
                label_valid.append(label)
                for i, mlp in enumerate(mlps):
                    mlp.eval()
                    out = mlp(img)
                    mlps_pred_valid[i].append(out.to('cpu'))

                    _, prediction = torch.max(out, 1)  # 按行取最大值
                    pre_num = prediction.cpu().numpy()   
                    pre.append(pre_num)
                arr = np.array(pre)
                pre.clear()
                
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
                # for i_res in range(BATCHSIZE):
                #     for j_res in [1,5,3,2]:
                #         if j_res in arr[:, i_res]:
                #             result[i_res] = j_res
                vote_valid.extend(result)
    
        label_valid = torch.cat(label_valid)
        label_valid = label_valid.detach().numpy() 
        vote_valid = np.array(vote_valid)
        writeresult(vote_valid,name)
        vote_acc = accuracy_score(vote_valid, label_valid)
        vote_f1 = f1_score(vote_valid, label_valid, average='macro')
        # vote_loss = CE_loss_function(torch.from_numpy(vote_valid).unsqueeze(0).float(),torch.from_numpy(label_valid).unsqueeze(0).float()) + dice_loss_function(F.one_hot(torch.from_numpy(vote_valid),num_classes = numclass).float(),torch.from_numpy(label_valid))
        # mlp_loss = CE_loss_function(torch.from_numpy(mlp_train_c).float().to(device),torch.from_numpy(label_train).to(device)) + dice_loss_function(torch.from_numpy(mlp_train_c).float(),torch.from_numpy(label_train))
        
        # vote_loss = CE_loss_function(torch.from_numpy(vote_valid).float(),torch.from_numpy(label_valid).float())
        writer.add_scalar('vote_acc',vote_acc,ep)
        writer.add_scalar('vote_f1',vote_f1,ep)
        # writer.add_scalar('vote_loss',vote_loss.item(),ep)
        if ep%20 == 0:
            plot_matrix(y_true=label_valid, y_pred=vote_valid, labels_name=[0,1,2,3,4,5,6], save_dir='output/vote-pic-{}.png'.format(ep + 1), title='vote-pic-{}.png'.format(ep + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])

        
        if vote_f1 > best_f1:
            for i, mlp in enumerate(mlps):
                torch.save(mlp.state_dict(), 'model/model'+str(i)+'.pth')

        print("epoch:" + str(ep + 1) + "\n集成模型的acc=" + str(vote_acc) + ", f1=" + str(vote_f1) )    
        for idx, mlp_valid in enumerate(mlps_pred_valid):
            mlp_valid = torch.cat(mlp_valid)
            mlp_valid = mlp_valid.detach().numpy()
            mlp_valid_c = mlp_valid
            mlp_valid = mlp_valid.argmax(axis=1)
            mlp_acc = accuracy_score(mlp_valid, label_valid)
            mlp_f1 = f1_score(mlp_valid, label_valid, average='macro')
            mlp_loss = CE_loss_function(torch.from_numpy(mlp_valid_c).float().to(device),torch.from_numpy(label_valid).to(device)) + dice_loss_function(torch.from_numpy(mlp_valid_c).float(),torch.from_numpy(label_valid))
            
            # mlp_loss = CE_loss_function(torch.from_numpy(mlp_valid).float(),torch.from_numpy(label_valid).float()) 
            if ep%20 == 0:
                plot_matrix(y_true=label_valid, y_pred=mlp_valid, labels_name=[0,1,2,3,4,5,6], save_dir='output/valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised'])

            writer.add_scalar('valid_loss/'+str(idx),mlp_loss.item(),ep)
            writer.add_scalar('valid_acc/'+str(idx),mlp_acc,ep)
            writer.add_scalar('valid_f1/'+str(idx),mlp_f1,ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item()))


def predict_batch(mlps, df):
    img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images')
    fake_set = dataload(df,'fake',img_dir) 
    batch_size = BATCHSIZE
    fake_data = DataLoader(fake_set, batch_size, shuffle=True, num_workers=8)

    pre = []
    all_result = []
    with torch.no_grad():
        for img, label in tqdm(fake_data):
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
            for i in range(BATCHSIZE):
                for j in [1,5,3,2]:
                    if j in arr[:, i]:
                        result[i] = j
                
            all_result.extend(result)
    
    return all_result
    


# def predict(mlps, img_dir):

# preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.367035294117647,0.41083294117647057,0.5066129411764705], (1, 1, 1))
#             ])

#     img = Image.open(img_dir)
#     img = preprocess(img)
#     img = img.unsqueeze(0)

#     pre = []

#     with torch.no_grad():
#         img = img.to(device)
#         for i, mlp in enumerate(mlps):
#             mlp.eval()
#             out = mlp(img)

#             _, prediction = torch.max(out, 1)  # 按行取最大值
#             pre_num = prediction.cpu().numpy()   
#             pre.append(pre_num)
#         arr = np.array(pre)
#         pre.clear()
                
#         result = Counter(arr[:,0]).most_common(1)[0][0]
#     return result

def add_fake_label(mlps):
    #img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images')

    df_train = pd.read_csv('train.csv')
    df_train = df_train.drop(df_train[(df_train['lable'] == 0)|((df_train['lable'] == 1))|((df_train['lable'] == 2))|((df_train['lable'] == 3))|((df_train['lable'] == 4))|((df_train['lable'] == 5))|((df_train['lable'] == 6))].index)
    # df_train = df_train[:550]

    df_valid = pd.read_csv('valid.csv')
    df_valid = df_valid.drop(df_valid[(df_valid['lable'] == 0)|((df_valid['lable'] == 1))|((df_valid['lable'] == 2))|((df_valid['lable'] == 3))|((df_valid['lable'] == 4))|((df_valid['lable'] == 5))|((df_valid['lable'] == 6))].index)
    # df_valid = df_valid[:550]

    df_need_fake = pd.concat([df_train,df_valid])
    dataframe = pd.DataFrame({'img_name':df_need_fake['img_name'],'lable':df_need_fake['lable']})
    df_fake = dataframe[:940864]

    img_labels = predict_batch(mlps, df_fake)

    # for img_name in tqdm(df_need_fake['img_name']):
    #     label = predict(mlps, img_dir / img_name)
    #     img_labels.append(label)
    #     img_names.append(img_name)

    dataframe = pd.DataFrame({'img_name':df_fake['img_name'],'lable':img_labels})
    dataframe.to_csv('fake.csv',index=False,sep=',')



if __name__ == '__main__':
    # mlps = [mbnet().to(device),  resnet152().to(device),  densenet121().to(device), resnet18().to(device), vggface().to(device),densenet201().to(device)]
    mlps = [mbnet().to(device),  resnet152().to(device),  densenet121().to(device), resnet18().to(device), densenet201().to(device)]

    for index, mlp in enumerate(mlps):
        state_saved = torch.load('vote_model_state/20240312BalCE/model'+str(index)+'.pth')
        mlp.load_state_dict(state_saved)
    
    # label = predict(mlps, '/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images/cropped_aligned/6-30-1920x1080_left/00027.jpg')
    # print(label)
    train_data, valid_data, label_conuts = data_process()
    process(mlps=mlps, trainloader=train_data , testloader=valid_data, label_conuts=label_conuts)

    # add_fake_label(mlps)

