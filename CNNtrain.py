import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path
import argparse
from torch.autograd import Variable
from cnn import *
from CNNconfig import *
from tqdm import tqdm
from dataloader import DataLoader
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import copy
import seaborn as sns
sns.set()
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',   type=str,     default = config['dataroot'], 
                    help = 'path to dataset')
parser.add_argument('--ckptroot',   type=str,     default = config['ckptroot'], 
                    help = 'path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr',         type = float, default = config['lr'], 
                    help = 'learning rate')
parser.add_argument('--wd',         type = float, default = config['wd'], 
                    help = 'weight decay')
parser.add_argument('--epochs',     type = int,   default = config['epochs'],
                    help = 'number of epochs to train')

parser.add_argument('--batch_size', type = int,   default = config['batch_size'], 
                    help = 'training set input batch size')
parser.add_argument('--input_size', type = int,   default = config['input_size'], 
                    help = 'size of input images')

# loading set 
parser.add_argument('--resume',     type = bool,  default = config['resume'],
                    help = 'whether training from ckpt')
parser.add_argument('--is_gpu',     type = bool,  default = config['is_gpu'],
                    help = 'whether training using GPU')

# parse the arguments
arg = parser.parse_args()
def main():

    # transform on images

    print("==> Data Augmentation ...")
    # exit()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    class minist_dataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, data,labels):

            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            re = normalize(self.data[idx]).reshape(1,28,28)
            re.astype(np.float32)
            re_label = self.labels[idx]

            return [re,re_label]

    # Loading -_- -_- -_- -_- -_-

    print("==> Preparing dataset ...")


    dataloader = DataLoader(Xtrainpath='data/train-images-idx3-ubyte.gz',
                            Ytrainpath='data/train-labels-idx1-ubyte.gz',
                            Xtestpath='data/t10k-images-idx3-ubyte.gz',
                            Ytestpath='data/t10k-labels-idx1-ubyte.gz')
    x_train, y_train, x_test, y_test = dataloader.load_data()

    trainset = minist_dataset(x_train,y_train)
    testset = minist_dataset(x_test,y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=arg.batch_size, shuffle=True, num_workers=2)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=arg.batch_size, shuffle=False, num_workers=2)

    # Initialize model

    print("==> Initialize CNN model ...")

    start_epoch = 0

    # resume training from the last time
    if arg.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint ...')
        assert os.path.isdir(
            './checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(arg.ckptroot)
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
    else:
        # start over
        print('==> Building new CNN model ...')
        net = CNN()
    for name, param in net.state_dict().items():
        print('Layer Name:',name)
        # print('Param:',param)
        # mean = torch.mean(param)
        # std = torch.std(param)
        # print('Mean Value:',mean)
        # print('Std Value:',std)
    # exit()
        # name: str
        # param: Tensor
    # To cuda
    if arg.is_gpu:
        net = net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=arg.wd)


    def calculate_accuracy(loader,is_gpu,criterion):
        correct = 0.
        total = 0.
        running_loss = 0.0
        for i,data in enumerate(loader,0):
            images, labels = data
            images, labels = Variable(images), Variable(labels)
            if is_gpu:
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.long()
            loss = criterion(outputs, labels)
            running_loss += loss.data
            total += labels.size(0)
            correct += (predicted == labels).sum()

        return correct / total,running_loss


    print("==> Start training ...")
    trainAcc = []
    testAcc = []
    trainLoss = []
    testLoss = []
    for epoch in tqdm(range(start_epoch, arg.epochs + start_epoch)):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if arg.is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            inputs, labels = Variable(inputs), Variable(labels)
            # start to train
            optimizer.zero_grad()
            inputs = inputs.float()
            outputs = net(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()

            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000
            optimizer.step()
            running_loss += loss.data
        running_loss /= len(trainloader)
        # print(f'running_loss:{running_loss}')

        # compute acc
        train_accuracy,train_loss = calculate_accuracy(trainloader, arg.is_gpu,criterion)
        test_accuracy,test_loss = calculate_accuracy(testloader, arg.is_gpu,criterion)
        
        train_loss = train_loss/len(trainloader)
        test_loss = test_loss/len(testloader)

        trainLoss.append(train_loss)
        testLoss.append(test_loss)
        trainAcc.append(train_accuracy)
        testAcc.append(test_accuracy)

        print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
            epoch+1, running_loss, train_accuracy, test_accuracy))

        # save model >>>>> no save !
        # if epoch % 10 == 0:
        #     print('==> Saving model ...')
        #     state = {
        #         'net': net.module if arg.is_gpu else net,
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/ckpt.t7')

    print('==> Finished Training ...')
    # epochs = np.arange(arg.epochs)
    # plt.figure(1)
    # plt.plot(epochs,trainLoss,'blue',label='Train Loss')
    # plt.plot(epochs,testLoss,'red',label='Test Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.savefig('CNN_Loss.png')

    # plt.figure(2)
    # plt.plot(epochs,trainAcc,'blue',label='Train Accuracy')
    # plt.plot(epochs,testAcc,'red',label='Test Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.savefig('CNN_ACC.png')
    # print('>>> figures are saving <<<')
    # plt.show()
    return trainLoss,testLoss,trainAcc,testAcc

if __name__ == '__main__':
    trainloss,testloss,trainacc,testacc = main()
    # exit()
    N = 50
    Runs = 3
    avg_trainloss = np.zeros(N)
    avg_testloss = np.zeros(N)
    avg_trainacc = np.zeros(N)
    avg_testacc = np.zeros(N)

    _std_trainloss = []
    _std_testloss = []
    _std_trainacc = []
    _std_testacc = []
    for i in range(Runs):
        trainloss,testloss,trainacc,testacc = main()
        avg_trainloss += np.array(trainloss)/Runs
        avg_testloss += np.array(testloss)/Runs
        avg_trainacc += np.array(trainacc)/Runs
        avg_testacc += np.array(testacc)/Runs


        _std_trainloss.append(trainloss)
        _std_testloss.append(testloss)
        _std_trainacc.append(trainacc)
        _std_testacc.append(testacc)

        _std_trainloss = np.array(_std_trainloss)
        _std_testloss = np.array(_std_testloss)
        _std_trainacc = np.array(_std_trainacc)
        _std_testacc = np.array(_std_testacc)

    std_trainloss = np.std(_std_trainloss,axis=0)
    std_testloss = np.std(_std_testloss,axis=0)
    std_trainacc = np.std(_std_trainacc,axis=0)
    std_testacc = np.std(_std_testacc,axis=0)




    ### figure labeling -> histogram labeling ###
    def autolabel(rects):

        for rect in rects:
            height = round(rect.get_height(),5)
            tmp = (rect.get_x() + rect.get_width() / 2, height)

            ax.annotate('{}'.format(height),
                        xy=tmp,
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # labels = ['Run 1', 'Run 2', 'Run 3']
    # x_pos = np.arange(len(labels))
    
    
    # ### plot ACC mean and std ###
    # fig, ax = plt.subplots()

    # rects1 = ax.bar(x_pos-0.35/2, avg_trainloss,
    #         yerr=std_trainloss,
    #         width=0.35,
    #         ecolor='black',
    #         capsize=10,
    #         color='green',
    #         label='Train')
    # # Save the figure and show
    # # plt.savefig('bar_plot_with_error_bars.png')

    # rects2 = ax.bar(x_pos+0.35/2, avg_testloss,
    #         yerr=std_testloss,
    #         width=0.35,
    #         ecolor='black',
    #         capsize=10,
    #         color='red',
    #         label='Test')
    # ax.set_ylabel('Loss')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(labels)
    # ax.set_title('Mean Loss for 3 Runs')
    # ax.yaxis.grid(True)
    # autolabel(rects1)
    # autolabel(rects2)

    # # Save the figure and show
    # plt.tight_layout()



    # plt.show()

    ### Plot boxfigure ###

    fig, ax = plt.subplots()

    rects1 = ax.bar(x_pos-0.35/2, avg_trainacc,
            yerr=std_trainacc,
            width=0.35,
            ecolor='black',
            capsize=10,
            color='green',
            label='Train')
    # Save the figure and show
    # plt.savefig('bar_plot_with_error_bars.png')

    rects2 = ax.bar(x_pos+0.35/2, avg_testacc,
            yerr=std_testacc,
            width=0.35,
            ecolor='black',
            capsize=10,
            color='red',
            label='Test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Mean accuracy for 3 Runs')
    ax.yaxis.grid(True)
    autolabel(rects1)
    autolabel(rects2)

    # Save the figure and show
    plt.tight_layout()
    plt.show()
    # epochs = np.arange(N)
    # plt.figure(1)
    # plt.plot(epochs,avg_trainloss,'blue',label='Train Loss mean')
    # plt.plot(epochs,avg_testloss,'red',label='Test Loss mean')
    # plt.title('Mean Loss vs. Iterations')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('CNN_Loss.png')

    # plt.figure(2)
    # plt.plot(epochs,avg_trainacc,'blue',label='Train Accuracy mean')
    # plt.plot(epochs,avg_testacc,'red',label='Test Accuracy mean')
    # plt.legend()
    # plt.grid(True)
    # plt.title('Mean Accuracy vs. Iterations')
    # plt.savefig('CNN_ACC.png')

    # plt.figure(3)
    # plt.plot(epochs,std_trainloss,'blue',label='Train Loss STD')
    # plt.plot(epochs,std_testloss,'red',label='Test Loss STD')
    # plt.legend()
    # plt.grid(True)
    # plt.title('STD Loss vs. Iterations')
    # plt.savefig('CNN_loss_std.png')


    # plt.figure(4)
    # plt.plot(epochs,std_trainacc,'blue',label='Train Accuracy STD')
    # plt.plot(epochs,std_testacc,'red',label='Test Accuracy STD')
    # plt.legend()
    # plt.grid(True)
    # plt.title('STD Accuracy vs. Iterations')
    # plt.savefig('CNN_ACC_std.png')
    # print('>>> figures are saving <<<')
    # plt.show()

    # plt.show()
