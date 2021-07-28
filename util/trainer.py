import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data

# from tqdm import tqdm
tqdm = lambda x: x


class SimpleNet(nn.Module):
    def __init__(self, depth, width, res= False, num_class=2, alpha=1.0):
        super(SimpleNet, self).__init__()
        self.alpha = alpha
        print('resnet: ',res)
        self.res = res
        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])

        n_final = 1 if num_class == 2 else num_class
        self.final = nn.Linear(width, n_final, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            if self.res:
                skip = x
                x = layer(x)
                x = skip + self.alpha * F.relu(x) * math.sqrt(2)
                x /= math.sqrt(1 + self.alpha**2)
            else:
                x = layer(x)
                x = F.relu(x) * math.sqrt(2)
        return self.final(x)


def train_network(train_loader, test_loader, depth, width, init_lr, decay, break_on_fit=True,
                  res=False, num_class=2, alpha=1, multi_class=False):
    
    if multi_class:
        num_class = 10
    model = SimpleNet(depth, width,res,num_class,alpha).cuda()
    optim = Nero(model.parameters(), lr=init_lr)      
    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_acc_list = []
    train_acc = 0

    for epoch in tqdm(range(10)):
        model.train()

        for data, target in train_loader:
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target, multi_class)
            data, target = (data.cuda(), target.cuda())
            #print(target.shape)
            y_pred = model(data).squeeze()
            #print(y_pred.shape)
            loss = (y_pred - target).norm()

            model.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target, multi_class)
            data, target = (data.cuda(), target.cuda())

            y_pred = model(data).squeeze()
            if multi_class:
                for i in range(target.shape[0]):
                    correct += (target[i,:].float() == y_pred[i,:].sign()).sum() // 10
                    #print((target[i,:].float() == y_pred[i,:].sign()).sum() // 10)
            else:
                correct += (target.float() == y_pred.sign()).sum().item()
            
            total += target.shape[0]

        train_acc = correct/total
        train_acc_list.append(train_acc)

        if break_on_fit and train_acc == 1.0: break
        #print("Epoch: ", epoch)
    model.eval()
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target,multi_class)
        data, target = (data.cuda(), target.cuda())

        y_pred = model(data).squeeze()
        if multi_class:
            for i in range(target.shape[0]):
                correct += (target[i,:].float() == y_pred[i,:].sign()).sum() // 10
        else:
            correct += (target.float() == y_pred.sign()).sum().item()

        total += target.shape[0]

    test_acc = correct/total
    
    return train_acc_list, test_acc, model
