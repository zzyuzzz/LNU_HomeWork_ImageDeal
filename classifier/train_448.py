# %%
import torch 
import torch.nn as nn
from classifier import Darknet19
# import torch_npu
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch.utils.tensorboard as tensorbord

# %matplotlib inline

# %%
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, Resize, Lambda
from torchvision import datasets

# %%

def transData(x:torch.Tensor):
    if len(x.shape) < 3:
        x = torch.stack([x] * (3-len(x.shape)))
    elif x.shape[0] < 3:
        x = torch.cat([x] * 3, 0)
    return (x).float().detach()



train_data = datasets.Caltech256(
    root="data",
    transform=Compose([
        Resize((448, 448)),
        ToTensor(),
        Lambda(transData)

    ]),
    download=True,
   )

# %%
# name = None
# for X,y in train_data:
#     if name is None or name != y:
#         print(train_data.categories[y])
#         name = y

# %%
training_data, testing_data = random_split(train_data, [0.7, 0.3])

# %%
print(training_data[1][0])
# print(testing_data[1][0])

# %%
batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# %%
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


model = Darknet19(448).to(device=device)
if os.path.exists('saved_model.pth'):
    model_t = torch.load('saved_model.pth')
    model.layers.load_state_dict(model_t.layers.state_dict())
else:
    raise ValueError("请先训练224大小的网络!!!")
print(model)

# %%

writer = tensorbord.SummaryWriter()

def train(model:nn.Module, optimizer:torch.optim.Optimizer, loss_fn, data:DataLoader, iter_num=0):
    data_len = len(data.dataset)
    model.train()
    with tqdm(enumerate(data), total=data_len//data.batch_size) as pbar:
        for batch, (X, y) in pbar:
            # y = torch.unsqueeze(y, -1)
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            writer.add_scalar('Loss/train', loss.item(), batch + iter_num*(data_len//data.batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f'{loss.item():>7f}')


# %%
def test(model:nn.Module, loss_fn, data:DataLoader, iter_num = 0):
    data_len = len(data.dataset)
    model.eval()
    with torch.no_grad():
        total_loss, correct = 0, 0
        with tqdm(enumerate(data), total=data_len//data.batch_size) as pbar:
            for batch, (X, y) in pbar:
                X,y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                

                correct_t = (pred.argmax(1) == y).type(torch.float).sum().item()
                correct += correct_t
                pbar.set_postfix(loss=f'{loss.item():>7f}', correct=correct)
                total_loss = (total_loss * batch + loss.item()) / (batch + 1)
                
                writer.add_scalar('Loss/test', loss.item(), batch + iter_num*(data_len//data.batch_size))
                writer.add_scalar('Accracy/test', correct_t, batch + iter_num*(data_len//data.batch_size))

        print(f"total loss:{total_loss:.9}, Accuracy:{correct/data_len:.9}")

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()



# %%
epsi = 10
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer, power=4, total_iters=epsi, verbose=True)

for i in range(epsi):
    print(f"第{i}次迭代")
    train(model=model, optimizer=optimizer, loss_fn=loss_fn, data=train_dataloader, iter_num=i)
    test(model=model, loss_fn=loss_fn, data=test_dataloader, iter_num=i)
    scheduler.step()

    print('starting save model')

    torch.save(model, 'saved_model.pth')


print('Done 448')


# saved_model224 = 'saved_model.pth'
# model = Darknet19(448).to(device=device)
# if os.path.exists(saved_model224):
#     model_t = torch.load(saved_model224)
#     model.layers.load_state_dict(model_t.layers.state_dict())

    
# print(model)
# epsi = 10
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer, power=4, total_iters=epsi, verbose=True)
# for i in range(epsi):
#     print(f"第{i}次迭代")
#     train(model=model, optimizer=optimizer, loss_fn=loss_fn, data=train_dataloader, iter_num=i)
#     test(model=model, loss_fn=loss_fn, data=test_dataloader, iter_num=i)
#     scheduler.step()

#     print('starting save model')

#     torch.save(model, 'saved_model448.pth')




