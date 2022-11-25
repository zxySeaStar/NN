
import hammingCode
from bitarray import bitarray
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data


def GenerateData():
    x = [ list(hammingCode.GetData("{:0>32b}".format(i))) for i in range(2047432,2047432+1000)]
    y = [ list(hammingCode.GetHamming(i)) for i in x]
    X = np.array(x)
    Y = np.array(y)
    np.save("X.npy",X)
    np.save("Y.npy",Y)

def GenerateData32():
    x = [ list(hammingCode.GetRandom(32)) for i in range(1000)]
    y = [list(hammingCode.GetHamming(i)) for i in x]
    X = np.array(x)
    Y = np.array(y)
    np.save("X32_1000.npy", X)
    np.save("Y32_1000.npy", Y)

def GenerateData8():
    x = [ list(hammingCode.GetRandom(8)) for i in range(1000)]
    y = [list(hammingCode.GetHamming(i)) for i in x]
    X = np.array(x)
    Y = np.array(y)
    np.save("X8_1000.npy", X)
    np.save("Y8_1000.npy", Y)

def GenerateData4():
    x = [ list(hammingCode.GetRandom(4)) for i in range(1000)]
    y = [list(hammingCode.GetHamming(i)) for i in x]
    X = np.array(x)
    Y = np.array(y)
    np.save("X4_1000.npy", X)
    np.save("Y4_1000.npy", Y)

def LoadData():
    # load data x and y
    X = np.load("X8_1000.npy")
    Y = np.load("Y8_1000.npy")
    return X,Y

def ViewData():
    X = np.load("X8_1000.npy")
    Y = np.load("Y8_1000.npy")
    print(X)
    print(Y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


BATCH_SIZE = 1      # 批训练的数据个数


def Train():
    x, y = LoadData()
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    indices = torch.LongTensor([0])
    y = torch.index_select(y, 1, indices)
    torch_dataset = Data.TensorDataset(x[0:700], y[0:700])

    # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    # y = x.pow(2) + 0.2 * torch.rand(x.size())
    # torch_dataset = Data.TensorDataset(x,y)

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=8,  # 多线程来读数据
    )

    net = Net(n_feature=x.shape[1], n_hidden=20, n_output=y.shape[1])  # define the network
    print(net)  # net architecture

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    for epoch in range(100):  # 训练所有!整套!数据 3 次
        loss_history = []
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # 假设这里就是你训练的地方...
            prediction = net(batch_x)  # input x and predict based on x
            loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            loss_history.append(loss.data.numpy())

                # 打出来一些数据
        total_loss = 0
        for i in loss_history:
            total_loss += i
        total_loss /= step

        print('Epoch: ', epoch, 'Loss=%.4f'% total_loss)

    count = 0
    for i in range(700,900):
        prediction = net(x[i])
        if (prediction>0.5 and y[i] == 1) or (prediction<0.5 and y[i]==0):
            count+=1
    print(200,count,count/200)

if __name__ == "__main__":
    #GenerateData4()
    Train()

    #ViewData()



