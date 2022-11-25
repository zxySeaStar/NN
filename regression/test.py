
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
    print(Y)
    print(X,len(X))
    np.save("X.npy",X)
    np.save("Y.npy",Y)

def LoadData():
    # load data x and y
    X = np.load("X.npy")
    Y = np.load("Y.npy")
    return X,Y

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


BATCH_SIZE = 10      # 批训练的数据个数

if __name__ == "__main__":
    #GenerateData()

    x, y =LoadData()
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    indices = torch.LongTensor([0])
    y=torch.index_select(y, 1, indices)
    torch_dataset = Data.TensorDataset(x[0:800], y[0:800])

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )

    net = Net(n_feature=x.shape[1], n_hidden=10, n_output=y.shape[1])  # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    for epoch in range(3):  # 训练所有!整套!数据 3 次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # 假设这里就是你训练的地方...
            for t in range(10):
                prediction = net(batch_x)  # input x and predict based on x
                loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                    # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ',step, 'Loss=%.4f' % loss.data.numpy())

    prediction = net(x[801])
    print(prediction,y[801])
    loss = loss_func(prediction, y[801])
    print(loss.data.numpy())
