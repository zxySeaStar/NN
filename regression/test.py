
import hammingCode
from bitarray import bitarray
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import adamod

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
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x =  self.predict(x)           # linear output
        return x


BATCH_SIZE = 50      # 批训练的数据个数


def Train():

    if_cuda= torch.cuda.is_available()
    print("if_cuda=",if_cuda)

    gpu_count = torch.cuda.device_count()
    print("gpu_count=", gpu_count)

    x, y = LoadData()
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    #indices = torch.LongTensor([0])
    #y = torch.index_select(y, 1, indices)

    testX = x[0:200]
    testY = y[0:200]

    torch_dataset = Data.TensorDataset(testX,testY)
    print(torch_dataset)
    # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    # y = x.pow(2) + 0.2 * torch.rand(x.size())
    # torch_dataset = Data.TensorDataset(x,y)

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多线程来读数据
    )

    net = Net(n_feature=x.shape[1], n_hidden=20, n_output=y.shape[1])  # define the network
    print(net)  # net architecture

    # optimizer = torch.optim.NAdam(net.parameters(),lr=0.01)
    optimizer = adamod.AdaMod(net.parameters(),lr=0.01)
    loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss

    if if_cuda:
        net = net.cuda()
        loss_func = loss_func.cuda()

    plt.ion()  # something about plotting
    total_loss_history = []
    for epoch in range(100):  # 训练所有!整套!数据 3 次
        loss_history = []
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # 假设这里就是你训练的地方...
            if if_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            prediction = net(batch_x)  # input x and predict based on x
            loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            loss_history.append(loss.data.cpu())

        # 打出来一些数据
        total_loss = 0
        for i in loss_history:
            total_loss += i
        total_loss /= step
        total_loss_history.append(total_loss)

        if epoch % 5 == 0:
            plt.cla()
            plt.ylim([0, 1])
            plt.xlim([0, 100])
            plt.plot([i for i in range(len(total_loss_history))], total_loss_history, 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % total_loss, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

            count = 0
            for i in range(700,900):
                testX = x[i]
                if if_cuda:
                    testX = testX.cuda()
                prediction = net(testX)
                prediction = prediction.cpu()
                if (prediction[0]>0.5 and y[i][0] == 1) or (prediction[0]<0.5 and y[i][0]==0):
                    count+=1
            print('Epoch: ', epoch, 'Loss=%.4f' % total_loss, "Presition",count/200 )

    plt.ioff()  # something about plotting
    plt.show()
    # save model
    torch.save(net, 'XY8_1000.pt')
    print("finished")

def Predict():
    if_cuda= torch.cuda.is_available()
    print("if_cuda=",if_cuda)

    gpu_count = torch.cuda.device_count()
    print("gpu_count=", gpu_count)

    # load data
    X = np.load("X8_2000.npy")
    Y = np.load("Y8_2000.npy")

    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))

    indices = torch.LongTensor([0])
    Y = torch.index_select(Y, 1, indices)

    # load model
    net = torch.load("XY8_1000.pt")

    x = X
    y = Y

    count = 0
    for i in range(len(x)):
        testX = x[i]
        if if_cuda:
            testX = testX.cuda()
        prediction = net(testX)
        prediction = prediction.cpu()
        print(prediction,y[i])
        if (prediction > 0.5 and y[i] == 1) or (prediction < 0.5 and y[i] == 0):
            count += 1
        else:
            print("error")
    print("Presition", count / len(x))

if __name__ == "__main__":
    #GenerateData8()



    Train()


    #Predict()



