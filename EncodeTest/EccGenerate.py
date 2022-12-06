import numpy as np
from bitarray import  bitarray

class DataLoader:
    def __init__(self,filePath):
        with open(filePath, "r") as f:
            content = f.read().strip().split("\n")

        priorityList = []
        for line in content:
            column = line.split(" ")
            for item in column:
                bitItem = "{:0>32b}".format(int.from_bytes(bytes.fromhex(item), "big"))
                priorityList.append(bitarray(bitItem))
                #print(bitItem,item,bitItem.count('1'))
        #print(len(priorityList))
        self.data = bitarray()
        for item in priorityList:
            self.data.extend(item)
        #print(len(self.data))
        self.ecc = self.data[32896-32:]
        self.data = self.data[0:32896-32]
        print(self.ecc)
        print(self.data)

    # def data(self):
    #     return self.data
    #
    # def ecc(self):
    #     return self.ecc

class EccGenerate:
    def __init__(self):
        # basic matrix
        self.H = np.zeros(4096*8*17, dtype=int).reshape(4096*8, 17)
        #print(self.H)

        # 对角线
        for i in range(0,self.H.shape[0]):
            self.H[i][(15-i)%16] ^= 1

        # 对于每个bit的变化规律
        # 第0bit   低16bits为1 高16bits为0
        self.DataBitInit(0, 16, 1)
        # for i in range(0, self.H.shape[0]):
        #     if i % 32 < 16:
        #         self.H[i][0] ^= 1

        # 第1bit   低64bits为0 高64bits为1 且每256\1024\2048\8192\16384bits会翻转一次
        self.DataBitInit(1, 64, 0)

        self.DataBitFlip(1, 256)
        self.DataBitFlip(1, 1024)
        self.DataBitFlip(1, 2048)
        self.DataBitFlip(1, 8192)
        self.DataBitFlip(1, 16384)
        # for i in range(0, self.H.shape[0]):
        #     if i % 32 < 16:
        #         self.H[i][0] ^= 1

        # 第2bit   低32bits为0 高32bits为1 且每256\512\2048\4096\16384bits会翻转一次
        self.DataBitInit(2, 32, 0)
        self.DataBitFlip(2, 256)
        self.DataBitFlip(2, 512)
        self.DataBitFlip(2, 2048)
        self.DataBitFlip(2, 4096)
        self.DataBitFlip(2, 16384)

        # 第3bit   低16384bits为0 高16384bits为1
        self.DataBitInit(3, 16384, 0)

        # 第4bit   低32bits为1 高32bits为0 且每64\128\2048\4096\8192bits会翻转一次
        self.DataBitInit(4, 32, 1)
        self.DataBitFlip(4, 64)
        self.DataBitFlip(4, 128)
        self.DataBitFlip(4, 2048)
        self.DataBitFlip(4, 4096)
        self.DataBitFlip(4, 8192)

        # 第5bit   低8192bits为0 高8192bits为1
        self.DataBitInit(5, 8192, 0)

        # 第6bit   低4096bits为0 高4096bits为1
        self.DataBitInit(6, 4096, 0)

        # 第7bit   低2048bits为0 高2048bits为1
        self.DataBitInit(7, 2048, 0)

        # 第8bit   低32bits为1 高32bits为0 且每64\128\256\512\1024bits会翻转一次
        self.DataBitInit(8, 32, 1)
        self.DataBitFlip(8, 64)
        self.DataBitFlip(8, 128)
        self.DataBitFlip(8, 256)
        self.DataBitFlip(8, 512)
        self.DataBitFlip(8, 1024)

        # 第9bit   低1024bits为0 高1024bits为1
        self.DataBitInit(9, 1024, 0)

        # 第10bit  低512bits为0 高512bits为1
        self.DataBitInit(10, 512, 0)

        # 第11bit  低256bits为0 高256bits为1
        self.DataBitInit(11, 256, 0)

        # 第12bit  低128bits为0 高128bits为1
        self.DataBitInit(12, 128, 0)

        # 第13bit  低64bits为1 高64bits为0
        self.DataBitInit(13, 64, 1)

        # 第14bit  低32bits为1 高32bits为0
        self.DataBitInit(14, 32, 1)

        # 第15bit  1
        self.DataBitInit(15, 4096*8+32*4, 1)

        # 第16bit  低16bits为0 高16bits为1 且每128\512\1024\4096\8192\16384bits会翻转一次
        self.DataBitInit(16, 16, 0)
        self.DataBitFlip(16, 128)
        self.DataBitFlip(16, 512)
        self.DataBitFlip(16, 1024)
        self.DataBitFlip(16, 4096)
        self.DataBitFlip(16, 8192)
        self.DataBitFlip(16, 16384)

        #print(self.H[:32])
        #print(self.H[32:64])

        #print("After !")
        # 32 位大小端交换
        for i in range(0, self.H.shape[0],32):
            # temp = self.H[i+0:i+8,:].copy()
            # self.H[i+0:i+8,:] = self.H[i+24:i+32,:].copy()
            # self.H[i+24:i+32, :] = temp
            # temp = self.H[i + 8:i + 16, :].copy()
            # self.H[i + 8:i + 16, :] = self.H[i + 16:i + 24, :].copy()
            # self.H[i + 16:i + 24, :] = temp

            temp = self.H[i:i+32, :].copy()
            temp[:, :] = temp[::-1, :]
            self.H[i:i+32, :] = temp

        #print(self.H[:32])
        #print(self.H[32:64])
        #print(self.H[64:64+32])
        #np.save("test.npy",self.H)

    def DataBitInit(self, offset, step, value):
        if value == 1:
            valueInverse = 0
        else:
            valueInverse = 1
        for i in range(0, self.H.shape[0]):
            if (i % (step * 2)) < step:
                self.H[i][offset] ^= value
            else:
                self.H[i][offset] ^= valueInverse

    def DataBitFlip(self, offset, step):
        for i in range(step, self.H.shape[0]):
            if (i % (step*2) ) >= step:
                self.H[i][offset] ^= 1

    def DiffBit(self, oldData:bitarray, newData:bitarray)->list:
        length = len(oldData)
        result = list()
        for i in range(length):
            if oldData[i] != newData[i]:
                result.append(i)
        return result

    def DiffXor(self, diffBitList:list)->bitarray:
        result = np.zeros(17, dtype=bool)
        for line in diffBitList:
            result = np.logical_xor(result, self.H[line])
        # np array to bitarray
        result = bitarray("".join(list(map(lambda x: "1" if x else "0",result) )))
        return result

    def Generate(self, oldData, newData, oldEccCode):
        # collect the changed bit
        diffBitList = self.DiffBit(oldData, newData)
        print(diffBitList, self.H[24607])
        #print(list(map(hex,diffBitList)))
        # get the xor result of column of changed bit
        xorValue = self.DiffXor(diffBitList)
        print("xor val:", xorValue)
        # apply to then oldEccCode
        newEccCode = (oldEccCode ^ xorValue)
        return newEccCode

    def Display(self):

        pass

def test():
    data10 = DataLoader("./TestData10.txt")
    data11 = DataLoader("./TestData11.txt")

    # build the matrix
    eccGenerator = EccGenerate()
    print("ori xor:",(data10.ecc^data11.ecc)[0:17])
    # collect the data
    oldData = data10.data[0:32768]
    newData = data11.data[0:32768]
    oldEccCode = data10.ecc[0:17]  # bitarray("01111111111111101")

    # xor all the the column which data bit is changed
    newEccCode = eccGenerator.Generate(oldData, newData, oldEccCode)

    # display result
    print("old ecc:", data10.ecc)
    print("ori ecc:", data11.ecc)
    print("new ecc:", newEccCode)

if  __name__ == "__main__":

    test()
    #e = EccGenerate()
    # for i in range(1024):
    #     print(i,e.H[i])
    #print(e.H[24576:24608])
    #A = np.array([[1,2,0],[9,9,9],[2,3,4],[1,2,3],[9,9,9],[2,3,6]])

    #print(A)
    #temp = A[5::-1,:].copy()
    #A[:, :] =  A[5::-1,:] #temp
    #print(A[::-1,:])
    #A[0] = np.flip(A[0])
    # for i in range(len(e.H)):
    #     print("{:0>4x}".format(i),e.H[i])