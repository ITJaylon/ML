from math import exp
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels ,maxCycles = 5000):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001

    weights = ones((n,1))
    costArray = []
    xcord = []
    x0 = []
    x1 = []
    x2 = []
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        #print(h)
        #print(labelMat.dtype + h.dtype)
        error = (h-labelMat)
        cost = 0.0
        for j in range(n):
            cost = cost + labelMat[j]*(-log(h[j])) - (1-labelMat[j])*log(1-h[j])
        costArray.append(cost)
        x0.append(weights[0])
        x1.append(weights[1])
        x2.append(weights[2])
        # print(error)
        xcord.append(k)
        weights = weights - alpha * dataMatrix.transpose() * error #此处有一个数学定理证明，暂时未搞明白。
        #plotBestFit(weights)
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ycord = x0
    #print(xcord,ycord)
    ax0.scatter(xcord, ycord,s=1, c='green')
    ax1 = fig.add_subplot(312)
    ycord = x1
    # print(xcord,ycord)
    ax1.scatter(xcord, ycord,s=1, c='red')

    ax2 = fig.add_subplot(313)
    ycord = x2
    # print(xcord,ycord)
    ax2.scatter(xcord, ycord,s=1, c='yellow')
    #x = arange(-3.0, 3.0, 0.1)
    # print(type(weights[0]))
    #y = (-weights[0] - weights[1] * x) / weights[2]
    #ax.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('迭代次数')
    plt.show()
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(1,n)
    for j in range (numIter):
        dataIndex = [x for x in range(m)]
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = h - classLabels[randIndex]
            weights = weights - alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def plotBestFit(wei):

    weights = wei.getA() #返回矩阵自身为一个n维数组对象
    dataMat,labelMat = loadDataSet()
    #print(shape(dataMat)[0])
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range (n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #print(type(weights[0]))
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


#从疝气病预测病马的死亡率
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        lineList = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(lineList[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(lineList[-1]))
    #print(trainingSet)
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    #print(trainWeights)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != float(currLine[-1][0]):
            errorCount += 1.0
    errorRate = errorCount/numTestVec
    print("the error rate of this test is %f" %errorRate)
    return errorRate

if __name__ == '__main__':
    colicTest();