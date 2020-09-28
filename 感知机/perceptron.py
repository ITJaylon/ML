import numpy as np
from numpy import *

def classify_flowers():
    #从iris.txt读入数据
    file = open('iris.txt')
    file.readline()
    dataSet = file.readlines()
    m = len(dataSet)
    hirito = 0.1 #划分测试集和训练集
    testm,trainm = m*hirito,m-m*hirito
    flowerLabelDict = {1:"setosa",-1:"versicolor"}
    testFeature = []
    testLabels = []
    flowerFeatureMat = []
    flowerLabels = []
    for i in range(m):
        temp = dataSet[i].strip().split(' ')
        if temp[5].split('"')[1] == "setosa":
            temp[5] = 1
        elif temp[5].split('"')[1] == "versicolor":
            temp[5] = -1
        if i % (1/hirito) == 0:
            list = []
            for x in temp[1:5]:
                x = float(x)
                list.append(x)
            testFeature.append(list)
            testLabels.append(temp[5])
        else:
            list = []
            for x in temp[1:5]:
                x = float(x)
                list.append(x)
            flowerFeatureMat.append(list)
            flowerLabels.append(temp[5])
    errorCount = 0.0
    # print(testFeature[0])
    theta,prime_b = [],0
    for i in range(int(testm)):
        classifyResult,theta,prime_b = perceptron_classifier(testFeature[i],flowerFeatureMat,flowerLabels)
        print("the flower came from classifier is:%s,the real answer is:%s" % (flowerLabelDict[classifyResult],flowerLabelDict[testLabels[i]]))
        if(classifyResult != testLabels[i]):
            errorCount += 1
    print("the theta is:%s,the primeter_b is:%f" % (str(theta),prime_b))
    print("the total error rate is:%f" % (errorCount/testm))



def perceptron_classifier(inX,featureMat,labelVector):
    dataNum = len(featureMat)
    thetaNum = len(featureMat[0])
    theta = zeros((1,thetaNum))
    primeter_b = 0
    flag = 0
    learning_rate = 1
    while flag == 0:
        for i in range(dataNum):
            tempLabel = 0
            tempRes = sum(theta*featureMat[i]) + primeter_b
            if tempRes > 0:
                tempLabel = 1
            else:
                tempLabel = -1
            if(tempLabel == labelVector[i]):
                if (i == dataNum - 1):
                    flag = 1
                continue
            else:
                learning_rate_mat = tile(learning_rate,(1,thetaNum))
                theta = theta + learning_rate_mat*labelVector[i]*featureMat[i]
                primeter_b = primeter_b + learning_rate*labelVector[i]
                break
    #print("the theta is:%s,the primeter_b is:%f" % (str(theta[0]),primeter_b))
    flag = 0
    if(sum(theta*inX) + primeter_b) >0:
        flag = 1
    else:
        flag = -1
    return flag,theta[0],primeter_b

if __name__ == '__main__':
    # testVector = [1,2]
    # trainMat = [[3,3],[4,3],[1,1]]
    # labels = [1,1,-1]
    # perceptron_classifier(testVector,trainMat,labels)
    classify_flowers()