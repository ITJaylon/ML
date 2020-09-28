import numpy as np

#导入文档信息
def loadDataSet(filename):
    file = open(filename)
    list = file.readlines()
    returnList = []
    for line in list:
        temp = line.strip().split(' ')
        returnList.append(temp)
    classVec = [0,1,0,1,0,1]
    return returnList,classVec

# 创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 输入文本转为词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the world %s is not in the vocablist!" %word)
    return returnVec

def createTrainMax(vocabList, dataList):
    trainMax = []
    for data in dataList:
        trainMax.append(setOfWords2Vec(vocabList,data))
    return trainMax

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 1.0; p1Denom = 1.0       #对概率做平滑处理，分子预设为1，分母预设为2
    for i in range (numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #由于概率乘积数值过小，可能会引起下溢出，所以用log，
    p0Vect = np.log(p0Num/p0Denom)
    return p1Vect,p0Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1 > p0: return 1
    else: return 0

if __name__ == '__main__':
     datalist,classVec = loadDataSet('dataset.txt')
     vocablist = createVocabList(datalist)
     trainMax = createTrainMax(vocablist,datalist)
     #print(setOfWords2Vec(vocablist,['my','dog','stupid']))

     # print(trainNB0(trainMax,classVec))
     p1Vec,p0Vec,pClass1 = trainNB0(trainMax,classVec)
     myWord = ['my','love','my','dog']
     myWord2Vec = setOfWords2Vec(vocablist,myWord)
     print("the class is %d !" % classifyNB(myWord2Vec,p0Vec,p1Vec,pClass1))