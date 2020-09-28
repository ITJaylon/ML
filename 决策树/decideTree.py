from math import *

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#计算香浓熵
def calcshannoEnt(dataSet):
    numEntries = len(dataSet)
    labeldict = {}
    for entity in dataSet:
        label = entity[-1]
        if label not in labeldict:
            labeldict[label] = 0
        labeldict[label] += 1
    shannoEnt = 0.0
    for key in labeldict:
        classLabelNum = labeldict[key]
        shannoEnt -= (classLabelNum/numEntries)*log(classLabelNum/numEntries,2)
    return shannoEnt

#按特征划分数据集
def classifyDataSetByFeature(dataSet,axis,value):
    resDataSet = []
    for feaVct in dataSet:
        if feaVct[axis] == value:
            tempfea = feaVct[:axis]
            tempfea.extend(feaVct[axis+1:])
            resDataSet.append(tempfea)
    return resDataSet

def chooseBestFeature(dataset):
    featureNum = len(dataset[0])-1
    if featureNum == 1:
        return 0
    baseEntryEnt = calcshannoEnt(dataset)
    maxEnt = 0.0
    for i in range(featureNum):
        featureList = [feature[i] for feature in dataset]
        featureSet = set(featureList)
        featureEntryEnt,returnFeature = 0.0,0
        for value in featureSet:
            classifiedDataSet = classifyDataSetByFeature(dataset,i,value)
            rate = len(classifiedDataSet)/float(len(dataset))
            tempEnt = calcshannoEnt(classifiedDataSet)
            featureEntryEnt += rate*tempEnt
        tempEntryEnt = baseEntryEnt - featureEntryEnt
        if tempEntryEnt > maxEnt:
            maxEnt = tempEntryEnt
            returnFeature = i
    return returnFeature

def majorityCnt(dataSet): #对于最后的特征，若还不是单一分类，采用多数表决
    labelDict = {}
    for item in dataSet:
        label = item[-1]
        if label not in labelDict.keys():
            labelDict[label] = 0
            labelDict[label] += 1
    returnList = sorted(labelDict.items(),key=lambda labelDict:labelDict[1],reverse=True)
    return returnList[0][0]

def createDecideTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels = [featLabel for featLabel in labels] #必须用一个变量接受labels，否则对label的修改会影响到函数外，因为python中传递的是label的引用
    myTree = {bestFeatLabel:{}}

    del(featLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues: #value是这个最佳分类特征的各种取值，构成树的边
        sublabels = featLabels[:]
        myTree[bestFeatLabel][value] = createDecideTree(classifyDataSetByFeature
                    (dataSet,bestFeat,value),sublabels)
    return myTree

def classfy(testVct, inputTree, featLabels):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVct[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classfy(testVct,secondDict[key],featLabels)
                break
            else:
                classLabel = secondDict[key]
                break
    return classLabel


