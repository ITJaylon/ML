import decideTree
import storeTree
import plotTree

if __name__ == '__main__':
    #data = [[1,1,'是'],[1,1,'是'],[2,1,'否'],[2,2,'否'],[2,2,'否']]
    # shannoEnt = calcshannoEnt(data)
    # print(shannoEnt)
    #print(classifyDataSetByFeature(data,0,2))
    #dataset,labels = decideTree.createDataSet()
    #decideTree = decideTree.createDecideTree(dataset,labels)
    #print(decideTree)
    #plotTree.createPlot()
#--------------------------------------------------------------
    #storeTree.storeTree(decideTree,'myDecideTree.txt')
    # myTree = storeTree.grabTree('myDecideTree.txt')
    # print(myTree)

#--------------------------------------------------------------
    file = open('lenses.txt')
    labels = file.readline().strip().split('\t')
    lenses = [example.strip().split('\t') for example in file.readlines()]
    decideTree = decideTree.createDecideTree(lenses,labels)
    print(decideTree)
    plotTree.createPlot(decideTree)
