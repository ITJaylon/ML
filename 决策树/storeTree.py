import pickle
import os

def storeTree(inputTree,fileName):
    fw = open(fileName,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)