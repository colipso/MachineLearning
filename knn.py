# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:46:15 2014

@author: hp
"""

import numpy as np
import operator
import matplotlib as mp

def createDataSet():
    group=np.random.rand(4,2)
    labels=['a','b','c','d']
    return group,labels
    
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndicies[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
    
def file2matrix(filename):
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    fr1=open(filename)
    indexa=0
    for line in fr1.readlines():
        line=line.strip()
        listFromLine=line.split()
        returnMat[indexa,:]=listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        indexa+=1
    mp.pyplot.scatter(returnMat[:,1],returnMat[:,2])
    return returnMat,classLabelVector
    

    
    
group,labels=createDataSet()
result=classify0([0,1],group,labels,3)
print '对坐标0,0的分类为%s'%(result)

print '-----------喜欢类型--------'
mat,clv=file2matrix('/home/hp/Downloads/machinelearninginaction/Ch02/datingTestSet.txt')


