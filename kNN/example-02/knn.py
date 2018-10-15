from numpy import *
import operator
from os import listdir


# knn算法：获取结果label
def classify0(inX, dataSet, labels, k):
	# 数据行数
	dataSetSize = dataSet.shape[0]
	# inX行向拓展1，列向拓展dataSetSize（与训练数据同规模）
	# 获取新数据与训练集每行数据的差值矩阵
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffmat = diffMat ** 2
	# 将每行数据累加，获取累加和
	sqDistances = sqDiffmat.sum(axis=1)
	distances = sqDistances ** 0.5
	# 对distances元素从小到大排序，sortedDistIndicies按值从小到大顺序存储在distances中的索引
	# 假如：distances=[2,4,5,3,-10,1]，那sortedDistIndicies=[4 5 0 3 1 2]
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		# .get() dict中有该key就返回值，没有就返回0
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 对获取的k个最近邻数据进行排序和同类累加
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


# 将32*32的文本转为1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 读取训练数据目录，获取所有训练数据文件
    trainingFileList = listdir('trainingDigits')
    # 获取训练数据文件数量
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 获取0_1.txt的0_1
        fileStr = fileNameStr.split('.')[0]
        # 获取0_1的0
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
