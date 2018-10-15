from numpy import *
import operator


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


# 读取训练数据文件，获得data和lable
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


# 对特征值进行归一化处理
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	# 在numpy中，矩阵除法为linalg.solve(matA,matB)
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVals


# 计算错误率
def datingClassTest():
	# 测试数据比率
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 获取测试数据的数量
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # normMat[numTestVecs:m, :]--->测试数据
        # datingLabels[numTestVecs:m]--->测试数据标签
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))



# 对新数据进行分类
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: %s" % resultList[classifierResult - 1]
