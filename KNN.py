import numpy as np


def createDataSet():
    """创建数据集"""
    # 每组数据包含打斗数和接吻数；
    group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    # 每组数据对应的标签类型；
    labels = ['Roman', 'Roman', 'Roman', 'Action', 'Action', 'Action']
    return group, labels

def classify(inx, dataSet, labels, k):
    """
    KNN分类算法实现
    :param inx:要预测电影的数据, e.g.[18, 90]
    :param dataSet:传入已知数据集，e.g. group 相当于x
    :param labels:传入标签，e.g. labels相当于y
    :param k:KNN里面的k，也就是我们要选择几个近邻
    :return:电影类新的排序
    """
    dataSetSize = dataSet.shape[0]  # (6,2) -- 6行2列 ===> 6 获取行数
    # tile会重复inx， 把它重复成(dataSetSize, 1)型的矩阵
    # (x1 - y1), (x2 - y2)
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # 相加, axis=1行相加
    sqDistance = sqDiffMat.sum(axis=1)
    # 开根号
    distance = sqDistance ** 0.5
    # 排序索引： 输出的是序列号index， 而不是值
    sortedDistIndicies = distance.argsort()
    # print(sortedDistIndicies)

    classCount = {}
    for i in range(k):
        # 获取排前k个的标签名；
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                              key=lambda d: float(d[1]),
                              reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, label = createDataSet()
    result = classify([40, 45], group, label, 5)
    print("[40, 45]的电影类型：",  result)
