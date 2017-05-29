import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName

if __name__ == '__main__':
    data,cityName = loadData('D:\work\MOOCML\课程数据\聚类\city.txt')
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    print(expenses)
    CituCluster = [[],[],[],[]]
    for i in range(len(cityName)):
        CituCluster[label[i]].append(cityName[i])
    for i in range(len(CituCluster)):
        print("Expenses:%.df" % expenses[i])
        print(CituCluster[i])