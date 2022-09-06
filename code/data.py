import torch
import numpy as np
import pandas as pd
import csv
import random
from folium import plugins #地图可视化接口
from folium.features import CustomIcon
import folium

class CityData():
    def __init__(self, cityName):
        self.cityName = cityName

        # User Related Data
        self.InterData = self.__readInterest()
        self.RouteData = self.__readRoutes()
        self.UserProfile = self.__getUserProfile()

        # POI Related Data
        self.poiPos, self.poiName, self.poiCategory = self.__readPOI()

        # features
        self.numSample = len(self.UserProfile)
        self.numPOI = len(self.poiName)
        self.numCategory = len(self.InterData[0])
        self.__centerLat = sum([lat for [lat, long] in self.poiPos])/self.numPOI
        self.__centerLong = sum([long for [lat, long] in self.poiPos])/self.numPOI


    def __readInterest(self):
        '''
        读取用户对于各种type poi的兴趣
        output: List(numUsers, numCategories)
        '''
        filename = "data/" + self.cityName + "/fake_user.csv"
        data = pd.read_csv(filename, header=None)
        return data.values.tolist()  #需要打印看一下结构： 4×n

    def __readRoutes(self):
        '''
        读取3000个用户访问的路径
        ouput: List(3000, routeLen), poi ranges from 0->29/28
        '''
        filename = "data/" + self.cityName + "/fake_path.csv"
        data = pd.read_csv(filename, header=None)
        train_data = np.array(data)
        new_train_data = []
        for x in train_data:
            x = x[~np.isnan(x)]
            new_train_data.append([(int(poi)-1) for poi in x.tolist()])
        return new_train_data

    def __getUserProfile(self):
        '''
        把用户4维兴趣和起点、终点连接成为6维向量输入
        output: List(3000, Tensor(numCategory+2, 1, 1))， 3000个样本点，每个样本6维度
        '''
        userProfile = []
        for i in range(len(self.InterData)):
            item = self.InterData[i].copy()
            item.append(float(self.RouteData[i][0]))
            item.append(float(self.RouteData[i][-1]))
            item = torch.Tensor(item).reshape(-1,1,1)
            userProfile.append(item)
        return userProfile

    def __readPOI(self):
        '''
        读取POI数据
        output: poiPos, poiName, poiCategory
        '''
        filename = "data/" + self.cityName + "/POI.csv"
        data = pd.read_csv(filename, header=0, sep=";")
        data = data.values.tolist()
        poiPos = [[lat, long] for (_, _, lat, long, _) in data]
        poiName = [name for (_, name, _, _, _) in data]
        poiCategory = [category for (_, _, _, _, category) in data]
        temp = []
        for c in poiCategory:
            if c not in temp:
                temp.append(c)
        for i in range(len(poiCategory)):
            poiCategory[i] = temp.index(poiCategory[i])
        return poiPos, poiName, poiCategory

    def __route2tensor(self, userRoute):
        '''
        把一个route转换为训练时的decoder输入和输出
        output: de_i, de_o, (length-1, 1, numPOI)
        '''
        length = len(userRoute)
        if length != 1:
            tensor_i = torch.zeros(length-1, 1, self.numPOI)
            tensor_o = torch.zeros(length-1, 1, self.numPOI)
            for i in range(length-1):
                tensor_i[i][0][int(userRoute[i])] = 1
                tensor_o[i][0][int(userRoute[i+1])] = 1
        else:
            tensor_i = torch.zeros(1, 1, self.numPOI)
            tensor_o = torch.zeros(1, 1, self.numPOI)
            tensor_i[0][0][int(userRoute[0])] = 1
            tensor_o[0][0][int(userRoute[0])] = 1
        return tensor_i, tensor_o 

    def trainingExample(self, index = None):
        '''
        随机或者指定获取一个 Trianing Example, 忽视长度小于等于3的数据
        input : index(指定参数), 若没有指定, 随机获得一个trainingExample
        output : index, en_i, de_i, de_o
        '''
        if index==None or index>=self.numSample:
            index = random.randint(0, self.numSample-1)
        while len(self.RouteData[index]) <= 3: #忽视路径长度小于等于3的数据
            index = random.randint(0, self.numSample-1)
        en_i = self.UserProfile[index]
        de_i, de_o = self.__route2tensor(self.RouteData[index])
        return index, en_i, de_i, de_o

    def trainingExample_v2(self, index = None):
        '''
        随机或者指定获取一个 Trianing Example, 忽视长度小于等于3的数据
        input : index(指定参数), 若没有指定, 随机获得一个trainingExample
        output : index, pref_i, pos_i, de_i, de_o
        '''
        index, en_i, de_i, de_o = self.trainingExample(index)
        pref_i = en_i[:-2]

        pos_i = torch.zeros([6])
        pos_i[0] = en_i[-2].item()
        pos_i[1] = self.poiPos[int(pos_i[0])][0]
        pos_i[2] = self.poiPos[int(pos_i[0])][1] 

        pos_i[3] = en_i[-1].item()
        pos_i[4] = self.poiPos[int(pos_i[3])][0]
        pos_i[5] = self.poiPos[int(pos_i[3])][1] 
        pos_i = pos_i.view(-1,1,1)

        return index, pref_i, pos_i, de_i, de_o


    def poi2tensor(self, poiIndex):
        '''
        把poiID转为one-hot向量
        input: poiIndex
        output: Tensor(1, numPOI)
        '''
        tensor = torch.zeros(1,self.numPOI)
        tensor[0][poiIndex] = 1
        return tensor

    def tensor2poi(self, tensor, rank = 0):
        '''
        把one-hot向量转为poiID
        input: POITensor(numPOI), rank
        output: POI Index
        '''
        tensor = tensor.view(1,-1)
        _, top_i = tensor.topk(rank+1)
        top_i = top_i.flatten()
        return top_i[rank].item()

    def plotRoute(self, route, recRoute, savePath = "result/result.html"):
        '''
        在地图中画出推荐结果
        默认的储存路径为 "result/result.html"
        '''
        m = folium.Map([self.__centerLat, self.__centerLong], zoom_start=12, tiles='Stamen Terrain')
        location1 = [self.poiPos[poi] for poi in route]
        location2 = [self.poiPos[poi] for poi in recRoute]
        folium.PolyLine(location1,weight=5,color='blue',opacity=0.8).add_to(m)
        folium.PolyLine(location2,weight=5,color='red',opacity=0.8).add_to(m)

        folium.Marker(location=location1[0], icon=folium.Icon(color='green', icon='home')).add_to(m)
        folium.Marker(location=location1[-1], icon=folium.Icon(color='green', icon='flag')).add_to(m)
        for i in range(1, len(location1)-1):
            if location1[i] not in location2:
                folium.Marker(location=location1[i], icon=folium.Icon(color='blue', icon='ok')).add_to(m)
            else:
                folium.Marker(location=location1[i], icon=folium.Icon(color='purple', icon='ok')).add_to(m)
        for i in range(1, len(location2)-1):
            if location2[i] not in location1:
                folium.Marker(location=location2[i], icon=folium.Icon(color='red', icon='heart')).add_to(m)
        m.save("result/result.html")


if __name__ == '__main__':

    # trt = CityData("Toronto")
    # index, en_i, de_i, de_o = trt.trainingExample()

    osak = CityData("Osaka")
    index, en_i, de_i, de_o = osak.trainingExample(0)
    index, pref_i, pos_i, de_i, de_o = osak.trainingExample_v2(0)

    # osak.plotRoute([0,1,2],[0,3,2])
