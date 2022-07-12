from data import *
from model import *
from train import *
from evaluation import *
from generator import *

from folium import plugins
from folium.features import CustomIcon
import folium
import numpy as np
import pandas as pd
import csv
#？？？？

# 读取POI地理位置文件
filePath = "data/Toronto/POI.csv"
# filePath = "data/Osaka/POI.csv"

data = pd.read_csv(filePath, header=0, sep=";")
data = data.values.tolist()
poiPosList = [[lat, long] for (_, _, lat, long, _) in data]

# 在地图中画出路径
centerLat = sum([lat for [lat, long] in poiPosList])/len(poiPosList)-0.003
centerLong = sum([long for [lat, long] in poiPosList])/len(poiPosList)

def plotRoute(route, recRoute): 
    m = folium.Map([centerLat, centerLong], zoom_start=15, tiles='Stamen Terrain')
    location1 = [poiPosList[poi] for poi in route]
    location2 = [poiPosList[poi] for poi in recRoute]
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


def drawTime(trtData,osakData,labels,yrange,fileName):
    # plot attributes
    # colors = ["#85739D",  "#7EA6E0", "#97D077", "#FFD966", "#EA6B66"]
    # colors = ["#990099",  "#000099", "#009900", "#999900", "#990000"]
    colors = ["#3F485B", "#177E89", "#2E7866", "#FFC857", "#DB3A34"]
    # colors = ["#C43239", "#FFDA5C", "#5084C3", "#02206D", "#1B1419"]
    # colors = ["#1A1D1E", "#C64B2B", "#F9CE8C", "#C3E3E5", "#589BAD"]

    n_groups = 2
    index = np.arange(n_groups)/1.5
    bar_width = 0.1
    opacity = 1.0
    fig, ax=plt.subplots(figsize=(4,2.5))

    for i in range(len(trtData)):
        data = (trtData[i], osakData[i])
        rect = plt.bar(index+bar_width*i,data,bar_width-0.03,color=colors[(i+1)%len(colors)],label=labels[i],alpha=opacity)

    plt.xlabel('City')  
    plt.ylabel('Running Time / ms')  
    plt.xticks(index + bar_width*(len(labels)-1)/2, ('Toronto', 'Osaka'))  
    plt.ylim(yrange[0], yrange[1]);  
    plt.legend(loc='upper center', ncol=int((len(labels)+1)/2), fancybox=True, frameon=False, bbox_to_anchor=(0.5, 1.3));  

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.grid(axis='y')
    plt.savefig(fileName, dpi=300, pad_inches=0.1, bbox_inches="tight")
    plt.show()




if __name__ == "__main__":
    # straightly load model
    encoder = [torch.load("result/v2_PrefEncoder_trt.pkl"), torch.load("result/v2_PosEncoder_trt.pkl")]
    decoder = torch.load("result/v2_Decoder_trt.pkl")

    index = 40
    trt = CityData("Toronto")
    index, pref_i, pos_i, de_i, de_o = trt.trainingExample_v2(index)

    route = trt.RouteData[index]
    route = [int(i) for i in route]
    minLen = max(5, len(route)-5)
    maxLen = min(15, len(route)+5)
    # print(route)

    generator = POIGenerator(encoder, decoder, trt)
    generator.encode_v2(pref_i, pos_i)

    minLen = 7
    maxLen = 7
    # recRouts = generator.GridBeamSearchByAveProb(cons = [], minLength = minLen, maxLength = maxLen, beamWidth=10)
    recRoutes = generator.GridBeamSearchByAveProb(cons = [], minLength = minLen, maxLength = maxLen, beamWidth=5, maxNum=3, isForceEnd=True)
    print(route)
    print(recRoutes[0])
    # for recRoute in recRoutes:
    #     print(recRoute, f1(route, recRoute))
    # Recf1 = [f1(route, recRoute) for recRoute in recRoutes]
    # Recf1 = sorted(Recf1, reverse=True)[:5]
    # meanF1 = sum(Recf1)/len(Recf1)
    # print(meanF1)

    plotRoute(route, recRoutes[0])


    trtData = [1.776/3, 5.794/3, 25.02/3, 14.465/3, 22.286/3]
    osakData = [1.439/3, 5.081/3, 22.344/3, 12.528/3, 19.989/3]
    drawTime(trtData, osakData, labels = ["Random", "DP-Geo", "DP-Pref", "LearningTour", "MatTrip-1"], yrange = [0,11], fileName="result/testTime1.jpg")


    trtData = [22.286/3, 51.25/3, 85.633/3, 130.48/3, 156.57/3]
    osakData = [19.989/3, 50.14/3, 79.088/3, 121.33/3, 163.86/3]
    drawTime(trtData, osakData, labels = ["MatTrip-1", "MatTrip-2", "MatTrip-3", "MatTrip-4", "MatTrip-5"], yrange = [0,65], fileName="result/testTime2.jpg")
    
    trtData = [27.1421, 27.8427, 28.8820, 99.2484] #99.0721
    osakData = [23.4628, 24.8067, 29.3920, 95.1155] #92.5147
    drawTime(trtData, osakData, labels = ["No constraint", "Weather Dependency", "POI Opening Hours", "1 Mandatory POI"], yrange = [0,110], fileName="result/testTime3.jpg")

    trtData = [27.1421, 99.2484, 174.3407, 266.1320] #0,1,2,4
    osakData = [23.4628, 95.1155, 173.5567, 277.1363] #0,1,2,4
    drawTime(trtData, osakData, labels = ["#MandatoryPOI: 0","#MandatoryPOI: 1","#MandatoryPOI: 2","#MandatoryPOI: 4"], yrange = [0,325], fileName="result/testTime4.jpg")


    # plot poi distribution
    # route = []
    # for i in range(len(poiPosList)):
    #     route.append(i)
    # location1 = [poiPosList[poi] for poi in route]
    # m = folium.Map([centerLat, centerLong], zoom_start=12, tiles='Stamen Terrain')
    # for i in range(len(location1)):
    #     folium.Marker(location=location1[i]).add_to(m)
    # m.save("result/result.html")
    
        




# def drawTime1():   
#     n_groups = 2;     
#     random = (1.776/3, 1.439/3)
#     dist = (5.794/3, 5.081/3)
#     pref = (25.02/3, 22.344/3)
#     lt = (14.465/3, 12.528/3)
#     mnrt_1 = (22.286/3, 19.989/3)
#     err_attr={"elinewidth":2,"ecolor":"y","capsize":3}

       
#     # fig, ax = plt.subplots()  
#     index = np.arange(n_groups)/1.2
#     bar_width = 0.1
       
#     opacity = 1.0
#     rects1 = plt.bar(index, random, bar_width, yerr=((0,0.2),(0,0.9)),alpha=opacity, color='#D0D0D0',label= 'Random',error_kw=err_attr)  
#     rects2 = plt.bar(index + bar_width, dist, bar_width,alpha=opacity,color='#A0A0A0',label='DP-Dist',error_kw=err_attr) 
#     rects3 = plt.bar(index + bar_width*2, pref, bar_width,alpha=opacity,color='#707070',label='DP-Pref',error_kw=err_attr)
#     rects4 = plt.bar(index + bar_width*3, lt, bar_width,alpha=opacity,color='#404040',label='LearnigTour',error_kw=err_attr) 
#     rects5 = plt.bar(index + bar_width*4, mnrt_1, bar_width,alpha=opacity,color='#101010',label='MNRT-1',error_kw=err_attr)   

       
#     plt.xlabel('City')  
#     plt.ylabel('Running Time / ms')  
#     # plt.title('Scores by group and gender')  
#     plt.xticks(index + bar_width*2, ('Toronto', 'Osaka'))  
#     plt.ylim(0,12.5);  
#     plt.legend();  
    
#     # plt.tight_layout(); 
#     plt.savefig('result/testTime1.jpg', dpi=500)
#     plt.show()




# def drawTime2():   
#     n_groups = 2;     
#     mnrt_1 = (22.286/3, 19.989/3)
#     mnrt_2 = (51.25/3, 50.14/3)
#     mnrt_3 = (85.633/3, 79.088/3)
#     mnrt_4 = (130.48/3, 121.33/3)
#     mnrt_5 = (156.57/3, 163.86/3)
       
#     fig, ax = plt.subplots()  
#     index = np.arange(n_groups)/1.2
#     bar_width = 0.1
       
#     opacity = 1.0

#     rects1 = plt.bar(index + bar_width*0, mnrt_1, bar_width,alpha=opacity,color='#D0D0D0',label='MNRT-1')  
#     rects2 = plt.bar(index + bar_width*1, mnrt_2, bar_width,alpha=opacity,color='#A0A0A0',label='MNRT-2')  
#     rects3 = plt.bar(index + bar_width*2, mnrt_3, bar_width,alpha=opacity,color='#707070',label='MNRT-3')  
#     rects4 = plt.bar(index + bar_width*3, mnrt_4, bar_width,alpha=opacity,color='#404040',label='MNRT-4')  
#     rects5 = plt.bar(index + bar_width*4, mnrt_5, bar_width,alpha=opacity,color='#101010',label='MNRT-5')  
#     # rects6 = plt.bar(index + bar_width*5, mnrt_3, bar_width,alpha=opacity,color='b',label='MNRT-3')  
#     # rects7 = plt.bar(index + bar_width*6, mnrt_5, bar_width,alpha=opacity,color='r',label='MNRT-5')  

       
#     plt.xlabel('City')  
#     plt.ylabel('Running Time / ms')  
#     # plt.title('Scores by group and gender')  
#     plt.xticks(index + bar_width*2, ('Toronto', 'Osaka'))  
#     plt.ylim(0,60);  
#     plt.legend();  
    
#     # plt.tight_layout(); 
#     plt.savefig('result/testTime2.jpg', dpi=500)
#     plt.show()
