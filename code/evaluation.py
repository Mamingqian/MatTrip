from data import CityData
from model import *
from train import LT_Trainer, Trainer
from generator import *
from generator2 import *
import time
import math
import matplotlib.pyplot as plt
import numpy as np

def recall(route, recRoute, isInclude = False):
    if len(route) <= 2:
        return 0
    if not isInclude:
        route = route[1:-1]
        recRoute = recRoute[1:-1]
    insect = [poi for poi in route if poi in recRoute]
    return len(insect)/len(route)

def precision(route, recRoute, isInclude = False):
    if len(recRoute) <= 2:
        return 0
    if not isInclude:
        route = route[1:-1]
        recRoute = recRoute[1:-1]
    insect = [poi for poi in route if poi in recRoute]
    return len(insect)/len(recRoute)

def f1(route, recRoute, isInclude = False):
    r = recall(route, recRoute, isInclude)
    p = precision(route, recRoute, isInclude)
    if p==0 and r==0:
        return 0
    return 2*p*r/(p+r)

def timeSince(since):
    now = time.time()
    s = now - since
    return str(s)

def EvaPerformance(city = "Toronto", method = "Random", beamWidth = 1, isForceEnd=False, isWeather=False, isOpening = False, cons = [], maxLen = 10, numTime=None):
    if city=="Toronto":
        data = CityData("Toronto")
        LTEncoder = torch.load("result/LT_Encoder_trt.pkl")
        LTDecoder = torch.load("result/LT_Decoder_trt.pkl")

        PrefEncoder = torch.load("result/v2_PrefEncoder_trt2.pkl")
        PosEncoder = torch.load("result/v2_PosEncoder_trt2.pkl")
        Decoder = torch.load("result/v2_Decoder_trt2.pkl")

        baseline = Baseline(encoder = LTEncoder, decoder = LTDecoder, poiPos = data.poiPos, dataset = data)

        PrefMNTREncoder = torch.load("result/pref_PrefEncoder_trt.pkl")
        PrefMNTRDecoder = torch.load("result/pref_Decoder_trt.pkl")
        PosMNTREncoder = torch.load("result/pos_posEncoder_trt.pkl")
        PosMNTRDecoder = torch.load("result/pos_Decoder_trt.pkl")
        JoinMNTREncoder = torch.load("result/join_PosEncoder_trt.pkl")
        JoinMNTRDecoder = torch.load("result/join_Decoder_trt.pkl")

        NaPrefEncoder = torch.load("result/na_prefEncoder_trt.pkl")
        NaPosEncoder = torch.load("result/na_posEncoder_trt.pkl")
        NaDecoder = torch.load("result/na_decoder_trt.pkl")
        NaMLP = torch.load("result/na_mlp_trt.pkl")


    elif city=="Osaka":
        data = CityData("Osaka")
        LTEncoder = torch.load("result/LT_Encoder_osak.pkl")
        LTDecoder = torch.load("result/LT_Decoder_osak.pkl")

        PrefEncoder = torch.load("result/v2_PrefEncoder_osak.pkl")
        PosEncoder = torch.load("result/v2_PosEncoder_osak.pkl")
        Decoder = torch.load("result/v2_Decoder_osak.pkl")

        baseline = Baseline(encoder = LTEncoder, decoder = LTDecoder, poiPos = data.poiPos, dataset = data)
        

        PrefMNTREncoder = torch.load("result/pref_PrefEncoder_osak.pkl")
        PrefMNTRDecoder = torch.load("result/pref_Decoder_osak.pkl")
        PosMNTREncoder = torch.load("result/pos_posEncoder_osak.pkl")
        PosMNTRDecoder = torch.load("result/pos_Decoder_osak.pkl")
        JoinMNTREncoder = torch.load("result/join_PosEncoder_osak.pkl")
        JoinMNTRDecoder = torch.load("result/join_Decoder_osak.pkl")

        NaPrefEncoder = torch.load("result/na_prefEncoder_osak.pkl")
        NaPosEncoder = torch.load("result/na_posEncoder_osak.pkl")
        NaDecoder = torch.load("result/na_decoder_osak.pkl")
        NaMLP = torch.load("result/na_mlp_osak.pkl")
    
    if method=="MNTR":
        generator = POIGenerator(encoder = [PrefEncoder, PosEncoder], decoder = Decoder, dataset = data)

    allF1 = []
    allF2 = []
    allR1 = []
    allR2 = []
    allP1 = []
    allP2  =[]
    allTime = []

    iterTime = data.numSample
    if numTime!= None:
        iterTime = numTime
    for index in range(iterTime):
        start = time.time()
        newIndex, pref_i, pos_i, _, _ = data.trainingExample_v2(index)
        if newIndex!=index:
            continue
        _, en_i, _, _ = data.trainingExample(index)
        route = data.RouteData[newIndex]
        if method=="MNTR":
            generator.encode_v2(pref_i, pos_i)
            if isWeather:
                generator.enableWeather()
            if isOpening:
                generator.enableOpenTime()
            recRoutes = generator.GridBeamSearchByAveProb(cons=cons, minLength=5, maxLength=maxLen, beamWidth=beamWidth, isForceEnd=isForceEnd)
            if len(recRoutes)<1:
                continue
            recRoute = recRoutes[0]
        elif method=="Random":
            recRoute = baseline.Random(en_i, minLen = 5, maxLen = 15)
        elif method=="Dist":
            recRoute = baseline.Dist(en_i, minLen = 5, maxLen = 15)
        elif method=="Pref":
            recRoute = baseline.Pref(en_i, minLen = 5, maxLen = 15, rate=3)
        elif method=="LT":
            recRoute = baseline.LearningTour(en_i, minLen=5, maxLen=15)
        elif method=="PrefMNTR":
            # generator.encode_pref(pref_i, pos_i)
            # recRoutes = generator.GridBeamSearchByAveProb(cons=cons, minLength=5, maxLength=maxLen, beamWidth=beamWidth, isForceEnd=isForceEnd)
            # recRoute = recRoutes[0]
            recRoute = prefGenerate(PrefMNTREncoder, PrefMNTRDecoder, data, pref_i, pos_i, maxLength=maxLen)
        elif method=="PosMNTR":
            recRoute = posGenerate(PosMNTREncoder, PosMNTRDecoder, data, pref_i, pos_i, maxLength=maxLen)
        elif method=="JoinMNTR":
            recRoute = joinGenerate(JoinMNTREncoder, JoinMNTRDecoder, data, pref_i, pos_i, maxLength=maxLen)
        elif method=="NaMNTR":
            recRoute = naGenerate(NaPrefEncoder, NaPosEncoder, NaMLP, NaDecoder, data, pref_i, pos_i, maxLength=maxLen)


        end = time.time()
        allTime.append(end-start)
        allF1.append(f1(route, recRoute))
        allF2.append(f1(route, recRoute, isInclude=True))
        allR1.append(recall(route, recRoute))
        allR2.append(recall(route, recRoute, isInclude=True))
        allP1.append(precision(route, recRoute))
        allP2.append(precision(route, recRoute, isInclude=True))

    print("City : " + city)
    print("Method : " + method)
    print("F1 : ", sum(allF1)/len(allF1))
    # print("F1*: ", sum(allF2)/len(allF2))
    print("Recall : ", sum(allR1)/len(allR1))
    # print("Recall*: ", sum(allR2)/len(allR2))
    print("Precision : ", sum(allP1)/len(allP1))
    # print("Precision*: ", sum(allP2)/len(allP2))
    print("Time : ", sum(allTime)/3, " ms")
    print("MaxTime : ", max(allTime)*1000, " ms")
    print("MinTime : ", min(allTime)*1000, " ms")
    print("")

def EvaPerformance2(city = "Toronto", method = "MNTR", beamWidth = 1, isForceEnd=False, isWeather=False, isOpening = False, cons = [], maxLen = 10, numTime=None):

    if city=="Toronto":
        data = CityData("Toronto")

        #MNT
        PrefEncoder = torch.load("result/v2_PrefEncoder_trt2.pkl")
        PosEncoder = torch.load("result/v2_PosEncoder_trt2.pkl")
        Decoder = torch.load("result/v2_Decoder_trt2.pkl")

        #PrefMNTR
        PrefMNTREncoder = torch.load("result/pref_PrefEncoder_trt.pkl")
        PrefMNTRDecoder = torch.load("result/pref_Decoder_trt.pkl")

        #PosMNTR
        PosMNTREncoder = torch.load("result/pos_posEncoder_trt.pkl")
        PosMNTRDecoder = torch.load("result/pos_Decoder_trt.pkl")

        #JoinMNTR
        JoinMNTREncoder = torch.load("result/join_PosEncoder_trt.pkl")
        JoinMNTRDecoder = torch.load("result/join_Decoder_trt.pkl")

        #NaMNTR
        NaPrefEncoder = torch.load("result/na_prefEncoder_trt.pkl")
        NaPosEncoder = torch.load("result/na_posEncoder_trt.pkl")
        NaDecoder = torch.load("result/na_decoder_trt.pkl")
        NaMLP = torch.load("result/na_mlp_trt.pkl")

    elif city=="Osaka":
        data = CityData("Osaka")

        #MNTR
        PrefEncoder = torch.load("result/v2_PrefEncoder_osak2.pkl")
        PosEncoder = torch.load("result/v2_PosEncoder_osak2.pkl")
        Decoder = torch.load("result/v2_Decoder_osak2.pkl")

        #PrefMNTR
        PrefMNTREncoder = torch.load("result/pref_PrefEncoder_osak.pkl")
        PrefMNTRDecoder = torch.load("result/pref_Decoder_osak.pkl")

        #PosMNTR
        PosMNTREncoder = torch.load("result/pos_posEncoder_osak.pkl")
        PosMNTRDecoder = torch.load("result/pos_Decoder_osak.pkl")

        #JoinMNTR
        JoinMNTREncoder = torch.load("result/join_PosEncoder_osak.pkl")
        JoinMNTRDecoder = torch.load("result/join_Decoder_osak.pkl")

        #NaMNTR
        NaPrefEncoder = torch.load("result/na_prefEncoder_osak.pkl")
        NaPosEncoder = torch.load("result/na_posEncoder_osak.pkl")
        NaDecoder = torch.load("result/na_decoder_osak.pkl")
        NaMLP = torch.load("result/na_mlp_osak.pkl")
    
    if method=="MNTR":
        generator = POIGenerator2(encoder = [PrefEncoder, PosEncoder], decoder = Decoder, dataset = data, method=method)
    elif method=="PrefMNTR":
        generator = POIGenerator2(encoder = PrefMNTREncoder, decoder = PrefMNTRDecoder, dataset = data, method=method)
    elif method=="PosMNTR":
        generator = POIGenerator2(encoder = PosMNTREncoder, decoder = PosMNTRDecoder, dataset = data, method=method)
    elif method=="JoinMNTR":
        generator = POIGenerator2(encoder = JoinMNTREncoder, decoder = JoinMNTRDecoder, dataset = data, method=method)
    elif method=="NaMNTR":
        generator = POIGenerator2(encoder = [NaPrefEncoder, NaPosEncoder], decoder = NaDecoder, dataset = data, mlp=NaMLP,method=method)

    allF1 = []
    allF2 = []
    allR1 = []
    allR2 = []
    allP1 = []
    allP2  =[]
    allTime = []
    iterTime = data.numSample
    if numTime!= None:
        iterTime = numTime

    for index in range(iterTime):
        start = time.time()
        newIndex, pref_i, pos_i, _, _ = data.trainingExample_v2(index)
        if newIndex!=index:
            continue

        route = data.RouteData[newIndex]
        generator.encode(pref_i, pos_i)
        recRoute = generator.BeamSearch(minLength=5, maxLength=maxLen,beamWidth=beamWidth)

        end = time.time()
        allTime.append(end-start)
        allF1.append(f1(route, recRoute))
        allF2.append(f1(route, recRoute, isInclude=True))
        allR1.append(recall(route, recRoute))
        allR2.append(recall(route, recRoute, isInclude=True))
        allP1.append(precision(route, recRoute))
        allP2.append(precision(route, recRoute, isInclude=True))

    print("City : " + city)
    print("Method : " + method)
    print("F1 : ", sum(allF1)/len(allF1))
    # print("F1*: ", sum(allF2)/len(allF2))
    print("Recall : ", sum(allR1)/len(allR1))
    # print("Recall*: ", sum(allR2)/len(allR2))
    print("Precision : ", sum(allP1)/len(allP1))
    # print("Precision*: ", sum(allP2)/len(allP2))
    print("Time : ", sum(allTime)/3, " ms")
    print("MaxTime : ", max(allTime)*1000, " ms")
    print("MinTime : ", min(allTime)*1000, " ms")
    print("")

def drawTimeWithErrorBar(trtData,osakData,trtMin,trtMax,osakMin,osakMax,labels,yrange,fileName):
    # plot attributes
    colors = ["#D0D0D0", "#A0A0A0", "#707070", "#404040", "#101010"]
    n_groups = 2
    err_attr={"elinewidth":2,"ecolor":"y","capsize":3}
    index = np.arange(n_groups)/1.2
    bar_width = 0.1
    opacity = 1.0

    for i in range(len(trtData)):
        data = (trtData[i], osakData[i])
        error = ((trtData[i]-trtMin[i], osakData[i]-osakMin[i]), (trtMax[i]-trtData[i], osakMax[i]-osakData[i]))
        rect = plt.bar(index+bar_width*i,data,bar_width,yerr=error,color=colors[i%len(colors)],label=labels[i],error_kw=err_attr)

    plt.xlabel('City')  
    plt.ylabel('Running Time / ms')  
    plt.xticks(index + bar_width*2, ('Toronto', 'Osaka'))  
    plt.ylim(yrange[0], yrange[1]);  
    plt.legend();  

    plt.savefig(fileName, dpi=500)
    plt.show()

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
    ## Normal
    # EvaPerformance("Toronto", "Random")
    # EvaPerformance("Osaka", "Random")
    # EvaPerformance("Toronto", "Dist")
    # EvaPerformance("Osaka", "Dist")
    # EvaPerformance("Toronto", "Pref")
    # EvaPerformance("Osaka", "Pref")
    # EvaPerformance("Toronto", "MNTR", beamWidth=1, isForceEnd=True, maxLen=15)
    # EvaPerformance("Toronto", "MNTR", beamWidth=2, isForceEnd=True, maxLen=15)
    # EvaPerformance("Toronto", "MNTR", beamWidth=3, isForceEnd=True, maxLen=15)
    # EvaPerformance("Toronto", "MNTR", beamWidth=4, isForceEnd=True, maxLen=15)
    # EvaPerformance("Toronto", "MNTR", beamWidth=5, isForceEnd=True, maxLen=15)
    # EvaPerformance("Osaka", "MNTR", beamWidth=1, isForceEnd=True, maxLen=10)
    # EvaPerformance("Osaka", "MNTR", beamWidth=2, isForceEnd=True, maxLen=10)
    # EvaPerformance("Osaka", "MNTR", beamWidth=3, isForceEnd=True, maxLen=10)
    # EvaPerformance("Osaka", "MNTR", beamWidth=4, isForceEnd=True, maxLen=10)
    # EvaPerformance("Osaka", "MNTR", beamWidth=5, isForceEnd=True, maxLen=10)


    ## functionality
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True)
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,isWeather=True)
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,isOpening=True)
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,cons=[3])
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5])
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5,12])
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5,12,21])
    # EvaPerformance("Toronto", "MNTR",beamWidth=3,isForceEnd=True,isWeather=True, isOpening=True, cons=[3])
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True)
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,isWeather=True)
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,isOpening=True)
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,cons=[3])
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5])
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5,12])
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,cons=[3,5,12,21])
    # EvaPerformance("Osaka", "MNTR",beamWidth=3,isForceEnd=True,isWeather=True, isOpening=True, cons=[3])

    # trtData = [1.776/3, 5.794/3, 25.02/3, 14.465/3, 22.286/3]
    # osakData = [1.439/3, 5.081/3, 22.344/3, 12.528/3, 19.989/3]
    # drawTime(trtData, osakData, labels = ["Random", "DP-Geo", "DP-Pref", "LearningTour", "MatNN-1"], yrange = [0,11], fileName="result/testTime1.jpg")


    # trtData = [22.286/3, 51.25/3, 85.633/3, 130.48/3, 156.57/3]
    # osakData = [19.989/3, 50.14/3, 79.088/3, 121.33/3, 163.86/3]
    # drawTime(trtData, osakData, labels = ["MatNN-1", "MatNN-2", "MatNN-3", "MatNN-4", "MatNN-5"], yrange = [0,65], fileName="result/testTime2.jpg")
    
    # trtData = [27.1421, 27.8427, 28.8820, 99.2484] #99.0721
    # osakData = [23.4628, 24.8067, 29.3920, 95.1155] #92.5147
    # drawTime(trtData, osakData, labels = ["No constraint", "Weather Dependency", "POI Opening Hours", "1 Mandatory POI"], yrange = [0,110], fileName="result/testTime3.jpg")

    # trtData = [27.1421, 99.2484, 174.3407, 266.1320] #0,1,2,4
    # osakData = [23.4628, 95.1155, 173.5567, 277.1363] #0,1,2,4
    # drawTime(trtData, osakData, labels = ["#MandatoryPOI: 0","#MandatoryPOI: 1","#MandatoryPOI: 2","#MandatoryPOI: 4"], yrange = [0,325], fileName="result/testTime4.jpg")

    # maxLens = [10,8]

    # EvaPerformance("Toronto", "PrefMNTR", maxLen=maxLens[0])
    # EvaPerformance("Osaka", "PrefMNTR", maxLen=maxLens[1])
    
    # EvaPerformance("Toronto", "PosMNTR", maxLen=maxLens[0])
    # EvaPerformance("Osaka", "PosMNTR", maxLen=maxLens[1])

    # EvaPerformance("Toronto", "JoinMNTR", maxLen=maxLens[0])
    # EvaPerformance("Osaka", "JoinMNTR", maxLen=maxLens[1])

    # EvaPerformance("Toronto", "NaMNTR", maxLen=maxLens[0])
    # EvaPerformance("Osaka", "NaMNTR", maxLen=maxLens[1])
    
    # EvaPerformance("Toronto", "MNTR", beamWidth=1, isForceEnd=True, maxLen=maxLens[0])
    # EvaPerformance("Osaka", "MNTR", beamWidth=1, isForceEnd=True, maxLen=maxLens[1])
    
    # EvaPerformance("Toronto", "MNTR2", maxLen=maxLens[0])
    # EvaPerformance("Osaka", "MNTR2", maxLen=maxLens[1])


    maxLens = [15,15]

    # EvaPerformance2("Toronto", "PrefMNTR", maxLen=maxLens[0])
    # EvaPerformance2("Osaka", "PrefMNTR", maxLen=maxLens[1])
    
    # EvaPerformance2("Toronto", "PosMNTR", maxLen=maxLens[0])
    # EvaPerformance2("Osaka", "PosMNTR", maxLen=maxLens[1])

    # EvaPerformance2("Toronto", "JoinMNTR", maxLen=maxLens[0])
    # EvaPerformance2("Osaka", "JoinMNTR", maxLen=maxLens[1])

    # EvaPerformance2("Toronto", "NaMNTR", maxLen=maxLens[0])
    EvaPerformance2("Osaka", "NaMNTR", maxLen=maxLens[1])
    
    # EvaPerformance2("Toronto", "MNTR", maxLen=maxLens[0])
    # EvaPerformance2("Osaka", "MNTR", maxLen=maxLens[1])

    # EvaPerformance2("Toronto", "MNTR", maxLen=maxLens[0], beamWidth=2)
    # EvaPerformance2("Osaka", "MNTR", maxLen=maxLens[1], beamWidth=2)

    # EvaPerformance2("Toronto", "MNTR", maxLen=maxLens[0], beamWidth=3)
    # EvaPerformance2("Osaka", "MNTR", maxLen=maxLens[1], beamWidth=3)

    # EvaPerformance2("Toronto", "MNTR", maxLen=maxLens[0], beamWidth=4)
    # EvaPerformance2("Osaka", "MNTR", maxLen=maxLens[1], beamWidth=4)

    # EvaPerformance2("Toronto", "MNTR", maxLen=maxLens[0], beamWidth=5)
    # EvaPerformance2("Osaka", "MNTR", maxLen=maxLens[1], beamWidth=5)