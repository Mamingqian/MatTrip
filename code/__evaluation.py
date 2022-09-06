from generation import *
from train import *
from plot import *
import time

# count the length of generated route and test time
def testLength(encoder, decoder):
    lengthCountGBS = torch.zeros(30)
    lengthCountG = torch.zeros(30)
    lengthCount = torch.zeros(30)
    for i in range(3000):
        index, en_i, de_i, de_o = trainingExample(i)

        lengthCount[len(routesData[i])-1] += 1

        sampleTargetRoute = greedyRoute(encoder, decoder, en_i)
        lengthCountG[len(sampleTargetRoute)-1] += 1

        sampleTargetRouteBS = BSRoute(encoder, decoder, en_i)
        lengthCountGBS[len(sampleTargetRouteBS)-1] += 1

    print("original length: ", lengthCount)
    print("generateGre len: ", lengthCountG)
    print("generateBS leng: ", lengthCountGBS)

def testTime(encoder, decoder):
    allUsers = []
    greedyRoutes = []
    bsRoutes = []
    for i in range(3000):
        allUsers.append(trainingExample(i))

    startTime = time.time()
    for i in range(3000):
        greedyRoutes.append(greedyRoute(encoder, decoder, en_i))
    print("Greedy Time: ", timeSince(startTime))

    startTime = time.time()
    for i in range(3000):
        greedyRoutes.append(BSRoute(encoder, decoder, en_i))
    print("BS Time: ", timeSince(startTime))




if __name__ == "__main__":
    encoder = torch.load("result/encoder.pkl")
    decoder = torch.load("result/decoder.pkl")
    
    allF1 = []
    badIndex = []
    
    for index in range(1000):
        _, en_i, _, _ = trainingExample(index)
        if len(routesData[index])<=3:
            print(index, "skip")
            continue

        route = [int(i) for i in routesData[index]]
        generator = POIGenerator(encoder, decoder)
        generator.encode(en_i)

        minLen = max(5, len(route)-5)
        maxLen = min(15, len(route)+5)

        beamWidth = 3
        isContinue = False
        while True:
            recRoutes = generator.GridBeamSearchByAveProb(cons = [], minLength = minLen, maxLength=maxLen, beamWidth=beamWidth, maxNum=1)
            if len(recRoutes)!=0:
                break
            beamWidth *= 2
            if beamWidth>=24:
                recRoutes = generator.GridBeamSearchByAveProb(cons = [], minLength = minLen, maxLength=maxLen, beamWidth=beamWidth, maxNum=1, isForceEnd=True)
                break

        if isContinue:
            print(index, "failed")
            badIndex.append(index)
            continue

        Recf1 = [f1(route, recRoute) for recRoute in recRoutes]
        Recf1 = sorted(Recf1, reverse=True)[:5]
        meanF1 = sum(Recf1)/len(Recf1)
        if meanF1<0.6:
            badIndex.append(index)

        allF1.append(meanF1)
        print(index, meanF1, beamWidth)
    
    print(sum(allF1)/len(allF1))