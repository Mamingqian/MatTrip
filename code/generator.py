import torch
import operator
import copy
import random
import math
from collections import OrderedDict
from sortedcontainers import SortedListWithKey

from data import *
from model import *

class POIGenerator:
    def __init__(self, encoder, decoder, dataset, mlp=None):
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.mlp = mlp

        self.isWeather = False
        self.isOpenTime = False

    def encode(self, en_i):
        with torch.no_grad():
            en_i = en_i.view(-1,1)
            self.en_i = en_i
            self.input_length = en_i.shape[0]
            
            self.start_poi = self.dataset.poi2tensor(int(en_i[6].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(en_i[7].item()))              # one_hot vector
            self.encoder_output = self.encoder(en_i)                    # seq_len * 1 * (2 * hidden_size)

    def encode_v2(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector

            # encoder forward
            pref_encoder_output = self.encoder[0](pref_i)  # seq_len * 1 * (2 * hidden_size)
            pos_encoder_output = self.encoder[1](pos_i)  # seq_len * 1 * (2 * hidden_size)
            self.encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0)

    def encode_pref(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector

            # encoder forward
            self.encoder_output = self.encoder(pref_i)

    def encode_pos(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector

            # encoder forward
            self.encoder_output = self.encoder(pos_i)

    def encode_join(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector
            # encoder forward
            self.encoder_output = self.encoder(torch.cat([pref_i, pos_i], dim=0))

    def encode_na(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector

            # encoder forward
            pref_encoder_output = self.encoder[0](pref_i)  # seq_len * 1 * (2 * hidden_size)
            pos_encoder_output = self.encoder[1](pos_i)  # seq_len * 1 * (2 * hidden_size)
            encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0)
            self.encoder_output = self.mlp(encoder_output)


    # Enable Additional Functionalities
    def enableWeather(self):
        self.isWeather = True
        self.__setFakeOutDoor()

    def __setFakeOutDoor(self):
        """generate fake outdoor pois"""
        prob = 0.3
        outDoorPOIs = []                                            
        for i in range(self.dataset.numPOI):
            if random.random() < prob:
                outDoorPOIs.append(i)
        self.outDoorPOIs = outDoorPOIs

    def __isRainy(self):
        """check if is rainy now"""
        prob = 0.2
        return random.random() < prob

    def enableOpenTime(self):
        self.isOpenTime = True

    def __isOpen(self, POI_i):
        """Check if current POI_i is open"""
        prob = 0.8
        return random.random() < prob

    # Greedy Generation
    def GreedySearch(self, maxLength = 10):
        with torch.no_grad():
            de_in = self.start_poi
            de_h = self.decoder.initHidden()
            de_c = self.decoder.initCell()

            # init route without start poi, incase (start_poi = end_poi)
            targetRoute = []                          

            while True:
                de_out, (de_h, de_c) = self.decoder(de_in, de_h, de_c, self.encoder_output)
                de_out = de_out.flatten()
                _, top_i = de_out.topk(len(de_out))

                for poi in top_i:
                    poi = poi.detach().item()
                    if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
                        continue
                    if self.isOpenTime and not self.__isOpen(poi):
                        continue
                    if poi not in targetRoute:                      # check if poi is already in the route
                        de_in = self.dataset.poi2tensor(poi)
                        targetRoute.append(poi)
                        break


                if de_in.equal(self.end_poi):                       # iterate until get end_poi
                    break
                if len(targetRoute) >= (maxLength-2):               # or reach to the max length
                    targetRoute.append(tensor2poi(self.end_poi))
                    break                                   

            targetRoute.insert(0, self.dataset.tensor2poi(self.start_poi))
            return targetRoute

    # Our Proposed Methods
    class Route:
        def __init__(self, hidden_size, numPOI):
            self.path = []
            self.prob = 0
            self.nextInput = torch.zeros(1, numPOI)
            self.nextHidden = torch.zeros(1, hidden_size)
            self.nextCell = torch.zeros(1, hidden_size)

    class Beam:
        def __init__(self, beamWidth, initRoute = None):
            self.beamWidth = beamWidth
            self.routes = SortedListWithKey(initRoute, key=lambda x:-x.prob)

        def addRoute(self, route):
            self.routes.add(route)
            if(len(self.routes) > self.beamWidth):
                assert len(self.routes) == self.beamWidth+1
                del self.routes[-1]
        
        def __len__(self):
            return len(self.routes)

        def __iter__(self):
            for route in self.routes:
                yield route

    def BeamSearch(self, maxLength = 10, beamWidth = 3):
        with torch.no_grad():
            # initiate beam
            init_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
            init_route.nextInput = self.start_poi
            init_route.nextHidden = self.decoder.initHidden()
            init_route.nextCell = self.decoder.initCell()

            beam = self.Beam(beamWidth, initRoute = [init_route])

            while True:
                # iterate for each route in beam, no more than (beam_width) routes
                newBeam = self.Beam(beamWidth)
                for route in beam:
                    de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
                    de_out = de_out.flatten()
                    _, top_i = de_out.topk(len(de_out))

                    # choose (beamWidth) number of new routes with max prob
                    beamSize = 0
                    for poi in top_i:  
                        poi = poi.detach().item()
                        if beamSize >= beamWidth:
                            break
                        if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
                            continue
                        if self.isOpenTime and not self.__isOpen(poi):
                            continue
                        if poi not in route.path:
                            new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
                            new_route.nextInput = self.dataset.poi2tensor(poi)
                            new_route.nextHidden = de_h
                            new_route.nextCell = de_c
                            new_route.path = route.path.copy()
                            new_route.path.append(poi)
                            new_route.prob = route.prob + de_out[poi].item()
                            newBeam.addRoute(new_route)
                            beamSize += 1

                beam = newBeam
                for route in beam: 
                    if len(route.path) >= (maxLength-2):                   # terminate when best route reach max_len
                        route.path.append(tensor2poi(self.end_poi))
                        route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
                        return route.path
                    if self.dataset.tensor2poi(self.end_poi) == route.path[-1]:         # terminate when reach end_poi
                        route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
                        return route.path

    def GridBeamSearch(self, cons = [], minLength = None, maxLength = None, beamWidth = 3, isForceEnd=False):
        # if minLength == None:
        #     minLength = 3
        # if maxLength == None:
        #     maxLength = 20
        # with torch.no_grad():
        #     search_grid = self.__ConstructGrid(cons, minLength, maxLength, beamWidth, isForceEnd=isForceEnd)
            
        #     # find all results with end_poi
        #     resultPaths = []
        #     for ti in range(max(minLength-1, len(cons)), maxLength):
        #         beam = search_grid[(ti,len(cons))]
        #         # print(ti, len(beam))
        #         for route in beam:
        #             if route.path[-1] == self.dataset.tensor2poi(self.end_poi):
        #                 route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
        #                 resultPaths.append(route.path)

        #     return resultPaths
        return self.GridBeamSearchByAveProb(cons = cons, minLength = minLength, maxLength = maxLength, beamWidth = beamWidth, isForceEnd=isForceEnd, maxNum = 3)

    def GridBeamSearchByAveProb(self, cons = [], minLength = None, maxLength = None, beamWidth = 3, isForceEnd=False, maxNum = 3):
        if minLength == None:
            minLength = 3
        if maxLength == None:
            maxLength = 20
        with torch.no_grad():
            search_grid = self.__ConstructGrid(cons, minLength, maxLength, beamWidth, isForceEnd=isForceEnd)
            
            # find all results with end_poi
            resultPaths = []
            for ti in range(max(minLength-1, len(cons)), maxLength):
                beam = search_grid[(ti,len(cons))]
                # print(ti, len(beam))
                for route in beam:
                    if route.path[-1] == self.dataset.tensor2poi(self.end_poi):
                        route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
                        resultPaths.append(   (route.path, route.prob/(len(route.path)-1))   )
            resultPaths.sort(key=lambda x:x[1], reverse=True)
            return [x for (x,y) in resultPaths[:maxNum]]

    def __ConstructGrid(self, cons = [], minLength = 10, maxLength = 15,beamWidth = 3, isForceEnd = False):
        with torch.no_grad():
            # create an empty grid
            grid_height = len(cons)+1
            grid_width = maxLength
            search_grid = OrderedDict()

            # init beam at position (0,0)
            init_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
            init_route.nextInput = self.start_poi
            init_route.nextHidden = self.decoder.initHidden()
            init_route.nextCell = self.decoder.initCell()
            initBeam = self.Beam(beamWidth, initRoute = [init_route])
            search_grid[(0,0)] = initBeam

            for ti in range(1,grid_width):
                cStart = max(0, (ti+len(cons))-grid_width)
                cEnd = min(ti, len(cons))+1
                for ci in range(cStart, cEnd):
                    # generate route from (ti-1, ci) and (ti-1, ci-1)
                    newBeam = self.Beam(beamWidth)

                    #for the top right most beam, make sure the last poi is end_poi
                    if isForceEnd and ci == len(cons) and ti == maxLength-1:
                        beam = search_grid[(ti-1, ci)]
                        for route in beam:
                            de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
                            de_out = de_out.flatten()

                            new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
                            new_route.nextInput = self.end_poi
                            new_route.nextHidden = de_h
                            new_route.nextCell = de_c
                            new_route.path = route.path.copy()
                            new_route.path.append(self.dataset.tensor2poi(self.end_poi))
                            new_route.prob = route.prob + de_out[self.dataset.tensor2poi(self.end_poi)].item()
                            newBeam.addRoute(new_route)
                        search_grid[(ti, ci)] = newBeam
                        continue

                    if (ti-1, ci) in search_grid:
                        beam = search_grid[(ti-1, ci)]
                        for route in beam:
                            if len(route.path)>0 and route.path[-1] == self.dataset.tensor2poi(self.end_poi):
                                continue
                            de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
                            de_out = de_out.flatten()
                            _, top_i = de_out.topk(len(de_out))
                            # choose (beamWidth) number of new routes with max prob
                            beamSize = 0
                            for poi in top_i:  
                                poi = poi.detach().item()
                                if beamSize >= beamWidth:
                                    break
                                # in case generate end_poi before all mandatory pois
                                if poi==self.dataset.tensor2poi(self.end_poi) and ci!=(grid_height-1): 
                                    continue
                                if poi in cons:
                                    continue
                                if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
                                    continue
                                if self.isOpenTime and not self.__isOpen(poi):
                                    continue
                                if poi not in route.path:
                                    new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
                                    new_route.nextInput = self.dataset.poi2tensor(poi)
                                    new_route.nextHidden = de_h
                                    new_route.nextCell = de_c
                                    new_route.path = route.path.copy()
                                    new_route.path.append(poi)
                                    new_route.prob = route.prob + de_out[poi].item()
                                    newBeam.addRoute(new_route)
                                    beamSize += 1

                    if (ti-1, ci-1) in search_grid:
                        beam = search_grid[(ti-1, ci-1)]
                        for route in beam:
                            if len(route.path)>0 and route.path[-1] == self.dataset.tensor2poi(self.end_poi):
                                continue
                            de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
                            de_out = de_out.flatten()
                            _, top_i = de_out.topk(len(de_out))
                            # choose (beamWidth) number of new routes with max prob
                            beamSize = 0
                            for poi in top_i:  
                                poi = poi.detach().item()
                                if beamSize >= beamWidth:
                                    break
                                if poi==self.dataset.tensor2poi(self.end_poi) and ci!=(grid_height-1):           
                                    continue
                                if poi not in cons:
                                    continue
                                if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
                                    continue
                                if self.isOpenTime and not self.__isOpen(poi):
                                    continue
                                if poi not in route.path:
                                    new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
                                    new_route.nextInput = self.dataset.poi2tensor(poi)
                                    new_route.nextHidden = de_h
                                    new_route.nextCell = de_c
                                    new_route.path = route.path.copy()
                                    new_route.path.append(poi)
                                    new_route.prob = route.prob + de_out[poi].item()
                                    newBeam.addRoute(new_route)
                                    beamSize += 1
                    
                    search_grid[(ti, ci)] = newBeam

            return search_grid

class Baseline:
    def __init__(self, encoder, decoder, poiPos, dataset):
        self.encoder = encoder
        self.decoder = decoder
        self.poiPos = poiPos
        self.dataset = dataset
    
    def Random(self, en_i, minLen = 5, maxLen = 5):
        randLength = random.randint(minLen, maxLen)
        resultRoute = [int(en_i.flatten()[-2].item())]
        for _ in range(randLength-2):
            randPOI = random.randint(0, self.dataset.numPOI-1)
            resultRoute.append(randPOI)
        resultRoute.append(int(en_i.flatten()[-1].item()))
        return resultRoute


    def __CalDist(self, pos1, pos2):
        return math.sqrt(float((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2))
    
    def Dist(self, en_i, minLen = 5, maxLen = 5):
        startPOI = int(en_i.flatten()[-2].item())
        endPOI = int(en_i.flatten()[-1].item())

        POIrepo = [i for i in range(self.dataset.numPOI)]
        POIrepo.remove(startPOI)
        POIrepo.remove(endPOI)

        resultRoute = [startPOI, endPOI]

        randLength = random.randint(minLen, maxLen)
        for _ in range(randLength-2):
            minLen = 10000
            for poi in POIrepo:
                for i in range(len(resultRoute)-1):
                    currentPOI = resultRoute[i]
                    nextPOI = resultRoute[i+1]
                # print(self.poiPos[poi], self.poiPos[currentPOI])
                    newLen = self.__CalDist(self.poiPos[poi], self.poiPos[currentPOI]) + self.__CalDist(self.poiPos[poi], self.poiPos[nextPOI])
                    # oriLen = self.__CalDist(self.poiPos[nextPOI], self.poiPos[currentPOI])
                    # newLen = newLen-oriLen
                    if newLen < minLen:
                        minLen = newLen
                        minPOI = poi
                        newPos = i

            resultRoute.insert(newPos, minPOI)
            POIrepo.remove(minPOI)
        
        return resultRoute

    def Pref(self, en_i, minLen = 5, maxLen = 5, rate = 2):
        startPOI = int(en_i.flatten()[-2].item())
        endPOI = int(en_i.flatten()[-1].item())

        POIrepo = [i for i in range(self.dataset.numPOI)]
        POIrepo.remove(startPOI)
        POIrepo.remove(endPOI)

        resultRoute = [startPOI, endPOI]

        randLength = random.randint(minLen, maxLen)
        for _ in range(randLength-2):
            maxScore = 0
            for poi in POIrepo:
                for i in range(len(resultRoute)-1):
                    currentPOI = resultRoute[i]
                    nextPOI = resultRoute[i+1]
                    # print(self.poiPos[poi], self.poiPos[currentPOI])
                    newLen = self.__CalDist(self.poiPos[poi], self.poiPos[currentPOI]) + self.__CalDist(self.poiPos[poi], self.poiPos[nextPOI])
                    oriLen = self.__CalDist(self.poiPos[nextPOI], self.poiPos[currentPOI])
                    cate = self.dataset.poiCategory[poi]
                    pref = en_i.flatten()[int(cate)].item()+rate
                    score = pref/(newLen-oriLen+0.01)
                    if score > maxScore:
                        maxScore = score
                        minPOI = poi
                        newPos = i

            resultRoute.insert(newPos, minPOI)
            POIrepo.remove(minPOI)
        
        return resultRoute


    def LearningTour(self, en_i, minLen = 5, maxLen = 15):
        with torch.no_grad():
            targetRoute = []

            input_length = en_i.shape[0] # 8

            # encoder forward
            encoder_hidden = self.encoder.initHidden()
            encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(en_i[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output.reshape(-1)

            decoder_hidden = encoder_hidden

            # decoder forward
            decoder_input = self.dataset.poi2tensor(int(en_i.flatten()[-2]))
            while True:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                poi = self.dataset.tensor2poi(decoder_output)
                targetRoute.append(int(poi))
                if poi==int(en_i.flatten()[-1].item()):
                    break
                if len(targetRoute) >= (maxLen-2):
                    targetRoute.append(int(en_i[-1].item()))
                    break

                decoder_input = self.dataset.poi2tensor(poi)

            targetRoute.insert(0, int(en_i[-2].item()))
            return targetRoute

def prefGenerate(encoder, decoder, dataset, pref_i, pos_i, minLength = 5, maxLength = 15):
    with torch.no_grad():
        targetRoute = []

        # encoder forward
        encoder_output = encoder(pref_i)  # seq_len * 1 * (2 * hidden_size)

        decoder_hidden = decoder.initHidden()
        decoder_cell = decoder.initCell()

        # decoder forward
        decoder_input = dataset.poi2tensor(int(pos_i[3].item()))
        while True:
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)

            rank=0
            while True:
                poi = dataset.tensor2poi(decoder_output, rank=rank)
                if (poi in targetRoute) or (poi==int(pos_i[3].item()) and len(targetRoute)<=minLength-3):
                    rank+=1
                    continue
                break

            targetRoute.append(int(poi))
            if poi==int(pos_i[3].item()):
                break

            if len(targetRoute) >= (maxLength-2):
                targetRoute.append(int(pos_i[3].item()))
                break

            decoder_input = dataset.poi2tensor(poi)

        targetRoute.insert(0, int(pos_i[0].item()))
        return targetRoute

def posGenerate(encoder, decoder, dataset, pref_i, pos_i, minLength = 5, maxLength = 15):
    with torch.no_grad():
        targetRoute = []

        # encoder forward
        encoder_output = encoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        decoder_hidden = decoder.initHidden()
        decoder_cell = decoder.initCell()

        # decoder forward
        decoder_input = dataset.poi2tensor(int(pos_i[3].item()))
        while True:
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
            rank=0
            while True:
                poi = dataset.tensor2poi(decoder_output, rank=rank)
                if (poi in targetRoute) or (poi==int(pos_i[3].item()) and len(targetRoute)<=minLength-3):
                    rank+=1
                    continue
                break
            targetRoute.append(int(poi))
            if poi==int(pos_i[3].item()):
                break
            if len(targetRoute) >= (maxLength-2):
                targetRoute.append(int(pos_i[3].item()))
                break

            decoder_input = dataset.poi2tensor(poi)

        targetRoute.insert(0, int(pos_i[0].item()))
        return targetRoute

def joinGenerate(encoder, decoder, dataset, pref_i, pos_i, minLength = 5, maxLength = 15):
    with torch.no_grad():
        targetRoute = []

        # encoder forward
        encoder_output = encoder(torch.cat([pref_i, pos_i], dim=0))  # seq_len * 1 * (2 * hidden_size)

        decoder_hidden = decoder.initHidden()
        decoder_cell = decoder.initCell()

        # decoder forward
        decoder_input = dataset.poi2tensor(int(pos_i[3].item()))
        while True:
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
            rank=0
            while True:
                poi = dataset.tensor2poi(decoder_output, rank=rank)
                if (poi in targetRoute) or (poi==int(pos_i[3].item()) and len(targetRoute)<=minLength-3):
                    rank+=1
                    continue
                break
            targetRoute.append(int(poi))
            if poi==int(pos_i[3].item()):
                break
            if len(targetRoute) >= (maxLength-2):
                targetRoute.append(int(pos_i[3].item()))
                break

            decoder_input = dataset.poi2tensor(poi)

        targetRoute.insert(0, int(pos_i[0].item()))
        return targetRoute

def naGenerate(prefEncoder, posEncoder, mlp, decoder, dataset, pref_i, pos_i, minLength = 5, maxLength = 15):
    with torch.no_grad():
        targetRoute = []
        pref_encoder_output = prefEncoder(pref_i)  # seq_len * 1 * (2 * hidden_size)
        pos_encoder_output = posEncoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        # fusion
        encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0) #12*1*(2*hidden_size)

        # decoder forward
        encoder_output = mlp(encoder_output)

        decoder_h = decoder.initHidden()
        decoder_hidden = (decoder_h[0], encoder_output.view(1,1,-1))

        decoder_input = dataset.poi2tensor(int(pos_i[3].item()))
        while True:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Create new decoder input
            rank=0
            while True:
                poi = dataset.tensor2poi(decoder_output, rank=rank)
                if (poi in targetRoute) or (poi==int(pos_i[3].item()) and len(targetRoute)<=minLength-3):
                    rank+=1
                    continue
                break
            targetRoute.append(int(poi))
            if poi==int(pos_i[3].item()):
                break
            if len(targetRoute) >= (maxLength-2):
                targetRoute.append(int(pos_i[3].item()))
                break
                
            decoder_input = dataset.poi2tensor(poi)
            
        targetRoute.insert(0, int(pos_i[0].item()))
        return targetRoute

def Generate(prefEncoder, posEncoder, decoder, dataset, pref_i, pos_i, minLength = 5, maxLength = 15):
    with torch.no_grad():
        targetRoute = []
        pref_encoder_output = prefEncoder(pref_i)  # seq_len * 1 * (2 * hidden_size)
        pos_encoder_output = posEncoder(pos_i)  # seq_len * 1 * (2 * hidden_size)

        # fusion
        encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0) #12*1*(2*hidden_size)

        decoder_hidden = decoder.initHidden()
        decoder_cell = decoder.initCell()

        # decoder forward
        decoder_input = dataset.poi2tensor(int(pos_i[3].item()))
        while True:
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
            poi = dataset.tensor2poi(decoder_output)
            targetRoute.append(int(poi))
            if poi==int(pos_i[3].item()):
                break
            if len(targetRoute) >= (maxLength-2):
                targetRoute.append(int(pos_i[3].item()))
                break

            decoder_input = dataset.poi2tensor(poi)

        targetRoute.insert(0, int(pos_i[0].item()))
        return targetRoute

