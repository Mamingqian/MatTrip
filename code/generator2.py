import torch
import operator
import copy
import random
import math
from collections import OrderedDict
from sortedcontainers import SortedListWithKey

from data import *
from model import *

class POIGenerator2:
    def __init__(self, encoder, decoder, dataset, mlp=None, method="MNTR"):
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.mlp = mlp
        self.method = method

        self.isWeather = False
        self.isOpenTime = False

    def encode(self, pref_i, pos_i):
        with torch.no_grad():
            self.pref_i = pref_i
            self.pos_i = pos_i
            self.pref_length = pref_i.shape[0]
            self.pos_length = pos_i.shape[0]

            self.start_poi = self.dataset.poi2tensor(int(self.pos_i[0].item()))            # one_hot vector
            self.end_poi = self.dataset.poi2tensor(int(self.pos_i[3].item()))              # one_hot vector
            if self.method=="MNTR":
                pref_encoder_output = self.encoder[0](pref_i)  # seq_len * 1 * (2 * hidden_size)
                pos_encoder_output = self.encoder[1](pos_i)  # seq_len * 1 * (2 * hidden_size)
                self.encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0)
            elif self.method=="PrefMNTR":
                self.encoder_output = self.encoder(pref_i)
            elif self.method=="PosMNTR":
                self.encoder_output = self.encoder(pos_i)
            elif self.method=="JoinMNTR":
                self.encoder_output = self.encoder(torch.cat([pref_i, pos_i], dim=0))
            elif self.method=="NaMNTR":
                pref_encoder_output = self.encoder[0](pref_i)  # seq_len * 1 * (2 * hidden_size)
                pos_encoder_output = self.encoder[1](pos_i)  # seq_len * 1 * (2 * hidden_size)
                encoder_output = torch.cat([pref_encoder_output, pos_encoder_output], dim=0)
                self.encoder_output = self.mlp(encoder_output)

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
            self.routes = SortedListWithKey(initRoute, key=lambda x:-x.prob/(len(x.path)+0.0001))

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

    def BeamSearch(self, minLength = 5, maxLength = 10, beamWidth = 1):
        with torch.no_grad():
            # initiate beam
            init_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
            init_route.nextInput = self.start_poi
            init_route.nextHidden = self.decoder.initHidden()
            if self.method=="NaMNTR":
                init_route.nextCell = self.encoder_output.view(1,1,-1)
            else:
                init_route.nextCell = self.decoder.initCell()

            beam = self.Beam(beamWidth, initRoute = [init_route])
            totalBeam = self.Beam(beamWidth)
            isBreak = False
            while True:
                # iterate for each route in beam, no more than (beam_width) routes
                newBeam = self.Beam(beamWidth)
                for route in beam:
                    if self.method=="NaMNTR":
                        de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell)
                    else:
                        de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
                    de_out = de_out.flatten()
                    _, top_i = de_out.topk(len(de_out))

                    # choose (beamWidth) number of new routes with max prob
                    beamSize = 0
                    for poi in top_i:  
                        poi = poi.detach().item()
                        if beamSize >= beamWidth:
                            break
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
                        route.path.append(self.dataset.tensor2poi(self.end_poi))
                        if beamWidth==1:
                            route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
                            return route.path
                        else:
                            isBreak = True
                            totalBeam.addRoute(route)
                    if self.dataset.tensor2poi(self.end_poi) == route.path[-1]:         # terminate when reach end_poi
                        if beamWidth==1:
                            route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
                            return route.path
                        else:
                            totalBeam.addRoute(route)
                if isBreak:
                    break
            
            route = totalBeam.routes[0]
            route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
            return route.path










    # def GridBeamSearch(self, cons = [], minLength = None, maxLength = None, beamWidth = 3, isForceEnd=False):
    #     # if minLength == None:
    #     #     minLength = 3
    #     # if maxLength == None:
    #     #     maxLength = 20
    #     # with torch.no_grad():
    #     #     search_grid = self.__ConstructGrid(cons, minLength, maxLength, beamWidth, isForceEnd=isForceEnd)
            
    #     #     # find all results with end_poi
    #     #     resultPaths = []
    #     #     for ti in range(max(minLength-1, len(cons)), maxLength):
    #     #         beam = search_grid[(ti,len(cons))]
    #     #         # print(ti, len(beam))
    #     #         for route in beam:
    #     #             if route.path[-1] == self.dataset.tensor2poi(self.end_poi):
    #     #                 route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
    #     #                 resultPaths.append(route.path)

    #     #     return resultPaths
    #     return self.GridBeamSearchByAveProb(cons = cons, minLength = minLength, maxLength = maxLength, beamWidth = beamWidth, isForceEnd=isForceEnd, maxNum = 3)

    # def GridBeamSearchByAveProb(self, cons = [], minLength = None, maxLength = None, beamWidth = 3, isForceEnd=False, maxNum = 3):
    #     if minLength == None:
    #         minLength = 3
    #     if maxLength == None:
    #         maxLength = 20
    #     with torch.no_grad():
    #         search_grid = self.__ConstructGrid(cons, minLength, maxLength, beamWidth, isForceEnd=isForceEnd)
            
    #         # find all results with end_poi
    #         resultPaths = []
    #         for ti in range(max(minLength-1, len(cons)), maxLength):
    #             beam = search_grid[(ti,len(cons))]
    #             # print(ti, len(beam))
    #             for route in beam:
    #                 if route.path[-1] == self.dataset.tensor2poi(self.end_poi):
    #                     route.path.insert(0, self.dataset.tensor2poi(self.start_poi))
    #                     resultPaths.append(   (route.path, route.prob/(len(route.path)-1))   )
    #         resultPaths.sort(key=lambda x:x[1], reverse=True)
    #         return [x for (x,y) in resultPaths[:maxNum]]

    # def __ConstructGrid(self, cons = [], minLength = 10, maxLength = 15,beamWidth = 3, isForceEnd = False):
    #     with torch.no_grad():
    #         # create an empty grid
    #         grid_height = len(cons)+1
    #         grid_width = maxLength
    #         search_grid = OrderedDict()

    #         # init beam at position (0,0)
    #         init_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
    #         init_route.nextInput = self.start_poi
    #         init_route.nextHidden = self.decoder.initHidden()
    #         init_route.nextCell = self.decoder.initCell()
    #         initBeam = self.Beam(beamWidth, initRoute = [init_route])
    #         search_grid[(0,0)] = initBeam

    #         for ti in range(1,grid_width):
    #             cStart = max(0, (ti+len(cons))-grid_width)
    #             cEnd = min(ti, len(cons))+1
    #             for ci in range(cStart, cEnd):
    #                 # generate route from (ti-1, ci) and (ti-1, ci-1)
    #                 newBeam = self.Beam(beamWidth)

    #                 #for the top right most beam, make sure the last poi is end_poi
    #                 if isForceEnd and ci == len(cons) and ti == maxLength-1:
    #                     beam = search_grid[(ti-1, ci)]
    #                     for route in beam:
    #                         de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
    #                         de_out = de_out.flatten()

    #                         new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
    #                         new_route.nextInput = self.end_poi
    #                         new_route.nextHidden = de_h
    #                         new_route.nextCell = de_c
    #                         new_route.path = route.path.copy()
    #                         new_route.path.append(self.dataset.tensor2poi(self.end_poi))
    #                         new_route.prob = route.prob + de_out[self.dataset.tensor2poi(self.end_poi)].item()
    #                         newBeam.addRoute(new_route)
    #                     search_grid[(ti, ci)] = newBeam
    #                     continue

    #                 if (ti-1, ci) in search_grid:
    #                     beam = search_grid[(ti-1, ci)]
    #                     for route in beam:
    #                         if len(route.path)>0 and route.path[-1] == self.dataset.tensor2poi(self.end_poi):
    #                             continue
    #                         de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
    #                         de_out = de_out.flatten()
    #                         _, top_i = de_out.topk(len(de_out))
    #                         # choose (beamWidth) number of new routes with max prob
    #                         beamSize = 0
    #                         for poi in top_i:  
    #                             poi = poi.detach().item()
    #                             if beamSize >= beamWidth:
    #                                 break
    #                             # in case generate end_poi before all mandatory pois
    #                             if poi==self.dataset.tensor2poi(self.end_poi) and ci!=(grid_height-1): 
    #                                 continue
    #                             if poi in cons:
    #                                 continue
    #                             if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
    #                                 continue
    #                             if self.isOpenTime and not self.__isOpen(poi):
    #                                 continue
    #                             if poi not in route.path:
    #                                 new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
    #                                 new_route.nextInput = self.dataset.poi2tensor(poi)
    #                                 new_route.nextHidden = de_h
    #                                 new_route.nextCell = de_c
    #                                 new_route.path = route.path.copy()
    #                                 new_route.path.append(poi)
    #                                 new_route.prob = route.prob + de_out[poi].item()
    #                                 newBeam.addRoute(new_route)
    #                                 beamSize += 1

    #                 if (ti-1, ci-1) in search_grid:
    #                     beam = search_grid[(ti-1, ci-1)]
    #                     for route in beam:
    #                         if len(route.path)>0 and route.path[-1] == self.dataset.tensor2poi(self.end_poi):
    #                             continue
    #                         de_out, (de_h, de_c) = self.decoder(route.nextInput, route.nextHidden, route.nextCell, self.encoder_output)
    #                         de_out = de_out.flatten()
    #                         _, top_i = de_out.topk(len(de_out))
    #                         # choose (beamWidth) number of new routes with max prob
    #                         beamSize = 0
    #                         for poi in top_i:  
    #                             poi = poi.detach().item()
    #                             if beamSize >= beamWidth:
    #                                 break
    #                             if poi==self.dataset.tensor2poi(self.end_poi) and ci!=(grid_height-1):           
    #                                 continue
    #                             if poi not in cons:
    #                                 continue
    #                             if self.isWeather and self.__isRainy() and (poi in self.outDoorPOIs):
    #                                 continue
    #                             if self.isOpenTime and not self.__isOpen(poi):
    #                                 continue
    #                             if poi not in route.path:
    #                                 new_route = self.Route(self.decoder.hidden_size, self.dataset.numPOI)
    #                                 new_route.nextInput = self.dataset.poi2tensor(poi)
    #                                 new_route.nextHidden = de_h
    #                                 new_route.nextCell = de_c
    #                                 new_route.path = route.path.copy()
    #                                 new_route.path.append(poi)
    #                                 new_route.prob = route.prob + de_out[poi].item()
    #                                 newBeam.addRoute(new_route)
    #                                 beamSize += 1
                    
    #                 search_grid[(ti, ci)] = newBeam

    #         return search_grid