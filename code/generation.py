import torch
import operator
import copy

from data import *
from model import *

# 使用贪心策略生成路径
def greedyRoute(encoder, decoder, en_i, isBi = False):
    with torch.no_grad():
        input_length = en_i.shape[0]
        start_poi = poi2tensor(int(en_i[6].item()))
        end_poi = poi2tensor(int(en_i[7].item()))

        if start_poi.equal(end_poi): 
            return [int(en_i[6].item())]

        if isBi:
            # 改进模型的情况
            encoder_output = encoder(en_i)
            de_in = start_poi
            de_h = decoder.initHidden()
            de_c = decoder.initCell()

            targetRoute = [tensor2poi(start_poi)]

            while not de_in.equal(end_poi):
                de_out, de_h, de_c = decoder(de_in, de_h, de_c, encoder_output)
                de_out = de_out.flatten()
                _, top_i = de_out.topk(len(de_out))
                for i in range(len(top_i)):
                    if top_i[i].item() not in targetRoute:   # 所有
                        de_in = poi2tensor(top_i[i])
                        targetRoute.append(top_i[i].item())
                        break
            return targetRoute

        else:
            # 普通模型的情况
            # encoder forward
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(en_i[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output.reshape(-1)

            # decoder forward
            decoder_hidden = encoder_hidden
            decoder_input = start_poi
            targetRoute = [tensor2poi(start_poi)]

            while not decoder_input.equal(end_poi):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                decoder_output = decoder_output.flatten()
                _, top_i = decoder_output.topk(len(decoder_output))

                for i in range(len(top_i)):
                    if top_i[i].item() not in targetRoute:
                        decoder_input = poi2tensor(top_i[i])
                        targetRoute.append(top_i[i].item())
                        break
            return targetRoute








# 在BSRoute中使用到的类
class Route():
    def __init__(self, hidden_size):
        self.path = []
        self.prob = 0
        self.nextInput = torch.zeros(1, numPOI)
        self.nextHidden = torch.zeros(1, hidden_size)
        self.nextCell = torch.zeros(1, hidden_size)

# 使用Beam Search方法生成路径
# generate routes with beam search
def BSRoute(encoder, decoder, en_i, beam_width = 3, isBi = False):
    with torch.no_grad():
        input_length = en_i.shape[0] #8
        start_poi = poi2tensor(int(en_i[6].item()))
        end_poi = poi2tensor(int(en_i[7].item()))

        if start_poi.equal(end_poi):
            return [int(en_i[6].item())]

        if isBi:
            # 改进模型的情况
            encoder_output = encoder(en_i)

            init_route = Route(decoder.hidden_size)
            init_route.path.append(int(en_i[6].item()))
            init_route.nextInput = start_poi
            init_route.nextHidden = decoder.initHidden()
            init_route.nextCell = decoder.initCell()
            routes = [init_route]

            while True:
                # iterate for each route in routes, no more than (beam_width) routes
                newRoutes = [] # 用来取代routes
                for route in routes:
                    decoder_output, decoder_hidden, decoder_cell = decoder(route.nextInput, route.nextHidden, route.nextCell, encoder_output)
                    decoder_output = decoder_output.flatten()
                    top_v, top_i = decoder_output.topk(len(decoder_output))

                    # choose (beam_width) number of new routes with max prob
                    beamCount = 0
                    i = 0
                    while beamCount < beam_width and i < len(top_i):
                        if top_i[i].item() not in route.path:
                            new_route = copy.deepcopy(route)
                            new_route.nextInput = poi2tensor(top_i[i])
                            new_route.nextHidden = decoder_hidden
                            new_route.nextCell = decoder_cell

                            new_route.path.append(top_i[i].item())
                            new_route.prob += top_v[i].item()
                            # if en_i[7].item() == top_i[i].item():
                            #     return new_route.path
                            newRoutes.append(new_route)
                            beamCount += 1
                        i += 1

                # choose (beam_width) routes out of (beam_width*beam_width) routes
                cmpProb = operator.attrgetter('prob')
                newRoutes.sort(key = cmpProb, reverse = True)
                routes = newRoutes[:beam_width]

                for route in routes:
                    if en_i[7].item() == route.path[-1]:
                        return route.path


        else:
            # 普通模型的情况
            # encoder forward
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(en_i[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output.reshape(-1)

            # decoder forward
            init_route = Route(decoder.hidden_size)
            init_route.path.append(int(en_i[6].item()))
            init_route.nextInput = start_poi
            init_route.nextHidden = encoder_hidden
            routes = [init_route]

            while True:
                # iterate for each route in routes, no more than (beam_width) routes
                newRoutes = []
                for route in routes:
                    decoder_output, decoder_hidden = decoder(route.nextInput, route.nextHidden)
                    decoder_output = decoder_output.flatten()
                    top_v, top_i = decoder_output.topk(len(decoder_output))

                    # choose (beam_width) number of new routes with max prob
                    beamCount = 0
                    i = 0
                    while beamCount < beam_width and i < len(top_i):
                        if top_i[i].item() not in route.path:
                            new_route = copy.deepcopy(route)
                            new_route.nextInput = poi2tensor(top_i[i])
                            new_route.nextHidden = decoder_hidden
                            new_route.path.append(top_i[i].item())
                            new_route.prob += top_v[i].item()
                            if en_i[7].item() == top_i[i].item():
                                return new_route.path
                            newRoutes.append(new_route)
                            beamCount += 1
                        i += 1

                # choose (beam_width) routes out of (beam_width*beam_width) routes
                cmpProb = operator.attrgetter('prob')
                newRoutes.sort(key = cmpProb, reverse = True)
                routes = newRoutes[:beam_width]








"""
Constraints :
1. POI Opening Time
2. Weather Condition
3. Mandatory POIs
"""

# 随机生成outdoor的POI
def fakeOutDoorPOIs(prob):
    outDoorPOIs = []
    for i in range(numPOI):
        if random.random() < prob:
            outDoorPOIs.append(i)
    return outDoorPOIs

# 获得当前的天气状况
def getCurrentWeather(prob):
    return random.random() < prob

# 当前POI是否在营业
def isPOIOpen(prob, POI):
    return random.random() < prob


def ConsBSRoute(encoder, decoder, en_i, beam_width = 3):
    with torch.no_grad():
        prob = 0.3
        outDoorPOIs = fakeOutDoorPOIs(prob)
        input_length = en_i.shape[0] #8
        start_poi = poi2tensor(int(en_i[6].item()))
        end_poi = poi2tensor(int(en_i[7].item()))

        if start_poi.equal(end_poi):
            return [int(en_i[6].item())]
        # 最后的POI不作为outDoorPOIs考虑
        if int(en_i[7].item()) in outDoorPOIs:
            outDoorPOIs.remove(int(en_i[7].item()))

        # encoder forward
        encoder_output = encoder(en_i)

        # decoder forward
        init_route = Route(decoder.hidden_size)
        init_route.path.append(int(en_i[6].item()))
        init_route.nextInput = start_poi
        init_route.nextHidden = decoder.initHidden()
        init_route.nextCell = decoder.initCell()
        routes = [init_route]

        while True:
            # iterate for each route in routes, no more than (beam_width) routes
            newRoutes = [] # 用来取代routes
            for route in routes: # 对于routes里面的beam_width个route进行循环
                decoder_output, decoder_hidden, decoder_cell = decoder(route.nextInput, route.nextHidden, route.nextCell, encoder_output)
                decoder_output = decoder_output.flatten()
                top_v, top_i = decoder_output.topk(len(decoder_output))

                # choose (beam_width) number of new routes with max prob
                beamCount = 0
                i = 0
                while beamCount < beam_width and i < len(top_i):
                    if (top_i[i].item() not in route.path) and isPOIOpen(0.3, i) and not(getCurrentWeather and (i in outDoorPOIs)) : # constraint
                        new_route = copy.deepcopy(route)
                        new_route.nextInput = poi2tensor(top_i[i])
                        new_route.nextHidden = decoder_hidden
                        new_route.nextCell = decoder_cell

                        new_route.path.append(top_i[i].item())
                        new_route.prob += top_v[i].item()
                        # if en_i[7].item() == top_i[i].item():
                        #     return new_route.path
                        newRoutes.append(new_route)
                        beamCount += 1
                    i += 1

            # choose (beam_width) routes out of (beam_width*beam_width) routes
            cmpProb = operator.attrgetter('prob')
            newRoutes.sort(key = cmpProb, reverse = True)
            routes = newRoutes[:beam_width]

            for route in routes:
                if en_i[7].item() == route.path[-1]:
                    return route.path








# Fast Recalculation
def ReBSRoute(encoder, decoder, en_i, path, beam_width = 3):
    if len(path)==0:
        return BSRoute(encoder, decoder, en_i)

    with torch.no_grad():
        input_length = en_i.shape[0] #8
        start_poi = poi2tensor(int(en_i[6].item()))
        end_poi = poi2tensor(int(en_i[7].item()))

        # encoder forward
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(en_i[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output.reshape(-1)

        # decoder forward
        init_route = Route(decoder.hidden_size)
        init_route.path += path
        init_route.nextInput = poi2tensor(path[-1])

        decoder_hidden = encoder_hidden
        for di in range(len(path)-1):
            _, decoder_hidden = decoder(poi2tensor(path[di]), decoder_hidden)

        init_route.nextHidden = decoder_hidden
        routes = [init_route]

        while True:
            # iterate for each route in routes, no more than (beam_width) routes
            newRoutes = []
            for route in routes:
                decoder_output, decoder_hidden = decoder(route.nextInput, route.nextHidden)
                decoder_output = decoder_output.flatten()
                top_v, top_i = decoder_output.topk(len(decoder_output))

                # choose (beam_width) number of new routes with max prob
                beamCount = 0
                i = 0
                while beamCount < beam_width and i < len(top_i):
                    if top_i[i].item() not in route.path:
                        new_route = copy.deepcopy(route)
                        new_route.nextInput = poi2tensor(top_i[i])
                        new_route.nextHidden = decoder_hidden
                        new_route.path.append(top_i[i].item())
                        new_route.prob += top_v[i].item()
                        if en_i[7].item() == top_i[i].item():
                            return new_route.path
                        newRoutes.append(new_route)
                        beamCount += 1
                    i += 1

            # choose (beam_width) routes out of (beam_width*beam_width) routes
            cmpProb = operator.attrgetter('prob')
            newRoutes.sort(key = cmpProb, reverse = True)
            routes = newRoutes[:beam_width]