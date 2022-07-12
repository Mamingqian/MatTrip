import csv
import random
import numpy as np

def get_data(file, array, delimiter = ","):
    i = 0
    with open(file, 'r') as f:
        data = csv.reader(f, delimiter = delimiter)
        for line in data:
            array[i] = line
            i += 1
    return array

def get_tourists(array):
    tourists = []
    for i in array:
        line = array[i]
        if line[0] == "Amusement":
            continue
        tourist_interest = []
        for x in line:
            tourist_interest.append(float(x))
        tourists.append(tourist_interest)
    return tourists

def get_paths(array):
    paths = []
    for i in array:
        line = array[i]
        path = [-1]
        for x in line:
            if x == '':
                break
            path.append(int(x))
        path.append(-2)
        paths.append(path)
    return paths

def get_interest():
    file = "/Users/chenxiaoyang/Library/Mobile Documents/com~apple~CloudDocs/Project LearningTour /Keras/data/Toronto/fake_user.csv"
    file_read = {}
    get_data(file, file_read)
    return get_tourists(file_read)

def get_routes():
    file = "/Users/chenxiaoyang/Library/Mobile Documents/com~apple~CloudDocs/Project LearningTour /Keras/data/Toronto/fake_path.csv"
    file_read = {}
    get_data(file, file_read)
    return get_paths(file_read)

def combine(input_interest, target_routes):
    input_vectors = []
    for i in range(len(input_interest)):
        vector = []
        for x in input_interest[i]:
            vector.append(x)
        vector.append(target_routes[i][1])
        vector.append(target_routes[i][-2])
        input_vectors.append(vector)
    return input_vectors

def transform(poi_location, x_min, x_max, y_min, y_max):
    transform_poi_location = {}
    for x in poi_location:
        location = poi_location[x]
        transform_location = [(location[0] - x_min) / (x_max - x_min), (location[1] - y_min) / (y_max - y_min)]
        transform_poi_location[x] = transform_location
    return transform_poi_location

def embed(target_pois):
    file = "/Users/chenxiaoyang/Library/Mobile Documents/com~apple~CloudDocs/Project LearningTour /Keras/User Interest and PoI Visit Durations/TourismRecommendingProjectFinal-master/data-ijcai15/poiList-ijcai15/POI-Osak.csv"
    file_read = {}
    get_data(file, file_read, delimiter = ";")
    embeded_target_pois = dict()
    poi_location = {}
    for i in file_read:
        line = file_read[i]
        if line[0] == 'poiID':
            continue
        location = [float(line[2]), float(line[3])]
        poi_location[int(line[0])] = location
    poi_location[-1] = [random.uniform(34, 35), random.uniform(135, 136)]
    poi_location[-2] = [random.uniform(34, 35), random.uniform(135, 136)]
    transform_poi_location = transform(poi_location, 34.6, 34.74, 135.42, 135.54)
    for x in target_pois:
        embeded_target_pois[x] = transform_poi_location[x]
    return embeded_target_pois

def random_test():
    i1 = random.uniform(0, 2)
    i2 = random.uniform(0, 2)
    i3 = random.uniform(0, 2)
    i4 = random.uniform(0, 2)
    i5 = random.uniform(0, 2)
    i6 = random.uniform(0, 2)
    test = np.zeros((1, 8, 1), dtype='float32')
    start = random.randint(1, 30)
    end = random.randint(1, 30)
    while start == 5 or end == 5 or start == end:
        start = random.randint(1, 30)
        end = random.randint(1, 30)
    test[0, 0, 0] = i1
    test[0, 1, 0] = i2
    test[0, 2, 0] = i3
    test[0, 3, 0] = i4
    test[0, 4, 0] = i5
    test[0, 5, 0] = i6
    test[0, 6, 0] = start
    test[0, 7, 0] = end
    print([i1, i2, i3, i4, i5, i6])
    print([start, end])
    return test

