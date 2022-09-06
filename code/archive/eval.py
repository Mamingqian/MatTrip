import csv
from data import get_interest
from TourRecAlgo import *
import time
import os

'''tourists = []
path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\RNN\\data\\Toronto"
with open(os.path.join(path, "fake_user.txt"), 'r') as f:
    read = csv.reader(f, delimiter=',')
    for line in read:
        list = line[0].split(" ")
        tourists.append(list)
fake_tourists = generate_tourist(tourists)

with open(os.path.join(path, "fake_path.txt"), 'w+') as f:
    for x in fake_tourists:
        start = random.randint(1, 30)
        end = random.randint(1, 30)
        while start == end or start == 5 or end == 5:
            start = random.randint(1, 30)
            end = random.randint(1, 30)
        start = str(start)
        end = str(end)
        print(start + " " + end)
        fake_path = poi_2_group(pois, x, costs, start, end)
        for x in fake_path:
            f.write(x)
            f.write(" ")
        f.write('\n')'''
path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\Result\\Osaka\\result.csv"
tourists = []
paths = []
starts = []
ends = []
runtime = []
file = []
with open(path, 'r') as f:
    read = csv.reader(f)
    for line in read:
        file.append(line)

for i in range(len(file)):
    line = file[i]
    if i % 4 == 0:
        tourists.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
    elif i % 4 == 1:
        starts.append(line[0])
        ends.append(line[1])
    elif i % 4 == 2:
        path = []
        for x in line:
            if x == '':
                break
            path.append(x)
        paths.append(path)
    else:
        runtime.append(float(line[0]))

print(tourists)
base_tourists = generate_tourist(tourists)
base_path = 'D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\baseline\\User Interest and PoI Visit Durations\\data'
pois, _, costs = create_data(base_path)

def revise(path, start, end):
    revised_path = list(path)
    if revised_path[0] != start:
        if start in revised_path:
            tmp_index = revised_path.index(start)
            revised_path[0], revised_path[tmp_index] = revised_path[tmp_index], revised_path[0]
        else:
            revised_path[0] = start
    if revised_path[-1] != end:
        if end in revised_path:
            tmp_index = revised_path.index(end)
            revised_path[tmp_index], revised_path[-1] = revised_path[-1], revised_path[tmp_index]
        else:
            revised_path[-1] = end
    return revised_path

def compute_cost(path, costs, tourist, pois):
    Costs = create_costs_for_pois(costs)
    interest_in_concept = tourist.interest_in_concept
    interest_in_pois = concept_to_pois(interest_in_concept, pois)
    duration = duration_time(Costs, interest_in_pois, pois)
    total_cost = total(Costs, duration)
    cost = 0
    for i in range(len(path) - 1):
        cost += total_cost[(path[i], path[i + 1])]
    return cost

'''cnt = 0
revised_paths = []
for i in range(len(paths)):
    path = paths[i]
    revised_path = revise(path, starts[i], ends[i])
    revised_paths.append(revised_path)
    if compute_cost(revised_path, costs, base_tourists[i], pois) > 28800:
        cnt += 1
print(cnt / 500)'''

'''for x in range(len(base_tourists)):
    start = time.clock()
    base_path = random_route(pois, base_tourists[x], costs, starts[x], ends[x])
    end = time.clock()
    run = end - start
    print(run)
    print(base_path)'''

file = []
tmp_path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\Result\\Osaka\\baseline_1.csv"
with open(tmp_path, "r") as f:
    read = csv.reader(f)
    for line in read:
        file.append(line)
baseline_routes = []
baseline_runtime = []
for i in range(len(file)):
    baseline_route = []
    if i % 2 == 1:
        for x in file[i]:
            if x == '':
                break
            baseline_route.append(x)
        baseline_routes.append(baseline_route)
    else:
        print(file[i][0])
        baseline_runtime.append(float(file[i][0]))
for x in baseline_runtime:
    print(x)

'''print()
print()
for i in range(len(base_tourists)):
    score = eval_route(pois, base_tourists[i], costs, baseline_routes[i])
    print(score)'''

