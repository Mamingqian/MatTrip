import itertools
import os
import csv
import datetime
import random
import numpy as np
from pyomo.core.kernel import value
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import jaccard_similarity_score
from scipy import spatial
from pyomo.environ import *
from pyomo import *
from collections import OrderedDict

alpha = 0.25
beta = 0.25
gamma = 0.25
delta = 0.25


def categories(x):  # needs to change
    return {
        'Sport': 0,
        'Cultural': 1,
        'Amusement': 2,
        'Beach': 3,
        'Shopping': 4,
        'Structure': 5
    }[x]
# {"Sport": 0, "Cultural": 0, "Amusement": 0, "Beach": 0, "Shopping": 0, "Structure": 0}  # needs to change toronto
# {"Amusement": 0, "Park": 0, "Historical": 0, "Entertainment": 0} # osaka

class POI:
    def __init__(self, poi_id, poi_name, poi_lat, poi_long, poi_theme):
        self.id = poi_id
        self.name = poi_name
        self.lat = poi_lat
        self.long = poi_long
        self.concept = poi_theme
        self.popularity = 0
        self.norm_pop = 0
        self.norm_improved_pop = 0
        self.total_visit_time = 0
        self.total_visit_num = 0
        self.average_visit_time = 0


class Tourist:
    def __init__(self, user_id, seq_id, visit):
        self.id = user_id
        self.travel_sequence = {seq_id: [visit]}
        # self.total_visiting_time_in_concepts = {"Sport": 0, "Cultural": 0, "Amusement": 0, "Beach": 0, "Shopping": 0, "Structure": 0}  # needs to change
        #self.interest_in_concept = {"Sport": 0, "Cultural": 0, "Amusement": 0, "Beach": 0, "Shopping": 0, "Structure": 0}  # needs to change
        self.total_visit_time_in_poi = {}
        self.interest_in_concept = {"Amusement": 0, "Park": 0, "Historical": 0, "Entertainment": 0}  # needs to change
        self.binary_interest = []

    def get_interest(self):
        to_ret = np.zeros(6)  # needs to change
        i = 0
        for key, value in self.interest_in_concept.items():
            to_ret[i] = value
            i += 1
        return to_ret


class POI_Visit:
    def __init__(self, t_id, poi_id, arrival_time):
        self.t_id = t_id
        self.poi_id = poi_id
        self.arrival_time = arrival_time
        self.departure_time = datetime.datetime.strptime(self.arrival_time, "%Y-%m-%d %H:%M:%S")+datetime.timedelta(0, 1)
        self.duration = 1

    def set_departure(self, departue_time):
        self.departure_time = departue_time
        d1 = datetime.datetime.strptime(self.departure_time, "%Y-%m-%d %H:%M:%S")
        d2 = datetime.datetime.strptime(self.arrival_time, "%Y-%m-%d %H:%M:%S")
        self.duration = (d1 - d2).seconds


class Group:
    def __init__(self):
        self.interest = 0
        self.tourists = {}

    def compute_group_interest(self):
        to_mean = []
        for id in self.tourists:
            to_add = self.tourists[id].get_interest()
            to_mean.append(to_add)
        self.interest = np.mean(to_mean, axis=0)


class TripAdvisorRecord:
    def __init__(self, id, score_of_5, num_of_revs, place_in_list, total_in_list):
        self.poi_id = id
        self.score_of_5 = score_of_5
        self.num_of_reviews = num_of_revs
        self.place_in_ttd_list = place_in_list
        self.amount_in_ttd_list = total_in_list


def readTripAdvisorData(path, city_name):
    TAdata = {}
    path_of_tripAdvisor_file = os.path.join(path, "tripAdvisor\\" + city_name + ".csv")
    with open(path_of_tripAdvisor_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if line[0] == "id":
                continue
            id = line[0]
            score_of_5 = float(line[1])
            num_of_revs = int(line[2])
            place_in_list = int(line[3])
            total_in_list = int(line[4])
            TAdata[id] = TripAdvisorRecord(id, score_of_5, num_of_revs, place_in_list, total_in_list)
    return TAdata


def compute_improved_pop(poi, tripAdvisorRecord):
    try:
        improved_pop = alpha * poi.norm_pop + beta * tripAdvisorRecord.score_of_5 / 5 + gamma * (1 - 1 / tripAdvisorRecord.num_of_reviews) + delta * (1 - tripAdvisorRecord.place_in_ttd_list * 1.0 / tripAdvisorRecord.amount_in_ttd_list)
    except:
        improved_pop = 0
    return improved_pop

def create_tourists():
    tourists = {}
    for i in range(3000):
        tmp = Tourist(i, 0, 0)
        r1 = random.uniform(0, 2)
        r2 = random.uniform(0, 2)
        r3 = random.uniform(0, 2)
        r4 = random.uniform(0, 2)
        r5 = random.uniform(0, 2)
        r6 = random.uniform(0, 2)
        tmp.interest_in_concept["Sport"] = r1
        tmp.interest_in_concept["Cultural"] = r2
        tmp.interest_in_concept["Amusement"] = r3
        tmp.interest_in_concept["Beach"] = r4
        tmp.interest_in_concept["Shopping"] = r5
        tmp.interest_in_concept["Structure"] = r6
        tourists[i] = tmp
    return tourists

def generate_tourist(tourists):
    Tourists = []
    for x in tourists:
        tmp = Tourist(0, 0, 0)
        '''tmp.interest_in_concept["Sport"] = float(x[0])
        tmp.interest_in_concept["Cultural"] = float(x[1])
        tmp.interest_in_concept["Amusement"] = float(x[2])
        tmp.interest_in_concept["Beach"] = float(x[3])
        tmp.interest_in_concept["Shopping"] = float(x[4])
        tmp.interest_in_concept["Structure"] = float(x[5])'''
        tmp.interest_in_concept["Amusement"] = float(x[0])
        tmp.interest_in_concept["Park"] = float(x[1])
        tmp.interest_in_concept["Historical"] = float(x[2])
        tmp.interest_in_concept["Entertainment"] = float(x[3])
        Tourists.append(tmp)
    return Tourists

def create_data(path):
    all_travel_seqs = {}
    tripAdvisorData = readTripAdvisorData(path, "Osaka")  # needs to change
    max_pop = 0
    pois = {}
    i = 0
    directory_of_file = os.path.join(path, "poiList-ijcai15")
    dir = os.listdir(directory_of_file)
    for file in dir:
        filename = os.fsdecode(file)
        if filename != "POI-Osak.csv":  # needs to change
            continue
        with open(os.path.join(directory_of_file, file), "r") as f:
            reader = csv.reader(f, delimiter=";")
            for line in reader:
                if line[0] == "poiID":
                    continue
                poi_id = line[0]
                poi_name = line[1]
                poi_lat = line[2]
                poi_long = line[3]
                poi_con = line[4]
                pois[poi_id] = POI(poi_id, poi_name, poi_lat, poi_long, poi_con)

    tourists = {}
    directory_of_file = os.path.join(path, "userVisits-ijcai15")
    dir = os.listdir(directory_of_file)
    for file in dir:
        filename = os.fsdecode(file)
        if filename != "userVisits-Osak.csv":  # needs to change
            continue
        with open(os.path.join(directory_of_file, file), "r") as f:
            reader = csv.reader(f, delimiter=";")
            for line in reader:
                if line[0] == "photoID":
                    continue
                i += 1
                t_id = line[1]
                seq_id = line[6]
                poi_id = line[3]
                poi_pop = line[5]
                if int(poi_pop) > max_pop:
                    max_pop = int(poi_pop)
                pois[poi_id].popularity = poi_pop  # original algorithm
                try:
                    time = datetime.datetime.fromtimestamp(int(line[2])).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    continue
                if t_id not in tourists.keys():
                    visit = POI_Visit(t_id, poi_id, time)
                    tourists[t_id] = Tourist(t_id, seq_id, visit)
                else:
                    if seq_id not in tourists[t_id].travel_sequence.keys():
                        visit = POI_Visit(t_id, poi_id, time)
                        tourists[t_id].travel_sequence[seq_id] = [visit]
                    else:
                        last_visit = tourists[t_id].travel_sequence[seq_id][-1]
                        if last_visit.poi_id == poi_id:
                            tourists[t_id].travel_sequence[seq_id][-1].set_departure(time)
                        else:
                            visit = POI_Visit(t_id, poi_id, time)
                            tourists[t_id].travel_sequence[seq_id].append(visit)
    for poi in pois:
        pois[poi].norm_pop = int(pois[poi].popularity) / int(max_pop)
        pois[poi].norm_improved_pop = compute_improved_pop(pois[poi], tripAdvisorData[poi])

    for tourist in tourists:
        for key in tourists[tourist].travel_sequence:
            for x in tourists[tourist].travel_sequence[key]:
                pois[x.poi_id].total_visit_time += x.duration
                pois[x.poi_id].total_visit_num += 1

    # iterate over travel sequences and remove travel sequences with less than 3 pois
    to_delete_seq = []
    to_delete_tourists = []
    for tourist in tourists:
        for travel_seq in tourists[tourist].travel_sequence:
            if tourists[tourist].travel_sequence[travel_seq].__len__() < 3:
                to_delete_seq.append(tourist+":"+travel_seq)
    for item in to_delete_seq:
        tourist = item.split(":")[0]
        travel_seq = item.split(":")[1]
        tourists[tourist].travel_sequence.pop(travel_seq)
    for tourist in tourists:
        if tourists[tourist].travel_sequence.keys().__len__() == 0:
            to_delete_tourists.append(tourist)
    for t_id in to_delete_tourists:
        tourists.pop(t_id)

    '''path1 = 'D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\baseline\\User Interest and PoI Visit Durations\\data\\tourist\\' + "Osaka_route.txt"
    text_file = open(path1, "w+")
    for tourist in tourists:
        for key in tourists[tourist].travel_sequence:
            text_file.write(tourists[tourist].id + " : ")
            for x in tourists[tourist].travel_sequence[key]:
                text_file.write(x.poi_id + " , " + x.arrival_time + " " + str(x.duration) + " ; ")
            text_file.write('\n')
    text_file.close()'''

    for x in pois:
        try:
            pois[x].average_visit_time = float(pois[x].total_visit_time) / pois[x].total_visit_num
        except:
            pois[x].average_visit_time = 0

    # go over all visits of all tourists and calculate interest in each concept
    for tourist in tourists:
        for travel_seq in tourists[tourist].travel_sequence:
            print("starts: " + tourists[tourist].travel_sequence[travel_seq][0].poi_id + " ends: " + tourists[tourist].travel_sequence[travel_seq][-1].poi_id)
            try:
                all_travel_seqs["starts: " + tourists[tourist].travel_sequence[travel_seq][0].poi_id + " ends: " + tourists[tourist].travel_sequence[travel_seq][-1].poi_id] += 1
            except:
                all_travel_seqs["starts: " + tourists[tourist].travel_sequence[travel_seq][0].poi_id + " ends: " + tourists[tourist].travel_sequence[travel_seq][-1].poi_id] = 1
            for visit in tourists[tourist].travel_sequence[travel_seq]:
                poi_id = visit.poi_id
                poi_concept = pois[poi_id].concept
                try:
                    tourists[tourist].total_visit_time_in_poi[poi_id] += visit.duration
                except:
                    tourists[tourist].total_visit_time_in_poi[poi_id] = visit.duration
                tourists[tourist].interest_in_concept[poi_concept] += float(visit.duration) / pois[poi_id].average_visit_time

    '''path1 = 'D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\baseline\\User Interest and PoI Visit Durations\\data\\tourist\\' + "Osaka_user.txt"
    text_file = open(path1, "w+")
    for tourist in tourists:
        text_file.write(tourists[tourist].id + " ")
        for x in tourists[tourist].interest_in_concept:
            text_file.write(x + " : " + str(tourists[tourist].interest_in_concept[x]) + " ")
        text_file.write('\n')
    text_file.close()'''

    # binary version of interest in concept for jaccard
    for tourist in tourists:
        for concept in tourists[tourist].interest_in_concept:
            if tourists[tourist].interest_in_concept[concept] == 0:
                tourists[tourist].binary_interest.append(0)
            else:
                tourists[tourist].binary_interest.append(1)
    costs = {}
    directory_of_file = os.path.join(path, "costProf-ijcai15")
    dir = os.listdir(directory_of_file)
    for file in dir:
        filename = os.fsdecode(file)
        if filename != "costProfCat-OsakPOI-all.csv":  # needs to change
            continue
        with open(os.path.join(directory_of_file, file), "r") as f:
            reader = csv.reader(f, delimiter=";")
            for line in reader:
                if line[0] == "from":
                    continue
                poi_from = line[0]
                poi_to = line[1]
                cost = line[2]
                costs[poi_from+"-"+poi_to] = cost
    all_travel_seqs_sorted_by_value = OrderedDict(sorted(all_travel_seqs.items(), key=lambda x: x[1]))

    print("all travel seqs : ")
    print(all_travel_seqs_sorted_by_value)
    return pois, tourists, costs


def create_df_to_cluster(tourists, set_of_tourists):
    X = []
    list_of_tourist_ids = []
    for id in set_of_tourists:
        to_add = tourists[id].get_interest()
        list_of_tourist_ids.append(id)
        X.append(to_add)
    return X, list_of_tourist_ids

def tourist_2_group(tourists):
    # first, sample 100 random 100 users
    number_of_tourists = min(100, len(tourists))
    n_clusters = 5
    set_of_tourists = set()
    r = random.Random()
    while len(set_of_tourists) < number_of_tourists:
        set_of_tourists.add(r.choice(list(tourists)))

    all_allocations = {"kmean": {}, "hiera": {}, "fcfa": {}, "rand": {}}
    X, list_of_tourist_ids = create_df_to_cluster(tourists, set_of_tourists)
    i = 0
    while True:
        kmeans = KMeans(init='random', n_clusters=n_clusters).fit(X)
        if (kmeans.labels_ == 0).sum() > 1 and (kmeans.labels_ == 1).sum() > 1 and (kmeans.labels_ == 2).sum() > 1 and (kmeans.labels_ == 3).sum() > 1 and (kmeans.labels_ == 4).sum() > 1:
            break
    for group_id in kmeans.labels_:
        try:
            all_allocations["kmean"][group_id].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        except:
            all_allocations["kmean"][group_id] = Group()
            all_allocations["kmean"][group_id].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        i += 1
    i = 0
    hiera = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    for group_id in hiera.labels_:
        try:
            all_allocations["hiera"][group_id].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        except:
            all_allocations["hiera"][group_id] = Group()
            all_allocations["hiera"][group_id].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        i += 1
    # baselines
    for i in range(5):
        all_allocations["fcfa"][i] = Group()
    group_size = number_of_tourists / 5
    for i in range(number_of_tourists):
        if i < group_size:
            all_allocations["fcfa"][0].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        elif i < 2 * group_size:
            all_allocations["fcfa"][1].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        elif i < 3 * group_size:
            all_allocations["fcfa"][2].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        elif i < 4*group_size:
            all_allocations["fcfa"][3].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
        else:
            all_allocations["fcfa"][4].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
    for i in range(5):
        all_allocations["rand"][i] = Group()
    r = random.Random(500)
    for i in range(number_of_tourists):
        group_id = r.choice([0, 1, 2, 3, 4])
        all_allocations["rand"][group_id].tourists[list_of_tourist_ids[i]] = tourists[list_of_tourist_ids[i]]
    for alloc in all_allocations:
        for group_id, group in all_allocations[alloc].items():
            group.compute_group_interest()
    return all_allocations


def compute_jac_of_group(tourists):
    sum = 0
    count = 0
    for tourist_i in tourists:
        for tourist_j in tourists:
            if tourist_i == tourist_j:
                continue
            count += 1
            sum += jaccard_similarity_score(tourists[tourist_i].binary_interest, tourists[tourist_j].binary_interest)
    try:
        return sum / count
    except:
        return 1

def compute_cos_of_group(tourists):
    sum = 0
    count = 0
    for tourist_i in tourists:
        for tourist_j in tourists:
            if tourist_i == tourist_j:
                continue
            interest_i = tourists[tourist_i].get_interest()
            interest_j = tourists[tourist_j].get_interest()
            count += 1
            sum += (1 - spatial.distance.cosine(interest_i, interest_j))
    try:
        return sum / count
    except:
        return 1


def compute_com_of_group(tourists):
    max = 0
    for i in range(4):  # needs to change
        sum = 0
        for tourist in tourists:
            sum += tourists[tourist].binary_interest[i]
        if 1 / len(tourists) * sum > max:
            max = 1 / len(tourists) * sum
    return max


def evaluate_tourist_2_group(all_allocations):
    all_scores = {"kmean": {"jac": [], "cos": [], "com": []}, "hiera": {"jac": [], "cos": [], "com": []}, "fcfa": {"jac": [], "cos": [], "com": []}, "rand": {"jac": [], "cos": [], "com": []}}
    for alloc in all_allocations:
        for group in all_allocations[alloc]:
            jac = compute_jac_of_group(all_allocations[alloc][group].tourists)
            cos = compute_cos_of_group(all_allocations[alloc][group].tourists)
            com = compute_com_of_group(all_allocations[alloc][group].tourists)
            all_scores[alloc]["jac"].append(jac)
            all_scores[alloc]["cos"].append(cos)
            all_scores[alloc]["com"].append(com)
    return all_scores


def write_evaluation(evaluation_to_write, file_name):
    to_write = "OSAKA:\n\n"
    for alloc in evaluation_to_write:
        to_write += (alloc + ":\n")
        for sim in evaluation_to_write[alloc]:
            if sim == "total" or sim == "max" or sim == "min":
                evaluation_to_write[alloc][sim] = list(itertools.chain.from_iterable(evaluation_to_write[alloc][sim]))
            to_write += (sim + ": " + str(evaluation_to_write[alloc][sim]) + ", mean: " + str(np.mean(evaluation_to_write[alloc][sim], axis=0)) + ", std: " + str(np.std(evaluation_to_write[alloc][sim], axis=0))+"\n")
    path = 'D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\baseline\\User Interest and PoI Visit Durations\\data\\evaluation\\osaka\\1to24\\' + file_name + ".txt"
    text_file = open(path, "w+")
    text_file.write(to_write)
    text_file.close()


def total_rule(model):
    sum = 0
    for i, j in model.A:
        sum += model.r[(i, j)] * (value(model.eta) * model.g_i[i] + (1 - value(model.eta)) * model.p[i])
    return sum


def starts_1(model):
    sum = 0
    for j in model.N:
        if (value(model.s), j) in model.A.value:
            sum += model.r[(model.s, j)]
    return sum == 1


def starts_2(model):
    sum = 0
    for j in model.N:
        if (j, value(model.s)) in model.A.value:
            sum += model.r[(j, model.s)]
    return sum == 0


def ends_1(model):
    sum = 0
    for i in model.N:
        if (i, value(model.t)) in model.A.value:
            sum += model.r[(i, model.t)]
    return sum == 1


def ends_2(model):
    sum = 0
    for i in model.N:
        if (value(model.t), i) in model.A.value:
            sum += model.r[(model.t, i)]
    return sum == 0


def re_visited_1(model, k):
    if k == value(model.s) or k == value(model.t):
        return Constraint.Skip
    found = 0
    for (i, j) in model.A:
        if k in (i, j):
            found = 1
            break
    if found == 0:
        return Constraint.Skip
    into_k = sum(model.r[(i, j)] for i, j in model.A if j == k and i != value(model.t))
    out_of_k = sum(model.r[(i, j)] for i, j in model.A if i == k and j != value(model.s))
    return into_k == out_of_k


def re_visited_2(model, k):
    if k == value(model.s) or k == value(model.t):
        return Constraint.Skip
    found = 0
    for (i, j) in model.A:
        if k in (i, j):
            found = 1
            break
    if found == 0:
        return Constraint.Skip
    into_k = sum(model.r[(i, j)] for i, j in model.A if j == k)
    return into_k <= 1


def re_visited_3(model, i, j):
    return (model.r[(i, j)] + model.r[(j, i)]) <= 1


def costs_(model):
    return sum(model.r[(i, j)] * model.c[(i, j)] for i, j in model.A) <= value(model.b)


def create_pop_of_pois_arr(pois, improved):
    pop_of_pois = {}
    for poi in pois:
        if improved:
            poi_norm_pop = pois[poi].norm_improved_pop
        else:
            poi_norm_pop = pois[poi].norm_pop
        pop_of_pois[poi] = poi_norm_pop
    return pop_of_pois


def create_group_interest_in_pois(group, pois):
    group_interest_in_pois = {}
    for poi in pois:
        group_interest_in_poi = group.interest[categories(pois[poi].concept)]
        group_interest_in_pois[poi] = group_interest_in_poi
    return group_interest_in_pois

def create_costs_for_pois(costs):
    costs_of_pois = {}
    for cost in costs:
        poi_s = cost.split('-')[0]
        poi_t = cost.split('-')[1]
        costs_of_pois[(poi_s, poi_t)] = float(costs[cost])
    return costs_of_pois


def create_set_of_pois(pois):
    set_of_pois = []
    for poi in pois:
        set_of_pois.append(poi)
    return set_of_pois


def find_next_step(all_steps, step_to_find):
    for step in all_steps:
        if step[0] == step_to_find:
            return step


def define_path(model):
    start = value(model.s)
    end = value(model.t)
    all_steps = []
    path = [start]
    for v in model.component_objects(Var, active=True):
        for index in v:
            if v[index].value == 1:
                all_steps.append(index)
    start_step = find_next_step(all_steps, start)
    if type(start_step) == type(None):
        return path
    next = start_step[1]
    path.append(next)
    if next == end:
        end_not_found = False
    else:
        end_not_found = True
    while end_not_found:
        next_step = find_next_step(all_steps, next)
        next = next_step[1]
        path.append(next)
        if next == end:
            end_not_found = False
    return path

def duration_time(costs, interest_in_pois, pois):
    duration = {}
    for x in pois:
        duration[x] = interest_in_pois[x] * pois[x].average_visit_time
    return duration

def total(costs, duration):
    total_cost = {}
    for x in costs:
        total_cost[x] = costs[x] + duration[x[1]]
    return total_cost

def concept_to_pois(interest_in_concept, pois):
    interest_in_pois = {}
    for x in pois:
        interest_in_pois[x] = interest_in_concept[pois[x].concept]
    return interest_in_pois

def poi_2_group(pois, tourist, costs, start, end, improved=True,):
    #print("IMPROVED: " + str(improved))
    eta = 0.5
    #all_pois_2_group_I1 = {"kmean": {}, "hiera": {}, "fcfa": {}, "rand": {}}
    all_pois_2_group_I_5 = {}
    #paths = {}
    num_of_pois = len(pois)
    set_of_pois = create_set_of_pois(pois)
    pop_of_pois = create_pop_of_pois_arr(pois, improved)
    costs = create_costs_for_pois(costs)

    #file_path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\RNN\\data\\Osaka\\fake_st.txt"
    #text = open(file_path, "w+")
    interest_in_concept = tourist.interest_in_concept
    interest_in_pois = concept_to_pois(interest_in_concept, pois)
    duration = duration_time(costs, interest_in_pois, pois)
    total_cost = total(costs, duration)
    model = ConcreteModel()
    # nodes
    model.N = Set(initialize=set_of_pois)
    # arcs
    model.A = Set(within=model.N * model.N, initialize = total_cost.keys())
    # source POI
    model.s = Param(initialize=start, within=model.N)
    # target POI
    model.t = Param(initialize=end, within=model.N)
    # group interest in category of POI i
    model.g_i = Param(model.N, initialize=interest_in_pois)
    # popularity of POI i
    model.p = Param(model.N, initialize=pop_of_pois)
    # cost of getting from POI i to POI j
    model.c = Param(model.A, initialize=total_cost)
    # budget
    model.b = Param(initialize=28800, within=PositiveReals)
    # r(i,j)
    model.r = Var(model.A, within=Binary)
    # objective
    model.eta = Param(initialize=eta)
    model.objective = Objective(rule=total_rule, sense=maximize)
    # constraint 1: recommended tour starts and ends at POI s and t
    model.starts1 = Constraint(rule=starts_1)
    model.starts2 = Constraint(rule=starts_2)
    model.ends1 = Constraint(rule=ends_1)
    model.ends2 = Constraint(rule=ends_2)
    # constraint 2: no POIs are re-visited and all paths are connected
    model.re_visited_1 = Constraint(model.N, rule=re_visited_1)
    model.re_visited_2 = Constraint(model.N, rule=re_visited_2)
    model.re_visited_3 = Constraint(model.A, rule=re_visited_3)
    # constraint 3: the total time needed to visit all POIs in the recommended tour is within the budget B
    model.cost = Constraint(rule=costs_)
    opt = SolverFactory("glpk", executable="D:\\Shanghai JiaoTong University\\Research\\tool\\glpk-4.65\\w64\\glpsol")
    results = opt.solve(model)
    path_of_pois = define_path(model)
    #paths[tourist] = path_of_pois
    #text.close()
    return path_of_pois


def get_interest_of_user_in_poi(tourist, poi, pois):
    return tourist.interest_in_concept[pois[poi].concept]


def compute_total_of_tourists(path, tourists, pois):
    score_for_tourist = 0
    for poi in path:
        score_for_tourist += get_interest_of_user_in_poi(tourists, poi, pois)
    score = score_for_tourist
    return score


def compute_max_of_group(path, tourists, pois):
    max_to_tourist = 0
    for poi in path:
        score_to_tourist = get_interest_of_user_in_poi(tourists, poi, pois)
        if score_to_tourist > max_to_tourist:
            max_to_tourist = score_to_tourist
    max = max_to_tourist
    return max


def compute_min_of_group(path, tourists, pois):
    min_to_tourist = 100
    for poi in path:
        score_to_tourist = get_interest_of_user_in_poi(tourists, poi, pois)
        if score_to_tourist < min_to_tourist:
            min_to_tourist = score_to_tourist
    min = min_to_tourist
    return min

def write_eva(evaluation_to_write, file_name):
    to_write = "OSAKA:\n\n"
    path = 'D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\baseline\\User Interest and PoI Visit Durations\\data\\evaluation\\osaka\\1to24\\' + file_name + ".txt"
    text_file = open(path, "w+")
    text_file.write("OSAKA:\n\n")
    for alloc in evaluation_to_write:
        for sim in evaluation_to_write[alloc]:
            text_file.write(str(alloc) + ":" + sim + ": " + str(evaluation_to_write[alloc][sim]) + "\n")
    text_file.close()

def evaluate_pois_2_group(all_pois_2_group, tourists, pois):
    all_scores = {}
    for alloc in all_pois_2_group:
        total = compute_total_of_tourists(all_pois_2_group[alloc], tourists[alloc], pois)
        max = compute_max_of_group(all_pois_2_group[alloc], tourists[alloc], pois)
        min = compute_min_of_group(all_pois_2_group[alloc], tourists[alloc], pois)
        all_scores[alloc] = {}
        all_scores[alloc]["total"] = total
        all_scores[alloc]["max"]= max
        all_scores[alloc]["min"] = min
    return all_scores

def write_path(tourists, paths):
    write_path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\RNN\\data\\Toronto\\fake_path.txt"
    text_file = open(write_path, "w+")
    for x in paths:
        text_file.write(str(tourists[x].id) + ":[")
        for poi in paths[x]:
            text_file.write(poi + ",")
        text_file.write("]\n")
    text_file.close()

def write_user(tourists):
    path = "D:\\Shanghai JiaoTong University\\Research\\tourist recommendation\\RNN\\data\\Toronto\\fake_user.txt"
    text_file = open(path, "w+")
    for tourist in tourists:
        text_file.write(str(tourists[tourist].id) + " ")
        for x in tourists[tourist].interest_in_concept:
            text_file.write(x + " : " + str(tourists[tourist].interest_in_concept[x]) + " ")
        text_file.write('\n')
    text_file.close()

def random_route(pois, tourist, costs, start, end):
    interest_in_concept = tourist.interest_in_concept
    interest_in_pois = concept_to_pois(interest_in_concept, pois)
    Costs = create_costs_for_pois(costs)
    duration = duration_time(Costs, interest_in_pois, pois)
    total_cost = total(Costs, duration)
    path = [start]
    candidate = []
    for x in pois:
        if pois[x].id != start and pois[x].id != end and pois[x].id != '13' and pois[x].id != '14':
            candidate.append(pois[x].id)
    stop = False
    left_time = 28800
    while not stop:
        current = path[-1]
        next_poi = random.choice(candidate)
        path.append(next_poi)
        candidate.remove(next_poi)
        left_time -= float(total_cost[(current, next_poi)])
        if left_time < float(total_cost[(next_poi, end)]):
            stop = True
    path.append(end)
    return path

def eval_route(pois, tourist, costs, path):
    score = 0
    for poi in path:
        score += get_interest_of_user_in_poi(tourist, poi, pois) + pois[poi].norm_improved_pop
    return score