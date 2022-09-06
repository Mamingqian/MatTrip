import TourRecAlgo as TRC
path = 'C:\\Users\\Stav Yanovsky\\Downloads\\data-ijcai15'
pois, tourists, costs = TRC.create_data(path)
all_allocations = TRC.tourist_2_group(tourists)
evaluation_tourists_2_group = TRC.evaluate_tourist_2_group(all_allocations)
TRC.write_evaluation(evaluation_tourists_2_group, "tourists2group_osaka") # needs to change
all_pois_2_group_normal_I1, all_pois_2_group_normal_I_5 = TRC.poi_2_group(pois, all_allocations, costs, improved=False)
all_pois_2_group_improved_I1, all_pois_2_group_improved_I_5 = TRC.poi_2_group(pois, all_allocations, costs, improved=True)
evaluation_pois_2_group_normal_I1 = TRC.evaluate_pois_2_group(all_pois_2_group_normal_I1, all_allocations, pois)
evaluation_pois_2_group_normal_I_5 = TRC.evaluate_pois_2_group(all_pois_2_group_normal_I_5, all_allocations, pois)
evaluation_pois_2_group_improved_I1 = TRC.evaluate_pois_2_group(all_pois_2_group_improved_I1, all_allocations, pois)
evaluation_pois_2_group_improved_I_5 = TRC.evaluate_pois_2_group(all_pois_2_group_improved_I_5, all_allocations, pois)
TRC.write_evaluation(evaluation_pois_2_group_normal_I1, "pois2group_normal_I1")
TRC.write_evaluation(evaluation_pois_2_group_normal_I_5, "pois2group_normal_I_5")
TRC.write_evaluation(evaluation_pois_2_group_improved_I1, "pois2group_improved_I1")
TRC.write_evaluation(evaluation_pois_2_group_improved_I_5, "pois2group_improved_I_5")




