==================================================================
POI Cost-Profit Table
==================================================================

Dataset Information: 
This dataset comprises various cost-profit tables (for eight cities) that indicate the cost (based on distance) required to travel from one points-of-interest (POI) to another POI, and the resulting profit (based on popularity) gained from reaching that POI. These cost-profit tables are derived from the "Flickr User-POI Visits" dataset, i.e., the various "userVisits-{cityName}.csv" files from "data-ijcai15.zip".

File Description:
There are a total of eight files, each named "costProfCat-{cityName}POI-all.csv", where each row indicated a link from one POI to another and the associated cost (distance), profit (popularity), theme (category) and lat/lon coordinates.

The cost-profit table for each city is stored in a single csv file that contains the following columns/fields:
 - from: poiID of the starting POI.
 - to: poiID of the destination POI.
 - cost: distance (metres) between the starting POI (from) to the destination POI (to).
 - profit: popularity of the destination POI (to), based on number of POI visits
 - theme: category of the POI (e.g., Park, Museum, Cultural, etc).

------------------------------------------------------------------
References / Citations
------------------------------------------------------------------
If you use this dataset, please cite the following papers:
 - Kwan Hui Lim, Jeffrey Chan, Christopher Leckie and Shanika Karunasekera. "Personalized Tour Recommendation based on User Interests and Points of Interest Visit Durations". In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15). Pg 1778-1784. Jul 2015.
 - Kwan Hui Lim, Jeffrey Chan, Christopher Leckie and Shanika Karunasekera. "Towards Next Generation Touring: Personalized Group Tours". In Proceedings of the 26th International Conference on Automated Planning and Scheduling (ICAPS'16). Pg 412-420. Jun 2016.

The corresponding bibtex for these papers are:
 @INPROCEEDINGS { lim-ijcai15,
	AUTHOR = {Kwan Hui Lim and Jeffrey Chan and Christopher Leckie and Shanika Karunasekera},
	TITLE = {Personalized Tour Recommendation based on User Interests and Points of Interest Visit Durations},
	BOOKTITLE = {Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15)},
	PAGES = {1778-1784},
	YEAR = {2015}
 }
 @INPROCEEDINGS { lim-icaps16,
	AUTHOR = {Kwan Hui Lim and Jeffrey Chan and Christopher Leckie and Shanika Karunasekera},
	TITLE = {Towards Next Generation Touring: Personalized Group Tours},
	BOOKTITLE = {Proceedings of the 26th International Conference on Automated Planning and Scheduling (ICAPS'16)},
	PAGES = {412-420},
	YEAR = {2016}
 } 