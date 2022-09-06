==================================================================
List of POIs
==================================================================

Dataset Information: 
This dataset comprises various points-of-interest (POI) that are found in each of the eight cities, based on their entries/articles on Wikipedia. Information such as the POI name, lat/lon coordinates and theme (category) are also obtained from Wikipedia. This dataset is used to derive the "POI Cost-Profit Table" dataset ("costProfCat-{cityName}POI-all.csv" files from "costProf-ijcai15.zip") and "Flickr User-POI Visits" dataset ("userVisits-{cityName}.csv" files from "userVisits-ijcai15.zip") 

File Description:
There are a total of eight files, each named "POI-{cityName}.csv", where each row indicate a specific POI and its associated ID, name, lat/long and theme.

The list of POI for each city is stored in a single csv file that contains the following columns/fields:
 - poiID: ID of the POI
 - poiName: name of the POI
 - lat: latitude coordinates
 - lon: longitude coordinates
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