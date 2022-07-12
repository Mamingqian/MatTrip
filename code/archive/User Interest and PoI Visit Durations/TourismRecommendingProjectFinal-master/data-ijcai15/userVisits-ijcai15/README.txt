==================================================================
Flickr User-POI Visits
==================================================================

Dataset Information: 
This dataset comprises a set of users and their visits to various points-of-interest (POIs) in eight cities. The user-POI visits are determined based on geo-tagged YFCC100M Flickr photos that are: (i) mapped to specific POIs location and POI categories; and (ii) grouped into individual travel sequences (consecutive user-POI visits that differ by <8hrs). Other associated datasets are the "List of POIs" dataset ("POI-{cityName}.csv" files from "poiList-ijcai15.zip") and "POI Cost-Profit Table" dataset ("costProfCat-{cityName}POI-all.csv" files from "costProf-ijcai15.zip").

File Description and Dataset Statistics:
Filenames 		Description		#Users	#POI_Visits	#Travel_Sequences
userVisits-Toro.csv	POI visits in Toronto	1395	39419 		6057
userVisits-Osak.csv	POI visits in Osaka	450 	7747 		1115
userVisits-Glas.csv	POI visits in Glasgow	601 	11434 		2227
userVisits-Edin.csv	POI visits in Edinburgh	1454 	33944 		5028
userVisits-Pert.csv	POI visits in Perth	159 	3643 		716
userVisits-Buda.csv	POI visits in Budapest	935 	18513 		2361
userVisits-Delh.csv	POI visits in Delhi	279 	3993 		489
userVisits-Vien.csv	POI visits in Vienna	1155 	34515 		3193

All user-POI visits in each city are stored in a single csv file that contains the following columns/fields:
 - photoID: identifier of the photo based on Flickr.
 - userID: identifier of the user based on Flickr.
 - dateTaken: the date/time that the photo was taken (unix timestamp format).
 - poiID: identifier of the place-of-interest (Flickr photos are mapped to POIs based on their lat/long).
 - poiTheme: category of the POI (e.g., Park, Museum, Cultural, etc).
 - poiFreq: number of times this POI has been visited.
 - seqID: travel sequence no. (consecutive POI visits by the same user that differ by <8hrs are grouped as one travel sequence).

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