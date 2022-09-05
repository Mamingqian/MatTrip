import numpy as np
import pandas as pd
import csv

path = "/Users/mamingqian/Desktop/DCE/转期刊/dataset/Flickr/userVisits-ijcai15/"
fake_path = (open("/Users/mamingqian/Desktop/DCE/转期刊/dataset/fake_path-Toro.csv",'w',newline='',encoding='utf8'))
fake_user = (open("/Users/mamingqian/Desktop/DCE/转期刊/dataset/fake_user-Toro.csv",'w',newline='',encoding='utf8'))

input_name = "userVisits-Toro.csv"
output_name = "fake_user-Buda.csv"


#fake user generation
csv_writer = csv.writer(fake_user)
csv_writer.writerow(["Structure","Cultural","Amusement","Beach","Sport","Shopping"])

    
with open((path + input_name)) as routes:
    total_len = 0
    Sport_total = 0
    Beach_total = 0
    Amusement_total = 0
    Structure_total = 0
    shopping_total = 0
    cultural_total = 0
    csvReader1 = csv.reader(routes, delimiter=';')
    for row in csvReader1:
        photoid,uid,date,pid,crt_cat,pfreq,seqid = row
        if seqid == "seqID":
            continue
        total_len += 1
        if crt_cat == "Structure":
            Structure_total += 1
        elif crt_cat == "Cultural":
            cultural_total += 1
        elif crt_cat == "Amusement":
            Amusement_total += 1
        elif crt_cat == "Beach":
            Beach_total += 1
        elif crt_cat == "Sport":
            Sport_total += 1
        else:
            shopping_total += 1

Structure_avg = Structure_total/total_len
cultural_avg = cultural_total/total_len
Amusement_avg = Amusement_total/total_len
Beach_avg = Beach_total/total_len
Sport_avg = Sport_total/total_len
shopping_avg = shopping_total/total_len
            
with open((path + input_name)) as routes:
    csvReader1 = csv.reader(routes, delimiter=';')
    total_len = 0
    Sport_total = 0
    Beach_total = 0
    Amusement_total = 0
    Structure_total = 0
    shopping_total = 0
    cultural_total = 0
    last_time = 0
    last_seqid = 0
    cat = ""
    pref = []
    for row in csvReader1:
        photoid,uid,date,pid,crt_cat,pfreq,seqid = row
        if seqid != "seqID":
            if seqid != last_seqid:
                if last_seqid == 0:
                    last_seqid = seqid
                    total_len += 1
                    if crt_cat == "Structure":
                        Structure_total += 1
                    elif crt_cat == "Cultural":
                        cultural_total += 1
                    elif crt_cat == "Amusement":
                        Amusement_total += 1
                    elif crt_cat == "Beach":
                        Beach_total += 1
                    elif crt_cat == "Sport":
                        Sport_total += 1
                    else:
                        shopping_total += 1
                    continue
                Structure_res = Structure_total/total_len if Structure_total != 0 else Structure_avg
                cultural_res = cultural_total/total_len if cultural_total != 0 else cultural_avg
                Beach_res = Beach_total/total_len if Beach_total != 0 else Beach_avg
                Sport_res = Sport_total/total_len if Sport_total != 0 else Sport_avg
                shopping_res = shopping_total/total_len if shopping_total != 0 else shopping_avg
                Amusement_res = Amusement_total/total_len if Amusement_total != 0 else Amusement_avg
                pref = [Structure_res,cultural_res,Amusement_res,Beach_res,Sport_res,shopping_res]
                csv_writer.writerow(pref)
                total_len = 0
                Sport_total = 0
                Beach_total = 0
                Amusement_total = 0
                Structure_total = 0
                shopping_total = 0
                cultural_total = 0
                pref =  []
                total_len += 1
                if crt_cat == "Structure":
                    Structure_total += 1
                elif crt_cat == "Cultural":
                    cultural_total += 1
                elif crt_cat == "Amusement":
                    Amusement_total += 1
                elif crt_cat == "Beach":
                    Beach_total += 1
                elif crt_cat == "Sport":
                    Sport_total += 1
                else:
                    shopping_total += 1
            else:
                total_len += 1
                if crt_cat == "Structure":
                    Structure_total += 1
                elif crt_cat == "Cultural":
                    cultural_total += 1
                elif crt_cat == "Amusement":
                    Amusement_total += 1
                elif crt_cat == "Beach":
                    Beach_total += 1
                elif crt_cat == "Sport":
                    Sport_total += 1
                else:
                    shopping_total += 1
            last_seqid = seqid
    Structure_res = Structure_total/total_len if Structure_total != 0 else Structure_avg
    cultural_res = cultural_total/total_len if cultural_total != 0 else cultural_avg
    Beach_res = Beach_total/total_len if Beach_total != 0 else Beach_avg
    Sport_res = Sport_total/total_len if Sport_total != 0 else Sport_avg
    shopping_res = shopping_total/total_len if shopping_total != 0 else shopping_avg
    Amusement_res = Amusement_total/total_len if Amusement_total != 0 else Amusement_avg
    pref = [Structure_res,cultural_res,Amusement_res,Beach_res,Sport_res,shopping_res]
    csv_writer.writerow(pref)


# fake_path generation
csv_writer = csv.writer(fake_path)
with open((path + input_name)) as routes:
    csvReader = csv.reader(routes, delimiter=';')
    previous_path = []
    previous_seqid = 1
    for row in csvReader:
        photoid,uid,date,pid,cat,pfreq,seqid = row
        if seqid != previous_seqid:
            csv_writer.writerow(previous_path)
            previous_path = [pid]
        else:
            previous_path.append(pid)
        previous_seqid = seqid
    csv_writer.writerow(previous_path)
fake_path.close()


                


        
