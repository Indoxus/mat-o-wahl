import pandas
import numpy as np
import fill_out
filename = "./data/data_bundestags_wahl2025.csv"
csv = pandas.read_csv(filename, encoding = 'cp1252')
clean = np.zeros(shape=(28,38),dtype="uint8")

def maptonumbers(s:str):
    if s == "stimme zu":
        return 2
    elif s == "neutral":
        return 1
    else:
        return 0
        
for r in csv.iterrows():
    clean[int(r[1]["Partei: Nr."])-1,int(r[1]["These: Nr."])-1] = maptonumbers(r[1]["Position: Position"])
print(clean)

def hundred_percent_n_partys(partys):
    q = clean[partys[0]]
    m = np.zeros(38,dtype="bool")
    for i in range(len(partys)-1):
        for ii in range(38):
            if q[ii] != clean[partys[i+1],ii]:
                q[ii] = 255
                print(ii)
    return (q,m)

(q,m) =  hundred_percent_n_partys([1,4])#
print(q)
fill_out.fill_out(q,m)