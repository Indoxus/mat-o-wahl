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

def check_for_point_seperation(partys):
    check = hundred_percent_n_partys(partys)
    for i in check:
        if i != 255:
            return False
    return True


def calulate_percentages(q,m,party):
    maxi = 0
    sum = 0
    raster = [  [2,1,0],
                [1,2,1],
                [0,1,2]]
    for i in range(38):
        if q[i] != 255:
            sum += raster[q[i]][clean[party,i]] * (1+m[i])
            maxi += 2 * (1+m[i])
    return sum/maxi

def calulate_all_percentages(q,m):
    out = []
    for i in range(28):
        out.append(calulate_percentages(q,m,i))
    return out

def print_all_percentages(q,m):
    perc = calulate_all_percentages(q,m)
    perc.sort()
    for i in range(28):
        print(round(perc[i]*100,1))

def random_multipliers():
    m = np.random.randint(0,2,size = 38)
    return m.astype("bool")

def random_beliefs(skip = False):
    num = [0,1,2]
    if skip == True:
        num.append(255)
    return np.random.choice(num,size= 38)

def set_beliefs(good, bad):
    if good+bad > 38:
        return random_beliefs
    out = []
    for i in range(good):
        out.append(2)
    for i in range(bad):
        out.append(0)
    while len(out) < 38:
        out.append(1)
    return np.array(out,dtype="uint8")

def set_multipliers(i):
    out= np.zeros(38,dtype="bool")
    for ii in range(i):
        out[ii] = True
    return out

def random_indexes(n):
    if n == 0:
        return (0,0)
    i = np.random.randint(0,n+1,1)[0]
    j = n-i
    return (i,j)




#(q,m) =  hundred_percent_n_partys([1,4])
#fill_out.fill_out(q,m)
