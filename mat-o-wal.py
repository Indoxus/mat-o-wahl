import pandas
import numpy as np
import fill_out
totalthesis = 38
totalparty = 28

filename = "./data/data_bundestags_wahl2025.csv"
csv = pandas.read_csv(filename, encoding = 'cp1252')
clean = np.zeros(shape=(totalparty,totalthesis),dtype="uint8")

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
    q = np.array(clean[partys[0]],copy=True)
    m = np.zeros(totalthesis,dtype="bool")
    for i in range(len(partys)-1):
        for ii in range(totalthesis):
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
    for i in range(totalthesis):
        if q[i] != 255:
            sum += raster[q[i]][clean[party,i]] * (1+m[i])
            maxi += 2 * (1+m[i])
    return sum/maxi

def calulate_all_percentages(q,m):
    out = []
    for i in range(totalparty):
        out.append(calulate_percentages(q,m,i))
    return out

def print_all_percentages(q,m):
    perc = calulate_all_percentages(q,m)
    perc.sort()
    for i in range(totalparty):
        print(round(perc[i]*100,1))

def random_multipliers():
    m = np.random.randint(0,2,size = totalthesis)
    return m.astype("bool")

def random_beliefs(skip = False):
    num = [0,1,2]
    if skip == True:
        num.append(255)
    return np.random.choice(num,size= totalthesis)

def set_beliefs(good, bad):
    if good+bad > totalthesis:
        return random_beliefs
    out = []
    for i in range(good):
        out.append(2)
    for i in range(bad):
        out.append(0)
    while len(out) < totalthesis:
        out.append(1)
    return np.array(out,dtype="uint8")

def set_multipliers(i):
    out= np.zeros(totalthesis,dtype="bool")
    for ii in range(i):
        out[ii] = True
    return out

def random_indexes(n):
    if n == 0:
        return (0,0)
    i = np.random.randint(0,n+1,1)[0]
    j = n-i
    return (i,j)

(q,m) =  hundred_percent_n_partys([1,4])
m = set_multipliers(7)
goal = calulate_all_percentages(q,m)

for g in goal:
    g = round(g,4)

q = clean[15]
m = np.zeros(totalthesis,"bool")

def norm(perc, goal,p = 2):
    for i in range(totalparty):
        if goal[i] == 0:
            perc[i] =0
        else: 
            perc[i] = round(perc[i],4)
    return np.linalg.norm(np.subtract(perc,goal),ord =p)

def use_best_value(q,i,m,goal):
    temp = [0,0,0,0]
    mini = 0
    for ii in range(4): # change to 4 to allow skipping
        if ii != 3:
            q[i] = ii
        else:
            q[i] = 255
        temp[ii] = norm(calulate_all_percentages(q,m),goal)
        if temp[ii] < temp[mini]:
            mini = ii
    if mini != 3:
        q[i] = mini
    else:
        q[i] = 255
    print(i,"dist = ", temp[mini])
    


def solve(q,m,goal):
    temp = []
    for i in range(totalthesis):
        use_best_value(q,i,m,goal)
    return (q,m)

(qg,m) =  hundred_percent_n_partys([1,4])
m = np.zeros(totalthesis,"bool")#set_multipliers(7)
goal = calulate_all_percentages(qg,m)
for g in goal:
    g = round(g,4)

q = np.array(clean[15],copy=True)
m = np.zeros(totalthesis,"bool") 

(qprime,mprime) = solve(np.array(q,copy=True),m,goal)


for i in range(totalthesis):
    print(qprime[i],qg[i])

perc = calulate_all_percentages(qprime,mprime)
for i in range(totalparty):
    print(round(goal[i]*100,1),round(perc[i]*100,1))


(qprimeprime,mprime) = solve(np.array(qprime,copy=True),m,goal)


for i in range(totalthesis):
    print(qprimeprime[i],qg[i])

perc = calulate_all_percentages(qprimeprime,mprime)
for i in range(totalparty):
    print(round(goal[i]*100,1),round(perc[i]*100,1))





#(q,m) =  hundred_percent_n_partys([1,4])
#fill_out.fill_out(q,m)
