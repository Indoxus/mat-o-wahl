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

def get_clean():
    return np.array(clean,copy=True)

def hundred_percent_n_partys(partys):
    q = np.array(clean[partys[0]],copy=True)
    m = np.zeros(totalthesis,dtype="bool")
    for i in range(len(partys)-1):
        for ii in range(totalthesis):
            if q[ii] != clean[partys[i+1],ii]:
                q[ii] = 255
                print("diffrent belief @",ii)
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

def calulate_all_percentages(q,m,filter = []):
    out = []
    for i in range(totalparty):
        if i in filter:
            out.append(0)
        else:
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
    temp = [0,0,0,0,0,0,0]
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

def cond(step,dist,N):
    r = np.random.rand()
    sum = 0.5
    sum += 1-(N-step)**2/(N**2)/2
    #return False
    if r > sum:
        return True
    return False


def use_best_value2(q,i,m,goal,step,N):
    temp = [[0,0,0,0],[0,0,0,0]]
    minj = 0
    sminj = 0
    minjj = 0
    sminjj = 0
    for j in range(2):
        for jj in range(4):
            if jj != 3:
                q[i] = jj
            else:
                q[jj] = 255
            temp[j][jj] = norm(calulate_all_percentages(q,m),goal)
            if temp[j][jj] < temp[minj][minjj]:
                sminj = minj
                minj =j
                sminjj =minjj
                minjj = jj
        m[i]= not m[i] 
    #if cond(step,temp[minj][minjj],N):
    #    m[i]= not m[i]
    if minjj != 3:
        q[i] = minjj
    else:
        q[i] = 255
    if minj == 1:
        m[i]= not m[i]
    print(i,"dist = ", temp[minj][minjj])


def solve(q,m,goal,step,N):
    temp = []
    for i in range(totalthesis):
        use_best_value2(q,i,m,goal,step,N)
    return (q,m)

def switch_n_bools(m,n):
    for i in range(n):
        r = np.random.randint(0,len(m))
        m[r] = not m[r]

def better_norm(perc, goal,p = 2):
    for i in range(totalparty):
        if goal[i] == 0:
            perc[i] =0
        else: 
            perc[i] = round(perc[i],4)
    return np.linalg.norm(np.subtract(perc,goal),ord =p) 

def better_solve(q,m,goal,maxIt=1000,tol = 0):
    change_amount = 5
    for i in range(maxIt):
        perc = calulate_all_percentages(q,m)
        dist = np.absolute(np.subtract(perc,goal))
        maxi = np.argmax(dist)
        mini = np.argmin(dist)
        r = np.random.randint(0,totalthesis)
        temp = 0
        for ii in range(totalthesis):
            ii = (r+i)%38
            if clean[maxi][ii] != clean[mini][ii]:
                [pmax,pmin] = [calulate_percentages(q,m,maxi),calulate_percentages(q,m,mini)]
                temp = q[ii]
                q[ii] = clean[mini][ii]
                [primemax,primemin] = [calulate_percentages(q,m,maxi),calulate_percentages(q,m,mini)]
                qperc = calulate_all_percentages(q,m)
                if abs(primemax-goal[maxi]) < abs(pmax-goal[maxi]) and abs(primemin-goal[mini]) < abs(pmin-goal[mini]):
                    temp2 = m[ii]
                    m[ii] = not m[ii]
                    mprime = calulate_all_percentages(q,m)
                    if better_norm(qperc,goal) < better_norm(mprime,goal):
                        m[ii] = temp2
                    break
                else:
                    q[ii] = temp
                
        print(better_norm(calulate_all_percentages(q,m),goal))
    return (q,m)

        #en = int(((maxIt-i)**2/(maxIt**2))/change_amount*totalthesis)
        #for ii in range(en):
        #    r = np.random.randint(0,totalthesis)
        #    for iii in range(10):
        #        if clean[maxi][r] != clean[mini][r]:# and q[r] != clean[mini][r]:
        #            q[r] = clean[mini][r]
        #            break
        #        r = (r+1)%totalthesis
        #print(norm(calulate_all_percentages(q,m),goal))
    #return (q,m)

def calculate_percentages_prime(thesis,q,m,qprime,mprime,leave_out = []):
    temp1 = q[thesis]
    temp2 = m[thesis]
    q[thesis] = qprime
    m[thesis] = mprime
    out = calulate_all_percentages(q,m,leave_out)
    q[thesis] = temp1
    m[thesis] = temp2
    return out

def random_permute(q,m,n):
    for i in range(n):
        ind = np.random.randint(0,totalthesis)
        print("perumtation @ ",ind)
        q[ind] = np.random.choice([0,1,2,0,1,2,0,1,2,255])
        m[ind] = np.random.choice([True, False,False,False])
    return (q,m)

def morebruteforcesolver(q,m,goal):
    zwischen = np.zeros((totalthesis,2,4),dtype = "float32")
    mprime = False
    best = 1000
    prevstar = 1000
    qstar = np.zeros(totalthesis,"uint8")
    mstar = np.zeros(totalthesis,"bool")
    same = 0
    while True:
        for thesis in range(totalthesis):
            for mult in range(2):
                for entry in range(4):
                    if entry == 3:
                        qprime = 255
                    else:
                        qprime = entry
                    zwischen[thesis][mult][entry] = norm(calculate_percentages_prime(thesis,q,m,qprime,mprime),goal)
                mprime = not mprime
        ind = np.argmin(zwischen)
        ind = np.unravel_index(ind,zwischen.shape)
        print("change @ ",ind[0])
        prevbest = best
        best = zwischen[ind]
        print("dist = ",best)
        if best == 0:
            return (q,m)
        if best == prevbest:
            print("permute")
            if prevstar < best:
                print("prev was better")

                (q,m) = random_permute(np.copy(qstar),np.copy(mstar),5+2*same)
                same +=1
                print("samestrak = ",same)
            else:
                print("new best")
                qstar = np.copy(q)
                mstar = np.copy(m)
                same = 0
                (q,m) = random_permute(q,m,5+2*same)
                prevstar = best
        t = ind[2]
        if ind[2] == 3:
            t = 255
        q[ind[0]] = t
        m[ind[1]] = ind[1]

    


## Linalg solver
#def evaluation_operator(q,m):
#    return calulate_all_percentages(q,m)
#
#def getbasis():
#    basis = np.stack(np.zeros(totalthesis,dtype="uint8"),np.zeros(totalthesis,dtype="bool"))
#    for mult in range(2):
#        for belief in range(totalthesis):
#            basis
#
#
#def linalg_solver(q,m,goal):
#    for mult in range(2):




(qg,mg) =  hundred_percent_n_partys([1,4])

#fill_out.fill_out(qg,mg)
qg = random_beliefs()
mg = random_multipliers()
goal = calulate_all_percentages(qg,mg)
for g in goal:
    g = round(g,4)

q = np.array(clean[1],copy=True)
m = np.zeros(totalthesis,"bool") 

N = 30
qprime = np.array(q,copy=True)
mprime = np.array(m,copy=True)

(qprime,mprime) = morebruteforcesolver(q,m,goal)

#for i in range(N):
#    (qprime,mprime) = solve(np.array(qprime,copy=True),np.array(mprime,copy=True),goal,i,N)
#    print(i,"--------------")
#    for i in range(totalthesis):
#        print(qprime[i],mprime[i],qg[i],mg[i])
#
#    perc = calulate_all_percentages(qprime,mprime)
#    for i in range(totalparty):
#        print(round(goal[i]*100,1),round(perc[i]*100,1))
#    
#    switch_n_bools(m,int((totalthesis-totalthesis*(N-i)**2/(N**2))/3))
#    
#(qprime,mprime) = better_solve(np.array(qprime,copy=True),np.array(mprime,copy=True),goal)









#(q,m) =  hundred_percent_n_partys([1,4])
#fill_out.fill_out(q,m)
