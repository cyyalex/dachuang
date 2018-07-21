import numpy as np
import random as rd
import math 

pop_size = 300  # 种群数量
G = 200 # 迭代次数
D = 2   # 问题的维数
F = 0.6  # 变异算子
CR = 0.1  # 交换概率
fitness = []  # 适应度
fitnessg = []
fitnessu = []
fitness_mean =[]       # 平均适应度
fitnessu_mean = [0]
optimum_solution = []  # 每次迭代的所获得的最优解
POP = []
POPV = []
POPVg=[]
U=[]

for i in range (pop_size):
    fitnessg.append(0)
    POPVg.append(0)
    fitnessu.append(0)
    U.append([0,0])
rng=np.random


POPV.append(rng.rand(pop_size,D))
POP.append(rng.rand(pop_size,D))

#定义适应度函数
def fitnessf(x1,x2):
    value = 3*math.cos(x1*x2)+x1+x2
    return value
    
#定义适应度计算函数
def calculate_fitness(g):
    for j in range(pop_size):
        y=POP[g]
        x1=y[j][0]
        x2=y[j][1]
        fitnessg[j]=fitnessf(x1,x2)
    fitness.append(fitnessg)
    fitness_mean.append(sum(fitnessg)/pop_size)

def calculate_fitnessu():
    for j in range (pop_size):
        x1=U[j][0]
        x2=U[j][1]
        fitnessu[j]=fitnessf(x1,x2)
        fitnessu_mean[0]=sum(fitnessu)/pop_size
        
def mutation(g):
    for i in range(pop_size):
        r=[0,0,0]
        r[0]=rd.randint(0,pop_size-1)
        r[1]=rd.randint(0,pop_size-1)
        r[2]=rd.randint(0,pop_size-1)
        POPVg[i]=X[r[0]]+F*(X[r[1]]-X[r[2]])
        POPV.append(POPVg)

def rn(i):
    return rd.randint(0,D-1)

def crossover():
    rc=0
    for i in range (pop_size):
        rc=rd.random()
        if (rc<=CR or rn(i)==0):
            U[i][0]=POPVg[i][0]
            U[i][1]=X[i][1]
        else:
            U[i][0]=X[i][1]
            U[i][1]=POPVg[i][0]
            
def choose(g):
    for i in range (pop_size):
        if fitnessu[i]>fitness[g][i]:
            X[i]=U[i]
        else:
            X[i]=X[i]
    POP.append(X)       
            
       
g=0
while (g<=G):
    calculate_fitness(g)#计算适应值 
    X=POP[g]
    j=0
    maxc=0
    for j in range (pop_size):
        if (fitness[g][j]>=max(fitness[g])):
            maxc=j
        optimum_solution.append(POP[g][maxc])
    if (fitness[g][maxc]-fitness_mean[g]<=0.1):  #终止条件
        break
    else:
        mutation(g)
        crossover()
        calculate_fitnessu()
        choose(g)
        g=g+1
print ("x=%p\n",optimum_solution[g])        
