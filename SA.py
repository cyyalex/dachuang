import numpy as np 
import random as rd 

'''背包容量c=8，物品数量n=5，对应价值如下 
求背包中物品的最大价值''' 
 
N=5 
C=8 
W=[2,3,5,1,4] 
V=[2,5,8,3,6] 
Xk=np.zeros(N)
Yk=np.zeros(N) #适应值为所有Yk的和 
Xnew=np.zeros(N)
Ynew=np.zeros(N)
L=100 
S=np.zeros(N) 
 
#初始化 

S=[1,0,0,1,1]
i=0
for i in range(N):
    if (S[i]==1):
        Xk[i]=W[i]
        Yk[i]=V[i]
T=(max(Xk)-min(Xk))*100 
tk=T 
k=0 
Xbest=Xk 
Ybest=Yk
print (Xbest) 
  
def drop(k): 
    return T/np.log(k+2) 
     
def near1(X1,X2): 
    i=rd.randint(0,N-1)
    Xg=np.zeros(N)
    Yg=np.zeros(N)
    while (sum(Xg)==0 or sum(Xg)>C):
        if (S[i]==1):
            S[i]=0
        else:
            S[i]=1
        for j in range (N):
            if (S[j]==1):
                Xg[j]=W[j]
                Yg[j]=V[j]
            else:
                Xg[j]=0
                Yg[j]=0
        i=i+1
        if (i>=N):
            i=i-N       
    global Xnew
    global Ynew
    Xnew=Xg
    Ynew=Yg
        
 #计算接受概率
def P (tk): 
     n=(sum(Ynew)-sum(Yk))/tk 
     return np.e**n              
             
while (1): 
    for i in range (L): 
        #从邻域函数生成新解 
        near1(Xnew,Ynew)
        #决定是否赋值 
        if (sum(Ynew)>sum(Yk)): 
            Xk=Xnew 
            Yk=Ynew 
            if (sum(Ynew)>sum(Ybest)): 
                Xbest=Xnew 
                Ybest=Ynew 
                continue 
        if (rd.random()<P(tk)): 
            Xk=Xnew 
        #降温               
        o=drop(k) 
        if (abs(tk-o)<=1): 
            #视为温度平衡 
            break 
        else: 
            tk=o  
            k+=1
    break        
         
print (Xbest) 
