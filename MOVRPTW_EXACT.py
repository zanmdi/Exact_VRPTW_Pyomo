#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Input.csv')


# In[3]:


# q1

int(df['Capacity1'][0])


# In[4]:


# q2

int(df['Capacity2'][0])


# In[5]:


# service time >> part of tij

int(df['Service time'][0])


# In[6]:


# V

int(df['Number of Available Vehicles'][0])


# In[8]:


# number of nodes including deopt(= x_0jk)

len(df)


# In[9]:


# ai

df ['startTime'] = pd. to_datetime (df ['startTime'])
df ['startTime'] = df['startTime'].dt.hour * 3600 + df['startTime'].dt.minute * 60


# In[10]:


# bi

df ['endTime'] = pd. to_datetime (df ['endTime'])
df ['endTime'] = df['endTime'].dt.hour * 3600 + df['endTime'].dt.minute * 60


# In[11]:


# r1i

# df['demands1']

# r2i

# df['demands2']


# In[12]:


def haversine(coord1, coord2):
    R =6371.0088  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 +         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


c_list = df['id'].tolist() # list of customers inorder to add them as the key of dictionary

#make a dictionary of customers' location (lat & lon)
customerloc ={}

for j in c_list:
    customerloc[j] = (df.iloc[c_list.index(j), 5] , df.iloc[c_list.index(j), 6])
    
    

    
#Create distance Matrix

matrix = []

for i in range(len(list(customerloc.values()))):
    # Append an empty sublist inside the list
    matrix.append([])
    
    m = list(customerloc.values())[i][0], list(customerloc.values())[i][1]
    
    for c_id, coord in customerloc.items():
        distance = haversine(m , coord)
        matrix[i].append(distance) 
        
        
mat = pd.DataFrame(matrix)
mat = mat.transpose()


# In[13]:


# C2ij

c = np.matrix(mat)


# c1ij = tij

t = c/60  # the spped is considered 60 km/h


# In[14]:


# Visualization of the depot and customers

df['NODE_TYPE'] = ['Customers' if  demand >= 0 else 'delivery' for demand in df['demands1'] if demand != 0] + ['Depot']

plt.figure(figsize=(12,8))

sns.scatterplot(x=df['lat'],y=df['long'], hue=df['NODE_TYPE'], size=df['NODE_TYPE'], sizes=[150, 400], alpha= 0.8)


plt.xlabel('X_coord', fontsize=20)

plt.ylabel('Y_coord', fontsize=20)

plt.legend(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.show()


# In[15]:


model=ConcreteModel()

# =============================================================================
#  index
# =============================================================================
# index of customer
model.n = RangeSet(0,8)
n = model.n
N=8

#index of vehicles
model.v = RangeSet(1,10)
v = model.v


#index of objective
model.ww = RangeSet(1,5)


# In[19]:


# =============================================================================
# parameter
# =============================================================================
#demand 1 customer i
model.r1=Param(n,mutable=True)
#model.r2=Param(n,mutable=True)
for i in model.n:
    model.r1[i] = df['demands1'][i]
    
model.r1[0] = 0
model.r1[N] = 0

#demand 2 customer i
# model.r2=Param(n,mutable=True)

# for i in model.n:
#     model.r2[i] = df['demands2'][i]
    
#model.r2[0] = 0
#model.r2[N] = 0


#capacity 1 of vehicles
model.q1 = Param(initialize=1200)



#capacity 2 of vehicles
model.q2 = Param(initialize=800)


#time window customer i 
model.a = Param(n,mutable=True)
model.b = Param(n,mutable=True)

for i in model.n:
    model.a[i] = df['startTime'][i]
    
for i in model.n:
    model.b[i] = df['endTime'][i]




#travel time from customer i to j includes service time
model.t=Param(n,n,mutable=True)

for i in model.n:
    for j in model.n:
        model.t[i,j] = t[i, j] + int(df['Service time'][0])

#Travel time from node i to node j
model.c1t=Param(n,n,mutable=True)

for i in model.n:
    for j in model.n:
        model.c1t[i,j] = t[i, j]


#Travel distance from node i to node j 
model.c2d=Param(n,n,mutable=True)

for i in model.n:
    for j in model.n:
        model.c2d[i,j] = c[i, j]

#is the linear distance from client i to client j 
model.d=Param(n,n,mutable=True)



for i in model.n:
    for j in model.n:
        model.d[i,j] = c[i, j]


#weigth of objective
model.w=Param(model.ww,mutable=True)

model.w[1] = 1
model.w[2] = 1
model.w[3] = 1
model.w[4] = 1
model.w[5] = 1

#Big M
model.M=Param(initialize=200000)
# =============================================================================
# Data
# =============================================================================




# =============================================================================
# Variables
# =============================================================================
model.x=Var(n,n,v,within=Binary) #1 if vehicle k goes from node i to node j, 0 otherwise
model.y=Var(v,within=Binary)    #1 if vehicle k is used, 0 otherwise
model.s=Var(n,v,within=NonNegativeReals)    #is defined for each vertex i and each vehicle k
model.z1max=Var(within=NonNegativeReals)
model.z2max=Var(within=NonNegativeReals)
model.z2min=Var(within=NonNegativeReals)
# =============================================================================
# objective
# =============================================================================

model.obj=Objective(expr=model.w[1]*sum(model.c1t[i,j]*model.x[i,j,k] for i in n for j in n for k in v)+model.w[2]*sum(model.c2d[i,j]*model.x[i,j,k] for i in n for j in n for k in v)+model.w[3]*sum(model.y[k] for k in v)+model.w[4]*model.z1max+model.w[5]*(model.z2max-model.z2min))

# =============================================================================
# constraint
# =============================================================================

#constraint 1

model.c1=ConstraintList()
for i in n :
    if i >0:
        model.c1.add(expr=sum(model.x[i,j,k]   for j in n for k in v if i!=j)==1)
    
#constraint 1'

model.c1p=ConstraintList()
for j in n :
    if j>0:  
        model.c1p.add(expr=sum(model.x[i,j,k]   for i in n for k in v if i!=j )==1)    
    
    
    
#constraint 2

model.c2=ConstraintList()
for k in v:
    model.c2.add(expr=sum(model.r1[i]*sum(model.x[i,j,k] for j in n) for i in n)<=model.q1*model.y[k])
    
    
    
    
#constraint 3

model.c3=ConstraintList()
for k in v:
    model.c3.add(expr=sum(model.x[i,j,k] for i in n for j in n)<=model.q2*model.y[k])
    
    
    
    
#constraint 4

model.c4=ConstraintList()
for k in v:
    model.c4.add(expr=sum(model.x[0,j,k] for j in n if j>0)==model.y[k])
    
    
    
#constraint 5

model.c5=ConstraintList()
for h in n:
    for k in v:
        model.c5.add(expr=sum(model.x[i,h,k] for i in n if i!=h)-sum(model.x[h,j,k] for j in n if j!=h)==0)

#constraint 6

model.c6=ConstraintList()
for i in n:
    model.c6.add(expr=sum(model.x[i,0,k] for i in n if i>0)==model.y[k])        
        
        
#constraint 7 and 8 and 9

model.c7=ConstraintList()
model.c8=ConstraintList()
model.c9=ConstraintList()
for k in v:
    model.c7.add(expr=model.z2max>=sum(model.x[i,j,k] for i in n for j in n))
    model.c8.add(expr=model.z2min<=sum(model.x[i,j,k] for i in n for j in n))
    model.c9.add(expr=model.z1max>=sum(model.d[i,j]*model.x[i,j,k] for i in n for j in n))
    
model.u=Var(n,within=NonNegativeReals)   

#constraint 10

model.c10=ConstraintList()
model.c14=ConstraintList()
for i in n:
    for j in n:
        for k in v:
          if i!=j and j>0:            
            model.c10.add(expr=model.s[i,k]+model.t[i,j]-model.M*(1-model.x[i,j,k])<=model.s[j,k])
          if i!=j and j>0:  
            model.c14.add(expr=model.u[i]-model.u[j]+1<=(N-1)*(1-model.x[i,j,k]))
#constraint 11 and 12 

model.c11=ConstraintList()
model.c12=ConstraintList()
for i in n:
    for k in v:
        model.c11.add(expr=model.a[i]*sum(model.x[i,j,k] for j in n)<=model.s[i,k])
        model.c12.add(expr=model.b[i]*sum(model.x[i,j,k] for j in n)>=model.s[i,k])    
model.c13=ConstraintList()
for i in n:
    for k in n:
        for k in v:
            model.c13.add(expr=model.x[i,j,k]<=model.y[k])


# In[ ]:


# =============================================================================
opt=SolverFactory("cplex")   
opt.solve(model) 
# =============================================================================
result=opt.solve(model)
result.write()
print(f'objective value={value(model.obj)}')
for i in n:
    for j in n:
        for k in v:
          if   value(model.x[i,j,k])==1:
            print(f"x{i,j,k}={value(model.x[i,j,k])}")
for k in v:
    if   value(model.y[k])==1:
        print(f"y({(k)})={value(model.y[k])}")