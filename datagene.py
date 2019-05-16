
# coding: utf-8

# In[7]:


import numpy as np 
import random
import math
import pickle                                       #data IO

n = 200                                             #size of the system
w =0.25                                              #initial disorder strength
dw=0.0                                                #step of disorder strength
n2=20                                              #number of different strengths to be computed
t = 10e4    
num=1500

global V, psit, Vpsit
V=np.zeros((num*n2,n),dtype=np.float32)
psit=np.zeros((num*n2,n),dtype=np.float32)

def diagno(strg,k):
    for i in range(num):
        V[i+k*num]=[random.uniform(-strg,strg) for x in range(n)]
        H = np.diag(V[i+k*num],0) #(-strg,strg)take random diagonal elements from(-strg,strg) and generate the diagonal matrix
        for x in range(n-1):                         
            H[x][x+1]=-1
        for x in range(n-1):
            H[x+1][x] = -1                            #fill in the tridiagonal elements with t=-1
        epsilon, Utran = np.linalg.eig(H)            #compute the eigen values and the transpose of the transform matrix
        U = np.transpose(Utran)
        psi0 = np.array([0 for x in range(n)])
        psi0[0] = 1                                    #particle is at x=0 at t=0
        expepsilon = np.exp(t * epsilon*(0-1j))
        E = np.diag(expepsilon,0)                           #diagonalize time evolution matrix
        psit[i+k*num] = np.abs(np.dot(Utran,np.dot(E,np.dot(U,psi0))))
        
        
output1 = open('V_25.pkl', 'wb')
output2 = open('psit_25.pkl', 'wb')

for j in range(n2):
    diagno(w,j)
    w+=dw
    print(j)
'''save data'''


pickle.dump(V, output1)
pickle.dump(psit, output2)


output1.close()
output2.close()






# In[8]:




