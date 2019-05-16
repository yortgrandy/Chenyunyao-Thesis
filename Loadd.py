
# coding: utf-8

# In[ ]:


import pickle
import pprint
def loadd(n):

    pkl_file2 = open('psit_%s.pkl'%(n), 'rb')



    t1_data2 = pickle.load(pkl_file2) # t1_data2 is psit



    pkl_file2.close()

    return t1_data2


def loada(n):
    pkl_file1 = open('answer_%s.pkl'%(n), 'rb')
    pkl_file2 = open('accuracy_%s.pkl'%(n), 'rb')

    t1_data1 = pickle.load(pkl_file1) # t1_data1 is V
    t1_data2 = pickle.load(pkl_file2) # t1_data2 is psit

    pkl_file1.close()
    pkl_file2.close()

    return (t1_data1,t1_data2)