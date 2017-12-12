# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:01:54 2017

@author: Vincent.EC.Lo
"""

#%%
import pandas as pd

#%%

### Create by single list

list1 = [1,2,3,4,5]

### Assign column name by columns
pd.DataFrame(list1,columns = ["col1"])

#%%

### Create by two list (or more)

list2 = [5,6,7,8,9]

pd.DataFrame(list(zip(list1,list2)),columns = ["col1","col2"])


#%%
##### Create by dict

pd.DataFrame({"col1":1,"col2":[1,2,3,4,5]})

df_dict = dict({"col1":list1,"col2":list2})

pd.DataFrame(df_dict)


#%%
#### create pandas with different length

list3 =[1,2,3]
df1 = pd.DataFrame(list3,columns = ["col3"])
df = pd.DataFrame(list1,columns = ["col1"])

### axis = 1 >> cbind
df_all = pd.concat([df1,df],axis = 1)

df_all

#%%