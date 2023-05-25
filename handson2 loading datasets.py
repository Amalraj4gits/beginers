#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pymatgen
from pymatgen.ext.matproj import MPRester
import pandas as pd
from pymatgen.core import Structure, Composition  #import Structure and Composition classes from pymatgen.core
from pymatgen.io.cif import CifParser
import pandas as pd
import matminer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LinearRegression
import matminer
from matminer.datasets import load_dataset


# https://hackingmaterials.lbl.gov/matminer/dataset_summary.html

# In[8]:


bnd_data=load_dataset("matbench_expt_gap")#expt bnd dataset


# In[9]:


bnd_data


# In[ ]:


flla_data = flla_data.dropna()


# In[10]:


bnd_data=bnd_data.dropna()


# In[6]:


flla_data=flla_data.drop(['material_id'],axis=1)


# In[7]:


flla_data


# In[ ]:




