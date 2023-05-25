#!/usr/bin/env python
# coding: utf-8

# Pymatgen (Python Materials Genomics) is a robust, open-source Python library for materials analysis. These are some of the main features.

# In[22]:


#https://pymatgen.org


# In[121]:


import numpy as np
from pymatgen.core import Structure
import pymatgen
from pymatgen.ext.matproj import MPRester
import pandas as pd
from pymatgen.core.composition import Composition
import matminer
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error as MSE
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import matminer
#from pymatgen.io.cif import CifParser
#from pymatgen.core.structure import Cifwriter
from pymatgen.core.structure import Structure
from matminer.datasets import load_dataset


# In[122]:


list_of_available_fields = mpr.summary.available_fields


# In[123]:


list_of_available_fields


# In[127]:


api='D32oOhE5YEc2fubxEdCv1khjg6EBIfCL'




# In[132]:


with MPRester(api) as mpr:
    docs = mpr.summary.search(elements=["Si", "O"],num_elements='3',is_gap_direct='True', band_gap=(0.5, 1.0))


# index error: as by default if u didn't specify the fields u require all available fields are taken 

# In[35]:


docs1 = mpr.summary.search(elements=["Si", "O"], band_gap=(0.5, 1.0),fields=["material_id", "band_gap", "volume"])


# In[113]:


docs2 = mpr.summary.search(elements=["Si", "O"], band_gap=(0.5, 1.0),fields=["material_id", "band_gap", "volume","composition"])


# specified elements will be present in the composition with other elements

# In[114]:


docs2


# In[48]:


docs3 = mpr.summary.search(formula='TiO2',is_gap_direct='True', band_gap=(0.5, 1.0),fields=["composition"])
#only 1 in materialproject database with specified value 


# In[49]:


docs4 = mpr.summary.search(formula='TiO2',is_gap_direct='True', band_gap=(0.5, 4),fields=["composition"])
#got 9 


# In[16]:


#with MPRester(api) as mpr:
#    search_results1=mpr.summary.search(chemsys=['Li','Si'],fields=["composition"])


# In[116]:


#with MPRester(api) as mpr:
search_results2=mpr.summary.search(chemsys=['Li-Si-O'],fields=["composition"])


# In[117]:


#search_results2


# In[21]:


#search_results1


# In[57]:


search_results3=mpr.summary.search(formula='ABO3',is_gap_direct=True,is_stable=False,fields=["composition","formation_energy_per_atom"])


# In[58]:


search_results3


# In[81]:


search_results4=mpr.summary.search(formula='ABO3',is_gap_direct="True",is_stable="True",fields=["structure","composition","formation_energy_per_atom","formula_pretty","band_gap"])


# In[82]:


#search_results=pd.DataFrame(search_results)


# In[ ]:





# In[83]:


search_results4


# In[93]:


data_list=[]
for result in search_results4:
    data_dict={}
    data_dict['structure']=result.structure
    data_dict['composition']=result.composition.reduced_formula
    data_dict['bandgap']=result.band_gap
    data_list.append(data_dict)


# In[94]:


table=pd.DataFrame(data_list)


# In[95]:


table


# In[88]:


#table.tail()


# In[96]:


pd.set_option('display.max_columns',None)


# In[110]:


pd.set_option('display.max_rows',None)


# In[111]:


table


# In[ ]:


#table.to_csv("table.csv")


# In[ ]:


#df = pd.read_csv(table.csv')


# loading exsiting datasets

# https://hackingmaterials.lbl.gov/matminer/dataset_summary.html

# In[ ]:


bnd_data=load_dataset("matbench_expt_gap")#expt bnd dataset


# In[ ]:


#bnd_data


# Clean the data before, using the data for ml,you can do it using excel,google sheets, or in python using pandas itself

# What to check in the data . 
# check for strings(words) ,ml only takes numeric data, you have to either convert the string in data to numeric value or simply avoid them.

# In[ ]:


bnd_data = bnd_data.dropna()# to drop empty cells in data(NAN values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




