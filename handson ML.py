#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[11]:


api='D32oOhE5YEc2fubxEdCv1khjg6EBIfCL'
with MPRester(api) as mpr:
    search_results4=mpr.summary.search(formula='ABO3',is_gap_direct="True",is_stable="True",fields=["structure","composition","formation_energy_per_atom","formula_pretty","band_gap"])


# In[12]:


search_results4


# In[14]:


data_list=[]
for result in search_results4:
    data_dict={}
    data_dict['structure']=result.structure
    data_dict['composition']=result.composition.reduced_formula
    data_dict['bandgap']=result.band_gap
    data_list.append(data_dict)


# In[16]:


data_list


# In[17]:


data_table=pd.DataFrame(data_list)


# In[18]:


data_table


# Note :we are going to apply matminer featurizers to generate featurizers based on structure and composition,but the composition is not in pymatgen composition format, convert it into pymatgen composition format.

# In[19]:


data_table["composition"]=data_table["composition"].apply(lambda x:Composition(x))


# In[20]:


data_table


# In[21]:


#data_table=data_table.drop(["composition"],axis=1)


# In[22]:


data_table


# https://hackingmaterials.lbl.gov/matminer/

# In[24]:


#from matminer.featurizers.conversions import StructureToComposition
#comp=StructureToComposition()
#data_table=comp.featurize_dataframe(data_table,col_id='structure',ignore_errors=True)


# In[25]:


#data_table


# In[29]:


from matminer.featurizers.composition import ElementFraction
ef = ElementFraction()
feat1=ef.featurize_dataframe(data_table,col_id='composition')

from matminer.featurizers.structure.bonding import StructuralHeterogeneity
cou=StructuralHeterogeneity()
cou.fit(data_table['structure'])
feat2=cou.featurize_dataframe(data_table,col_id='structure',ignore_errors=True)

from matminer.featurizers.conversions import CompositionToOxidComposition
oxy= CompositionToOxidComposition()
feat3=oxy.featurize_dataframe(data_table,col_id='composition',ignore_errors=True)


from matminer.featurizers.composition.ion import OxidationStates
oxida=OxidationStates()
feat4=oxida.featurize_dataframe(feat3,col_id='composition_oxid',ignore_errors=True)


from matminer.featurizers.composition.ion import IonProperty
ostafea= IonProperty()
feat5=ostafea.featurize_dataframe(data_table,col_id='composition',ignore_errors=True)

from matminer.featurizers.composition.orbital import ValenceOrbital
val=ValenceOrbital()
feat6=val.featurize_dataframe(data_table,col_id='composition',ignore_errors=True)

from matminer.featurizers.composition.element import TMetalFraction
met=TMetalFraction()
feat7=met.featurize_dataframe(data_table,col_id='composition',ignore_errors=True)


from matminer.featurizers.composition.element import Stoichiometry
stoich = Stoichiometry()
feat8=stoich.featurize_dataframe(data_table,col_id='composition',ignore_errors=True)



# In[50]:


# Concatenate the datasets horizontally
combined_df = pd.concat([feat1, feat2, feat3,feat4,feat5,feat6,feat7,feat8], axis=1)

# Identify duplicate column names
duplicate_cols = combined_df.columns[combined_df.columns.duplicated()]

# Select only the unique columns
unique_cols = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Save the new dataset
#unique_cols.to_csv('unique_dataset.csv', index=False)


# In[51]:


pd.set_option('display.max_columns',None)


# In[52]:


feat_table=unique_cols.drop(["structure","composition","compound possible","composition_oxid"],axis=1)


# In[53]:


feat_table


# In[54]:


feat_table=feat_table.dropna()


# In[55]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[56]:


y=feat_table.pop('bandgap')
x=feat_table


# In[57]:


x=np.array(x)
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[67]:


import matplotlib.pyplot as plt
import numpy as np

# Instantiate model
model = GradientBoostingRegressor()
# Fit model to training data
model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

import matplotlib.pyplot as plt
import numpy as np

# Calculate the evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print the evaluation metrics
#print(f"Training RMSE: {train_rmse:.3f}")
#print(f"Testing RMSE: {test_rmse:.3f}")
#print(f"Training R^2: {train_r2:.3f}")
#print(f"Testing R^2: {test_r2:.3f}")

# Plot the graph with automatic values
plt.scatter(y_test, y_pred_test, color='blue', label='Testing', s=20)
plt.text(.1, 11.5, f'train RMSE = {train_rmse:.3f}, train R2 = {train_r2:.3f}', style='italic', bbox={
       'facecolor': 'orange', 'alpha': 0.2, 'pad': .1}, fontweight='bold')
plt.text(.1, 11, f'test RMSE = {test_rmse:.3f}, test R2 = {test_r2:.3f}', style='italic', bbox={
        'facecolor': 'blue', 'alpha': 0.2, 'pad': .1}, fontweight='bold')
plt.scatter(y_train, y_pred_train, color='orange', label='Training', marker=6, linewidths=2, s=20)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black')
plt.subplots_adjust(left=0.15, bottom=0.15, top=.95, right=0.95)
plt.ylim([0, 12])
plt.xlim([0, 12])

# Set bold font for all text elements
plt.title('_ _ _ _ Regression', fontweight='bold')
plt.xlabel('Actual Values (eV)', fontweight='bold')
plt.ylabel('Predicted Values (eV)', fontweight='bold')
plt.legend(fontsize=15)

# Set bold frame for the plot
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

#plt.savefig('finalCatBoost10kmodel.png', dpi=550, bbox_inches='tight')
plt.show()



# In[ ]:




