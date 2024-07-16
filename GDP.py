#!/usr/bin/env python
# coding: utf-8

# ### Data Cleaning And Visualization

# # GDP 

# The dataset (https://drive.google.com/file/d/1h-_Vykiu8qOej8qrhsKrIuSk88E4_C9P/) is downloaded and imported into Python through a NumPy array or Pandas data frame.
# 
# 1)There are some missing values in the ‘Population’ column. Will Populate these missing values with the correct values in Python (not copy-paste) from the ‘Population.csv’ file (https://drive.google.com/file/d/1RViH13AHyAvw4datdgjdm4v2JgTa5C51/).
# 
# 2)After the data is cleaned and structured in NumPy or Pandas, will need to create a new ‘GDP Per Capita’ column and populate it with the correct values, which are calculated based on the table’s Population and GDP values.
# 
# 3)The data is now cleaned, structured and ready to be graphed. Using matplotlib, will create a graph that meets the following requirements:
# 3.1) Descriptive title above the graph
# 3.2) X-Axis (GDP per capita) has a label
# 3.3) Y-Axis (Life Expectancy) has a label
# 3.4) GDP per Capita and Life Expectancy data is graphed as a scatter plot
# 3.5) Scatter plot points are sized based on the country’s population (a larger population warrants a larger point size)
# 3.6) Scatter plot points are labelled by Country)

# In[1]:


# importing libariries

# pandas
import pandas as pd

# numpy
import numpy as np

# finding mean (average) of rental rates
#import statistics as stat

# matplotlib library for visualization
import matplotlib.pyplot as plt

#import seaborn library for visualization
import seaborn as sns
#%matplotlib inline


# In[2]:


# loading datasets
dfGdp = pd.read_csv('GDP-Life Expectancy.csv');
dfPopulation = pd.read_csv('Population.csv');


# In[3]:


# viewing dataset
dfGdp.head()


# In[4]:


# coulumns and rows
dfGdp.shape


# In[5]:


# description of dataset
dfGdp.describe()


# In[6]:


dfGdp.info()


# In[7]:


dfGdp.columns


# 1)There are some missing values in the ‘Population’ column will populate these missing values with the correct values in Python (not copy-paste) from the ‘Population.csv’ file (https://drive.google.com/file/d/1RViH13AHyAvw4datdgjdm4v2JgTa5C51/).
# 
# the population has 6 missing values so we need to fill it from population csv file

# In[8]:


dfPopulation.head()


# In[9]:


dfPopulation.columns


# In[10]:


# finding nan values in GDP - Live Expectancy

nanValues=dfGdp[dfGdp.isnull().any(axis=1)]
nanValues


# In[11]:


# finding countries that have nan values in GDP - Live Expectancy table
nanValues.Country


# In[12]:


# separating countries in "POPULATION table" that have NAN values in "Population field" from GDP
df_population_missing_values = dfPopulation[dfPopulation['CountryName'].isin(nanValues.Country)]

# separating only country and population field from above data frame
df_population_missing_values = pd.DataFrame(
    {'Country' : df_population_missing_values['CountryName'], 'Population' : df_population_missing_values['2019']})
df_population_missing_values


# In[13]:


df_population_missing_values['Population']


# In[14]:


#You can set Country as index for both dataframes, and then use 
#the fillna() method, which fill missing values, while matching the index of the two dataframes:

#filling NAN values in GDP table with the values in Population table
dfGdp = dfGdp.set_index("Country").fillna(df_population_missing_values.set_index("Country")).reset_index()


# ### Data Preprocessing

# Step 1 : sanity check

# In[15]:


# info
dfGdp.info()


# #### missing values has been treated

# In[16]:


# check duplicate values
dfGdp.duplicated().sum()


# In[17]:


# identifying garbage values
for i in dfGdp.select_dtypes(include="object").columns:
    print(dfGdp[i].value_counts())
    print ('***' * 10)


# Step 2 : EDA

# 1) descriptive analysis

# In[18]:


# descriptive statistics
dfGdp.describe().T


# In[19]:


dfGdp.describe(include="object")


# In[20]:


# histogram to understand the distribution
sns.set_style('darkgrid')
for i in dfGdp.select_dtypes(include="number").columns:
    sns.histplot(data=dfGdp,x=i, bins=10, kde=True)
    plt.axvline(x=np.mean(dfGdp[i]),color='red',ls='dashed', label='mean')
    plt.axvline(x=np.percentile(dfGdp[i],25),color='green',ls='dotted', label='25%')
    plt.axvline(x=np.percentile(dfGdp[i],75),color='lightgreen',ls='dotted', label='75%')
    plt.title(i)
    plt.legend()
    plt.show()
    


# In[21]:


# Boxplot to identify outliers

for i in dfGdp.select_dtypes(include="number").columns:
    sns.boxplot(data=dfGdp,x=i)

    plt.show()


# 2)After the data is cleaned and structured in NumPy or Pandas, will create a new ‘GDP Per Capita’ column and populate it with the correct values, which are calculated based on the table’s Population and GDP values.
# 
# GDP Per Capita = GDP of the Country / Population Of that Country

# In[22]:


dfGdp.insert(loc=len(dfGdp.columns), column = "GDP Per Capita", value =(dfGdp['PPP Adjusted GDP']/dfGdp['Population']))


# In[23]:


dfGdp.head()


# 3)The data is now cleaned, structured and ready to be graphed. Using matplotlib, will create a graph that meets the following requirements:
# 3.1)Descriptive title above the graph
# 3.2)X-Axis (GDP per capita) has a label
# 3.3)Y-Axis (Life Expectancy) has a label
# 3.4)GDP per Capita and Life Expectancy data is graphed as a scatter plot
# 3.5)Scatter plot points are sized based on the country’s population (a larger population warrants a larger point size)
# 3.6)Scatter plot points are labelled by Country)

# In[24]:


x=dfGdp['GDP Per Capita']
y=dfGdp['Life Expectancy']
color = dfGdp['Population']
font={
    'color' : 'red'
}
box= {'facecolor' : 'white',
     'alpha': 0.2}
plt.figure(figsize=(15,10))
plt.scatter(x, y,s=(dfGdp['Population']/dfGdp['Population'].min())**1.4,c=color)

for i in dfGdp['Country']:
    plt.text(dfGdp['GDP Per Capita'][dfGdp['Country']==i], dfGdp['Life Expectancy'][dfGdp['Country']==i]-0.36, s=i,
             bbox=box, ha="center", va="center_baseline", fontsize='x-small', fontdict=font)
    
plt.title("Life expectancy vs. GDP Per Capita")
plt.xlabel('GDP Per Capita')
plt.ylabel('Life Expectancy')
plt.show()


# In[ ]:




