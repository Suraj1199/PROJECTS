#!/usr/bin/env python
# coding: utf-8

# # Get to know your Data
# 
# 
# #### Import necessary modules
# 

# In[ ]:


import pandas as pd
import numpy as np


# #### Loading CSV Data to a DataFrame

# In[ ]:


iris_df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')


# #### See the top 10 rows
# 

# In[ ]:


iris_df.head()


# #### Find number of rows and columns
# 

# In[ ]:


print(iris_df.shape)

#first is row and second is column
#select row by simple indexing

print(iris_df.shape[0])
print(iris_df.shape[1])


# #### Print all columns

# In[ ]:


print(iris_df.columns)


# #### Check Index
# 

# In[ ]:


print(iris_df.index)


# #### Right now the iris_data set has all the species grouped together let's shuffle it

# In[ ]:


#generate a random permutaion of indices
print(iris_df.head())

new_index = np.random.permutation(iris_df.index)
iris_df = iris_df.reindex(index = new_index)

print(iris_df.head())


# #### We can also apply an operation on whole column of iris_df

# In[ ]:


#original

print(iris_df.head())

iris_df['sepal_width'] *= 10

#changed

print(iris_df.head())

#lets undo the operation

iris_df['sepal_width'] /= 10

print(iris_df.head())


# #### Show all the rows where sepal_width > 3.3

# In[ ]:


iris_df[iris_df['sepal_width']>3.3].head()


# #### Club two filters together - Find all samples where sepal_width > 3.3 and species is versicolor

# In[ ]:


iris_df[(iris_df['sepal_width']>3.3) & (iris_df['species'] == 'versicolor')] 


# #### Sorting a column by value

# In[ ]:


iris_df.sort_values(by='sepal_width')#, ascending = False)
#pass ascending = False for descending order


# #### List all the unique species

# In[ ]:


species = iris_df['species'].unique()

print(species)


# #### Selecting a particular species using boolean mask (learnt in previous exercise)

# In[ ]:


setosa = iris_df[iris_df['species'] == species[1]]

setosa.head()


# In[ ]:


# do the same for other 2 species 
versicolor = iris_df[iris_df['species'] == species[0]]

versicolor.head()


# In[ ]:




virginica = iris_df[iris_df['species'] == species[2]]

virginica.head()


# #### Describe each created species to see the difference
# 
# 

# In[ ]:


setosa.describe()


# In[ ]:


versicolor.describe()


# In[ ]:


virginica.describe()


# #### Let's plot and see the difference

# ##### import matplotlib.pyplot 

# In[ ]:


import matplotlib.pyplot as plt

#hist creates a histogram there are many more plots(see the documentation) you can play with it.

plt.hist(setosa['sepal_length'])
plt.hist(versicolor['sepal_length'])
plt.hist(virginica['sepal_length'])

