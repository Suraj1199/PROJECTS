#!/usr/bin/env python
# coding: utf-8

# # Pandas
# 
# Pandas is an open-source, BSD-licensed Python library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.In this tutorial, we will learn the various features of Python Pandas and how to use them in practice.
# 
# 
# ## Import pandas and numpy

# In[ ]:


import pandas as pd
import numpy as np


# ### This is your playground feel free to explore other functions on pandas
# 
# #### Create Series from numpy array, list and dict
# 
# Don't know what a series is?
# 
# [Series Doc](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.Series.html)

# In[ ]:


a_ascii = ord('A')
z_ascii = ord('Z')
alphabets = [chr(i) for i in range(a_ascii, z_ascii+1)]

print(alphabets)

numbers = np.arange(26)

print(numbers)

print(type(alphabets), type(numbers))

alpha_numbers = dict(zip(alphabets, numbers))

print(alpha_numbers)

print(type(alpha_numbers))


# In[ ]:


series1 = pd.Series(alphabets)
print(series1)


# In[ ]:


series2 = pd.Series(numbers)
print(series2)


# In[ ]:


series3 = pd.Series(alpha_numbers)
print(series3)


# In[ ]:


#replace head() with head(n) where n can be any number between [0-25] and observe the output in deach case 
series3.head()


# #### Create DataFrame from lists
# 
# [DataFrame Doc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)

# In[ ]:


data = {'alphabets': alphabets, 'values': numbers}

df = pd.DataFrame(data)

#Lets Change the column `values` to `alpha_numbers`

df.columns = ['alphabets', 'alpha_numbers']

df


# In[ ]:


# transpose

df.T

# there are many more operations which we can perform look at the documentation with the subsequent exercises we will learn more


# #### Extract Items from a series

# In[ ]:


ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]

vowels = ser.take(pos)

df = pd.DataFrame(vowels)#, columns=['vowels'])

df.columns = ['vowels']

#df.index = [0, 1, 2, 3, 4]

df


# #### Change the first character of each word to upper case in each word of ser

# In[ ]:


ser = pd.Series(['we', 'are', 'learning', 'pandas'])

ser.map(lambda x : x.title())

titles = [i.title() for i in ser]

titles


# #### Reindexing

# In[ ]:


my_index = [1, 2, 3, 4, 5]

df1 = pd.DataFrame({'upper values': ['A', 'B', 'C', 'D', 'E'],
                   'lower values': ['a', 'b', 'c', 'd', 'e']},
                  index = my_index)

df1


# In[ ]:


new_index = [2, 5, 4, 3, 1]

df1.reindex(index = new_index)

