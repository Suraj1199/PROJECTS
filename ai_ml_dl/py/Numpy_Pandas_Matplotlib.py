#!/usr/bin/env python
# coding: utf-8

# # Data Analysis 
# >Data Analysis is a process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, suggesting conclusions, and supporting decision-making.
# 
# Steps for Data Analysis, Data Manipulation and Data Visualization:
# 
# 1. Tranform Raw Data in a Desired Format
# 2. Clean the Transformed Data (Step 1 and 2 also called as a Pre-processing of Data)
# 3. Prepare a Model
# 4. Analyse Trends and Make Decisions

# # NumPy
# 
# >NumPy is a package for scientific computing.
# 
# 1. Multi dimensional array
# 2. Methods for processing arrays
# 3. Element by element operations
# 4. Mathematical operations like logical, Fourier transform, shape manipulation, linear algebra and random number generation

# In[ ]:


import numpy as np


# >### Creating NumPy arrays

# In[ ]:


# uninitialized ndarray using np.empty(shape, dtype)
x = np.empty((2,2), int)
x


# In[ ]:


# a single dimenstional array
a = np.array([1, 2, 3])
a


# In[ ]:


# a multi-dimensional array
b = np.array([[1, 2], [3, 4]])
b


# In[ ]:


# using np.arange(inclusive start, exclusive end, optional step)
c = np.arange(10,15)
c


# In[ ]:


# array of zeros using np.zeros(shape)
x = np.zeros((4,4))
x


# In[ ]:


# array of ones using np.ones(shape)
y = np.ones((2,2))
y


# In[ ]:


# idendity matrix (nxn) using np.eye(n)
z = np.eye(3)
z


# In[ ]:


# linearly separated array using np.linspace(inclusive strat, inclusive stop, step non-optional)
x = np.linspace(4, 20, 5)
x


# In[ ]:


# convert any python sequence into ndarray using np.asarray(sequence)
l = [1, 2, 3]
print(type(l), l)

x = np.asarray(l)
l


# In[ ]:


# get dimensions of array using array_name.shape
x = np.array([[2,2,3], [6,6,7]])
print("Shape : {}".format(x.shape))


# In[ ]:


# restructuring array dimensions using array_name.reshape(new_shape)
x1d = np.zeros(9)
print("Shape : ",x1d.shape,"\n",x1d)

x2d = x1d.reshape((3,3))
print("\nShape : ", x2d.shape,"\n", x2d)


# In[ ]:


# flatten ndarray into 1-D array using array_name.ravel()
x2d = np.zeros((3,4))
print("Shape : ", x2d.shape, "\n", x2d)

x1d = x2d.ravel()
print("\nShape : ", x1d.shape, "\n", x1d)


# >### Indexing of NumPy arrays

# In[ ]:


a = np.arange(1,10)
a


# In[ ]:


a[4]


# In[ ]:


a = a.reshape((3,3))
a


# In[ ]:


a[1,1]


# ## Slicing

# In[ ]:


a = np.arange(1, 10)
a


# In[ ]:


# using slice(start(inclusive), stop(exclusive), stop(optional))
a[slice(2,6)]


# In[ ]:


# array_name[start(inclusive):stop(exlusive):step(optional)]
a[2:6]


# In[ ]:


# array_name[start(inclusive):___] slice from start till end
a[3:]


# In[ ]:


# array_name[____:stop(exlusive)] slice from beginning to before stop 
a[:6]


# In[ ]:


# slice entire array 
a[:] 


# In[ ]:


a = a.reshape((3,3))
a


# In[ ]:


a[0:2,1:3]


# In[ ]:


a[::2,::2]    # same as a[0:3:2,0:3,2]


# OR, using slice(start, stop, step)

# In[ ]:


a[slice(0,3,2),slice(0,3,2)]


# # Pandas
# >Pandas is an open-source Python library providing efficient, easy-to-use data structure and data analysis tools. (The name Pandas is derived from "Panel Data" - an Econometrics from Multidimensional Data.)
# 
# >Pandas is well suited for many different kinds of data:
# 1. Tabular data with heterogeneously-type columns.
# 2. Ordered and unordered time series data.
# 3. Arbitary matrix data with row and column labels.
# 4. Any other form observational / statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure.
# 
# >Pandas provides three data structure - all of which are build on top of the NumPy array - all the data structures are value-mutable
# 1. Series (1D) - labeled, homogenous array of immutable size.
# 2. DataFrames (2D) - labeled, heterogeneously typed, size-mutable tabular data structures.
# 3. Panels (3D) - Labeled, size-mutable array.
# 

# In[ ]:


import pandas as pd


# ## Series
# > A Series is a single-dimensional array structures that stores homogenous data i.e., data of a single type.
# 
# 1. All the elements of a Series are value-mutable and size-immutable
# 2. Data can be of multiple data types such as ndarray, lists, constants, series, dict etc.
# 3. Indexes must be unique, hashable and have the same length as data. Defaults to np.arrange(n) if no index is passed.
# 4. Data type of each column; if none is mentioned, it will be inferred; automatically
# 5. Deep copies data, set to false as default

# >### Creating Series

# In[ ]:


# empty series using pd.Series() [without parameters]
series = pd.Series()
series


# In[ ]:


# series from list
list_ = [10, 20, 30, 40]
series = pd.Series(list_)
series


# In[ ]:


# series from ndarray using pd.Series(ndarray)
array = np.arange(1,5) * 10
series = pd.Series(array)
series


# NOTE : default indices provided automatically if not specified

# In[22]:


# series from list with custom indices using pd.Series'(list, index=index_list)
list_ = [1, 2, 6]
indices = ['one', 'two', 'six']

series = pd.Series(list_, index=indices)
series


# In[ ]:


# series from dictionary using pd.Series(dict)
dictionary = {'one':1, 'two':2, 'six':6}
series = pd.Series(dictionary)
series


# NOTE : keys of dictionary assigned as indices

# >### Slicing of Series
# 
# same as 1-D array slicing when using default integer indices (0,1,2...)

# In[ ]:


series[0:3:2]


# In[ ]:


series['one':'six':2]


# NOTE : when indices name specified then stop is inclusive

# ## DataFrames
# >A DataFrame is a 2D data structure in which data is aligned in a tabular fashion consisting of rows & columns
# 
# 1. A DataFrame can be created using the following constructor - <code>pandas.DataFrame(data, index, dtype, copy)</code>
# 2. Data can be of multiple data types such as ndarray, list, constants, series, dict etc.
# 3. Index Row and column labels of the dataframe; defaults to np.arrange(n) if no index is passed
# 4. There exists a data type of each column
# 5. Creates a deep copy of the data, set to false as default
# 

# >### Creating DataFrames

# In[ ]:


# dataframe from list of scalar values using pd.DataFrame(list)
data = [10, 20, 30]
df = pd.DataFrame(data)
df


# In[ ]:


# dataframe from list of lists using pd.DataFrame(lists_list)
list1 = [0, 1, 2, 3]
list2 = [2, 4, 6, 8]
list3 = [10, 20, 30, 40]

data = [list1, list2, list3]
df = pd.DataFrame(data)
df


# In[ ]:


# dataframe from series using pd.DataFrame(Series)
series = pd.Series([10, 20, 30, 40])
df = pd.DataFrame(data = series)
df


# In[ ]:


# dataframe from list of series using pd.DataFrame(series_list)
series1 = pd.Series([1, 2, 3, 4])
series2 = pd.Series([3, 6, 9, 12])
series3 = pd.Series([10, 20, 30, 40])

data = [series1, series2, series3]
df = pd.DataFrame(data)
df


# NOTE : 
# 1. default column names and indices (integers) and provided automatically if not specified 
# 2. no. of columns = no. of elements in the list/series
# 3. no. of rows = no. of lists/series in data
# 
# ---

# In[ ]:


# dataframe from list of dictionaries using pd.DataFrame(dicts_list)
dict1 = {'a':10, 'b':20, 'c':30}
dict2 = {'a':15, 'b':25}
dict3 = {'a':20, 'c':35}

data = [dict1, dict2, dict3]
df = pd.DataFrame(data)
df


# NOTE : 
# 1. <code>NaN</code> - Not a number , provided in areas where data is missing
# 2. name of columns = keys
# 3. no. of columns = no. of keys
# 4. no. of rows = no. of dictionaries
# 5. default indices (0,1,2...)

# In[ ]:


# same dataframe with custom index names
indices = ['first', 'second', 'third']
df = pd.DataFrame(data, index = indices)
df


# NOTE : no. of elements in 
# indices must be equal to no. of rows ( if not then results in error)
# <br>
# <br>
# 
# ---

# In[23]:


# dataframe from dictionary of lists using pd.DataFrame(lists_dict)
list1 = [1, 2, 3]
list2 = [2, 4, 6]
list3 = [1, 3, 5]

dictionary = {'nums':list1, 'even':list2, 'odd':list3}
df = pd.DataFrame(dictionary)
df


# In[24]:


# same dataframe with custom indices using pd.DataFrame(lists-dict, index=index_list)
indices = ['first', 'second', 'third']

df = pd.DataFrame(dictionary, index=indices)
df


# In[25]:


# dataframe from dictionary of series using pd.DataFrame(series_dict)
series1 = pd.Series([1, 2, 3])
series2 = pd.Series([2, 4, 6])
series3 = pd.Series([1, 3, 5])

dictionary = {'nums':series1, 'even':series2, 'odd':series3}
df = pd.DataFrame(dictionary)
df


# In[ ]:


# dataframe from dictionary of deries with custom indices using pd.DataFrame(series-with-index_dic)
series1= pd.Series([1,2,3], index=['first', 'second', 'third'])
series2 = pd.Series()

