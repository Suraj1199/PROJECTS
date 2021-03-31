#!/usr/bin/env python
# coding: utf-8

# # Numpy Examples
# 
# ## What is numpy?
# 
# #### Python has built-in:
# 
# -  containers: lists (costless insertion and append), dictionnaries (fast lookup)
# -  high-level number objects: integers, floating point
# 
# #### Numpy is:
# 
#  - extension package to Python for multidimensional arrays
#  - closer to hardware (efficiency)
#  - designed for scientific computation (convenience)
# 
# 
# #### Import numpy
# 
# 

# In[ ]:


import numpy as np


# 
# #### Create numpy arrays
# 

# In[3]:


a = np.array([1, 2, 3])   # Create a rank 1 array
print(a)
print(type(a)) #print type of a

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])


# #### Some basic functions for creating arrays. Print all the defined arrays and see the results.

# In[4]:


a = np.zeros(shape=(2,2))
b = np.ones(shape = (3,3))
c = np.eye(2)
d = np.full(shape=(3,3), fill_value=5)
e = np.random.random((2,2))

print('a', a)
print('b',b)
print('c',c)
print('d',d)
print('e',e)


# #### Execute and understand :)

# In[7]:


a = np.arange(10)
b = np.linspace(0,10, num=6)
print(a)
print(b)


# #### Array Indexing

# In[9]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.

print(b)   

b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a)   # Prints modified a


# #### Slicing

# In[12]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a

print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]

print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)


# #### Aritmetic operations

# In[16]:


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"


# #### Using Boolean Mask

# In[18]:


b = np.arange(10)

print(b)

mask = b%2!=0 #perform computations on the list 

print(mask)

print(b[mask]) #applying the mask on the numpy array


# In[19]:


modified_b = b
modified_b[mask] = -1

print(modified_b)


# #### Swapping two columns in a 2d numpy array

# In[23]:


a = np.arange(12).reshape(3,4)
print(a)

print(a[:, [1,0,3,2]])


# #### Swapping two rows in a 2d numpy array

# In[24]:


a = np.arange(12).reshape(3,4)
print(a)

print(a[[1,0,2], :])

