#!/usr/bin/env python
# coding: utf-8

# # Numpy Exercises
# 
# 1) Create a uniform subdivision of the interval -1.3 to 2.5 with 64 subdivisions

# In[76]:


import numpy as np #import numpy
a = np.linspace(1.3,2.5,64)
print (a)


# 2) Generate an array of length 3n filled with the cyclic pattern 1, 2, 3

# In[77]:


a = np.array([1,2,3])

x = np.resize(a,12)
print(x)


# 3) Create an array of the first 10 odd integers.

# In[78]:


a = np.arange(20)

i = a%2!=0

b = a[i]
print (b)


# 4) Find intersection of a and b

# In[79]:


#expected output array([2, 4])
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = list()
k=0
for value in a:
    for values in b:
        if value == values and value not in c:
            c.append(value)
            break
            
x = np.asarray(a)
x


# 5) Reshape 1d array a to 2d array of 2X5

# In[80]:


a = np.arange(10)
print(a)

print(a.reshape(2,5))


# 6) Create a numpy array to list and vice versa

# In[81]:


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a)
#converting to list
b = a.tolist()
print (b)
#again converting it to an array
c =np.asarray(b)
print (c)


# 7) Create a 10 x 10 arrays of zeros and then "frame" it with a border of ones.

# In[82]:


a = np.zeros(shape = (10,10))
for i in range(0,10):
    for j in range(0,10):
        if i == 0 or i == 9 or j == 0 or j == 9:
            a[i,j] = 1
          
print (a)


# 8) Create an 8 x 8 array with a checkerboard pattern of zeros and ones using a slicing+striding approach.

# In[83]:


a = np.zeros(shape = (8,8)) 
    # fill with 1 the alternate rows and columns 
  
a[1::2, ::2] = 1
a[::2, 1::2] = 1
      
print (a)

