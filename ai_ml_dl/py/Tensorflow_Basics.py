#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


x = tf.constant(11)


# In[4]:


type(x)


# In[ ]:


#creating tensorflow session

session = tf.Session()


# In[7]:


type(session)


# In[8]:


session.run(x)


# In[9]:


type(session.run(x))


# In[ ]:


#Operations

a = tf.constant(3)
b = tf.constant(2)


# In[22]:


with tf.Session() as session:
    print(f"Addition: {session.run(a+b)}")
    print(f"Subtraction: {session.run(a-b)}")
    print(f"Multiplication: {session.run(a*b)}")
    print(f"Division:{session.run(a/b)}")


# In[29]:


with tf.Session() as s:
    print("Addition :",s.run(x/y,d)) #d = Dictionary of Placeholder Values


# In[ ]:


#Placeholder
x = tf.placeholder(tf.int64)
y = tf.placeholder(tf.int64)


# In[ ]:


d = {x:10,y:2} #Dictionary of Placeholder values required


# In[43]:


with tf.Session() as session:
    print(f"Addition: {session.run(tf.add(x,y),feed_dict = d)}")
    print(f"Subtraction: {session.run(tf.subtract(x,y),d)}")
    print(f"Multiplication: {session.run(tf.multiply(x,y),d)}")
    print(f"Division:{session.run(tf.divide(x,y),d)}")


# In[ ]:


import numpy as np


# In[58]:


a = np.array([[1.0,2.0,5.0]])
b = np.array([[10.0],[5.0],[2.0]])

print("a :",a.shape)
print("b :",b.shape)


# In[ ]:


#Converting into Tensor

mat1 = tf.constant(a)
mat2 = tf.constant(b)


# In[62]:


mul_mats = tf.matmul(mat1,mat2)  

with tf.Session() as session:
    print("Resultant Matrix : ",session.run(mul_mats))


# In[ ]:




