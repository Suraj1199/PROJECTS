#!/usr/bin/env python
# coding: utf-8

# #Dictionaries
# 
# [Source](https://github.com/iArunava/Python-TheNoTheoryGuide/)

# In[1]:


# Simple Dictionary
# Dictionary allows to have key:value pairs
d1 = {"Jennifer":8, 'A':65, 66:'B', 9.45:"Decimals"}

print (d1["Jennifer"])
print (d1['A'])
print (d1[66])
print (d1[9.45])


# In[4]:


# Adding new ke y:value pairs
d1 = {"Jennifer":8, 'A':65, 66:'B', 9.45:"Decimals"}

d1["Scarlett":8,|z7.56:"Is a decimal!",'Q':17]


# In[ ]:


# Declaring an empty dictionary
d1 = {}

# Add new values
d1["Jennifer"] = "Python"
d1["Scarlett"] = "Python"
d1[45]         = 56

print (d1)


# In[ ]:


# Modifying values in a dictionary
d1 = {"Python":"Is a language", "Jennifer":"Feels like a supergirl with Python"}

d1["Python"] = "Is Love"
d1["Jennifer"] = 8
print (d1)


# In[ ]:


# Removing Key:Value pairs
d1 = {"Key":"Value", "Jennifer":"Scarlett", "Scarlett":"Jennifer"}

del d1["Key"]
print (d1)


# In[ ]:


# Storing a dictionary inside a dictionary
d1 = {'A':65, 'B':66, 'C':67, 'D': {
    "Breaking": "Dict into dicts", 
    "Inception": "All over", 
    '!': "XD"}, 
    'E':69, 
    "What happened with D?": "It became a 'D'ictionary! XD",
    66:66}

print (d1['D']["Inception"])
print (d1["What happened with D?"])
print ('\n')
print (d1)


# In[ ]:


# Looping through a dictionary
for key, value in d1.items():
    print("Key: "    + str(key))
    print ("Value: " + str(value))
    print('\n')


# In[ ]:


# Looping through the keys in dictionary
for key in d1.keys():
    print("Key: "    + str(key))
    print ("Value: " + str(d1[key]))
    print('\n')


# In[ ]:


# Note: Default behaviour is to loop through keys if not specified d1.keys()
for k in d1:
    print("Key: "    + str(k))


# In[ ]:


# Check if a particular key is  present
if 'F' not in d1.keys():
    print ("What happened to F?")


# In[ ]:


# Looping through dictonary keys in sorted order (increasing)
for key in sorted(d1.keys()):
    print ("Key :" + key + '\t' + "Value :" + str(d1[key]) + '\n')


# In[ ]:


# Looping through dictonary keys in sorted order (decreasing)
for key in sorted(d1.keys(), reverse=True):
    print ("Key :" + key + '\t' + "Value :" + str(d1[key]) + '\n')


# In[ ]:


# Looping through values in dictionary (with repeats)
# Note: If two or more keys have the same value then this method will print all of them
for value in d1.values():
    print (str(value).title())


# In[ ]:


# List in Dictionary
d1 = {"l1":['A', 'B', 'C', 'D'],
      "l2":['E', 'F', 'G', 'H'],
       45 : "qwerty",
      '$' : "$Dollar$"}

# Accessing the elements in list
print (d1["l1"][2])
print (d1["l2"][-1])


# In[ ]:


# Looping over just the lists in dictionary
for k in d1.keys():
    if type(d1[k]) == list:
        print ("List :" + k)
        for val in d1[k]:
            print (val)
        print ('\n')

