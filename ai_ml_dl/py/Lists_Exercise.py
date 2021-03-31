#!/usr/bin/env python
# coding: utf-8

# # Exercise - List

# 

# 1) Create  any random list and assign it to a variable dummy_list

# In[ ]:


rand_list = [7,11.11,19,11]
dummy_list = rand_list


# 2) print dummy_list

# In[9]:


print (dummy_list)


# 3) Reverse dummy_list and print

# In[10]:


dummy_list.reverse()

print (dummy_list)


# 4) Add the list dummy_list_2 to the previous dummy_list and now print dummy_list

# In[11]:


dummy_list_2 = [2, 200, 16, 4, 1, 0, 9.45, 45.67, 90, 12.01, 12.02]

for i in dummy_list_2:
    dummy_list.append(i)

print(dummy_list)


# 5) Create a dictionary named dummy_dict which contains all the elements of dummy_list as keys and frequency as values. 

# In[ ]:





# In[ ]:





# 6) print dummy_dict

# In[ ]:





# 7) Sort dummy_list in ascending order as well as descending order and print the changed lists 

# In[12]:


print (dummy_list)

print(sorted(dummy_list))

print(sorted(dummy_list,reverse=True))


# 8) Remove the first item from the list whose value is equal to x. It raises a ValueError if there is no such item.

# In[13]:


x = 200
dummy_list.remove(x)
print(dummy_list)
# Let's play: try the same with something which is not in the list to get the ValueError,remove #

#x = 5
#dummy_list.remove(x)
#print(dummy_list)


# 9) Remove the item at position x. x is any random integer

# In[16]:


dummy_list.pop(2)   #11.11 will be deleted
print(dummy_list)

# Let's play: try doing the same with x > len(dummy_list) + 1 and see what you get
#remove # and the following will give indexError

#dummy_list.pop(50)


# 10) Let's clean everything clear the list and then print

# In[ ]:


dummy_list = []
print (dummy_list)

