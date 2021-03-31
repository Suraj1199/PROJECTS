#!/usr/bin/env python
# coding: utf-8

# # lists
# 
# [Source](https://github.com/iArunava/Python-TheNoTheoryGuide/)

# In[ ]:


# Simple Lists
names = ["Jennifer", "Python", "Scarlett"]
nums  = [1, 2, 3, 4, 5]
chars = ['A', 'q', 'E', 'z', 'Y']

print (names)
print (nums)
print (chars)


# In[ ]:


# Can have multiple data types in one list
rand_list = ["Jennifer", "Python", "refinneJ", 'J', '9', 9, 12.90, "Who"]
print (rand_list)


# In[ ]:


# Accessing elements in a list
# O-indexed
print (names[2])
print (rand_list[3])
print (names[0] + " " + rand_list[2].title())


# In[ ]:


# Negetive indexes: Access elements from the end of the list without knowing the size of the list
print (rand_list[-1]) # Returns the last element of the list [1st from the end]
print (rand_list[-2]) # Returns the 2nd last element
# and so on..


# In[ ]:


# Now here's a question.
print (rand_list[-1] + " is " + names[2] + "?")
print ("A) " + rand_list[0] + "'s sister\tB) " + names[0] + "'s Friend\nC) Not Related to " + rand_list[-8] + "\tD) Nice question but I don't know")


# In[ ]:


# Modifying elements in a list
str_list = ["Scarlett", "is", "a", "nice", 'girl', '!']

print (str_list)
str_list[0] = "Jennifer"
print (str_list)


# In[ ]:


# Adding elements to a list
# Use append() to add elements to the end of the list
str_list.append ('She is 21.')
print (str_list)


# In[ ]:


# So, you can build lists like this
my_list = []
my_list.append ("myname")
my_list.append ("myage")
my_list.append ("myaddress")
my_list.append ("myphn")
my_list.append ("is")
my_list.append (1234567890)
print (my_list)


# In[ ]:


# Insert elements at specific positions of the list
# insert(index, element)
my_list.insert (0, "Mr/Miss/Mrs")
print (my_list)

my_list.insert(4, "mybday")
print (my_list)


# In[ ]:


# Using '-1' to insert at the end doesn't work and inserts element at the 2nd last position.
my_list = ['A', 'B', 'C', 'D']
my_list.insert (-1, 'E')
print (my_list)


# In[ ]:


# Using '-2' inserts at 3rd last position
# In general, use '-n' to insert at 'n+1'th position from end.
my_list = ['A', 'B', 'C', 'D']
my_list.insert (-2, 'E')
print (my_list)


# In[ ]:


# Insert elements at the end
l1 = ['A', 'B', 'C', 'D']
l2 = ['A', 'B', 'C', 'D']

l1.append('E')
l2.insert(len(my_list), 'E')
print (l1)
print (l2)


# In[ ]:


# Length of the list
l1 = ['A', 'B', 'C', 'D', 'E']
print (len(l1))


# In[ ]:


# # Removing elements from list
# del can remove any element from list as long as you know its index
l1 = ['A', 'B', 'C', 'D', 'E']
print (l1)

del l1[0]
print (l1)

del l1[-1]
print (l1)


# In[ ]:


# pop() can remove the last element from list when used without any arguments
l1 = ['A', 'B', 'C', 'D', 'E']
# pop() returns the last element, so c would store the popped element
c = l1.pop()

print (l1)
print (c) 


# In[ ]:


# pop(n) -> Removes the element at index 'n' and returns it
l1 = ['A', 'B', 'C', 'D', 'E']

# Removes the element at 0 position and returns it
c = l1.pop(0)
print (l1)
print (c)

# Works as expected with negetive indexes
c = l1.pop(-1)
print (l1)
print (c)


# In[ ]:


# Removing an item by value
# remove() only removes the first occurence of the value that is specified.
q1 = ["Seriously, ", "what", "happened", "to", "Jennifer", "and", "Jennifer", "?"]
print (q1)

q1.remove ("Jennifer")
print (q1)

n1 = "and"
q1.remove(n1)
print (q1)


# In[ ]:


# Sorting a list
# sort() -> sorts list in increasing or decreasing order, *permantantly*
# Sorts in alphabetical order
l1 = ['E', 'D', 'C', 'B', 'A']
l1.sort()
print (l1)

# Sorts in increasing order
l2 = [2, 200, 16, 4, 1, 0, 9.45, 45.67, 90, 12.01, 12.02]
l2.sort()
print (l2)


# In[ ]:


# Reverse sorts alphabetical order
l1 = ['E', 'D', 'C', 'B', 'A']
l1.sort(reverse=True)
print (l1)

# Sorts in decreasing order
l2 = [2, 200, 16, 4, 1, 0, 9.45, 45.67, 90, 12.01, 12.02]
l2.sort(reverse=True)
print (l2)


# In[ ]:


# sorted() -> Sorts list in increasing or decreasing order, *temporarily*
# Sorts in increasing order
l2 = [2, 200, 16, 4, 1, 0, 9.45, 45.67, 90, 12.01, 12.02]
print (l2)
print (sorted(l2))
print (l2)


# In[ ]:


# Sorts in decreasing order
l2 = [2, 200, 16, 4, 1, 0, 9.45, 45.67, 90, 12.01, 12.02]
print (l2)
print (sorted(l2, reverse=True))
print (l2)


# In[ ]:


# Reverse list
l1 = ['E', 'D', 'C', 'B', 'A']
l1.reverse()
print (l1)


# In[ ]:


# Looping Through a list using for
l1 = ["Scarlett", "is", "now", "back", "from", "her first", "Python", "lesson."]

# Do notice the indentations
for each_word in l1:
    print (each_word)


# In[ ]:


# Looping through a list using while
l1 = ["Scarlett", "is", "in", "love", "with", "Python"]
i = 0
while i is not len(l1):
    print (l1[i])
    i += 1


# In[ ]:


# Numerical lists
# Note: range(n, m) will loop over numbers from n to m-1
l1 = ['A', 'B', 'C', 'D', 'E']
print ("Guess how much Scarlett scored in her first lesson out of 5:")
for val in range(1, 6):
    print (l1[val-1] + ") " + str(val))


# In[ ]:


# Using range() to make a list of numbers
num_list = list(range(1, 6))
print (num_list)


# In[ ]:


# Use range() to skip values at intervals
# range (num_to_start_from, num_to_end_at+1, interval)
l1 = list(range(10, 51, 5))
print (l1)


# In[ ]:


# Operations with list of numbers with -> min() max() sum()
l1 = [2, 3, 4, 45, 1, 5, 6, 3, 1, 23, 14]

print ("Sum: " + str(sum(l1)))
print ("Max: " + str(max(l1)))
print ("Min: " + str(min(l1)))


# In[ ]:


# List Comprehensions
# Simple
l1 = [i for i in range(20, 30, 1)]
l2 = [i+1 for i in range(20, 30, 1)]
l3 = [[i, i**2] for i in range(2, 12, 3)]
print (l1)
print (l2)
print (l3)


# In[ ]:


# A few more list comprehension examples
equi_list_1 = [[x, y, z] for x in range(1, 3) for y in range(3, 6) for z in range(6, 9)]
print (equi_list_1)


# In[ ]:


# The above list comprehension is equivalent of the following code
equi_list_2 = []
for x in range(1, 3):
    for y in range(3, 6):
        for z in range(6, 9):
            equi_list_2.append([x, y, z])
print (equi_list_2)


# In[ ]:


# Proof of equivalence  (Do execute the above two blocks of code before running this)
print (equi_list_1 == equi_list_2)


# In[ ]:


# List Comprehension with conditionals
l1 = [x if x%2==0 else "Not" for x in range(0,10)]
print (l1)


# In[ ]:


# One more list comprehension with conditionals
l1 = ["Jennifer", "met", "Scarlett", "in", "Python", "lessons", "they", "take."]
l2 = [[str(x) + ") " + y] for x in range(len(l1)) for y in l1 if l1[x] == y]
print (l2)


# In[ ]:


# Slicing a list
l1 = ["Jennifer", "is", "now", "friends", "with", "Scarlett"]

# [start_index : end_index+1]
print("[2:5] --> " + str(l1[2:5]))
print("[:4]  --> " + str(l1[:4]))  # everthing before 4th index [excluding the 4th]
print("[2:]  --> " + str(l1[2:]))  # everything from 2nd index [including the 2nd]
print("[:]   --> " + str(l1[:]))   # every element in the list


# In[ ]:


# Some more slicing
l1 = ["Jennifer", "and", "Scarlett", "now", "Pythonistas", "!"]

print ("[-2:]   --> " + str(l1[-2:]))
print ("[:-3]   --> " + str(l1[:-3]))
print ("[-5:-2] --> " + str(l1[-5:-2]))
print ("[-4:-6] --> " + str(l1[-4:-6]))


# In[ ]:


# Looping through a slice
l1 = ["Pythonistas", "rock", "!!!", "XD"]
for w in l1[-4:-1]:
    print (w.upper())


# In[ ]:


# Copying a list
l1 = ["We", "should", "use", "[:]", "to", "copy", "the", "whole", "list"]
l2 = l1[:]
print(l2)


# In[ ]:


# Proof that the above two lists are different
l2.append(". Using [:] ensures the two lists are different")

print (l1)
print (l2)


# In[ ]:


# What happens if we directly assign one list to the other instead of using slices
l1 = ["Jennifer", "now", "wonders", "what", "happens", "if", "we", "directly", "assign."]
l2 = l1
l2.append("Both variables point to the same list")

print (l1)
print (l2)

