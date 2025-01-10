#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numpy
import numpy as np
a=np.array(20)
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([2,9,6,5,'a'])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([2,9,6,5,15])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([[2,3,4,5],[6,7,8,9],[1,6,5,3]])
print('dimension:',a.ndim)
print('shape:',a.shape)
print(a)


# In[ ]:


import numpy as np
a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print('dimension:',a.ndim)
print('shape:',a.shape)
# print(a)
# a=a.reshape(2,1,3,3)
# a=a.reshape(1,2,3,3)

# a=a.reshape(-1)
a=a.reshape(2,3,3)



print(a)


# In[ ]:


import numpy as np
l=[]
for i in range(1,211):
    l.append(i)
a=np.array(l)
print('dimension:',a.ndim)
print('shape:',a.shape)
a=a.reshape(5,6,7)
print(a[3,1,6])
print(a[0:3:2,::2,3:6])
print(a[0:,4:6,3])
print(a[])

print(a)


# In[ ]:


s=0
for i in arr:
    for j in i:
        st=j[1]


# In[3]:


import numpy as np
l=[]
for i in range(18):
    l.append(i+1)
sum=0
a=np.array(l)
a=a.reshape(3,2,3)
for i in a:
    for j in i:
        sum+=j[1]
print(sum)


# In[5]:


import numpy as np
a1=np.array([1,2,3])
a2=np.array([4,5,6,7])
print(a1.shape)
print(a2.shape)
c=np.concatenate((a1,a2))
print(c.shape)
print(c)


# In[20]:


# import numpy as np
# a1=np.array([[[1,2,3],[4,5,6]],[7,8,9],[[10,11,12]]])
# a2=np.array([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
# print(a1.shape)
# print(a2.shape)
# c=np.concatenate((a1,a2),axis=1)
# print(c.shape)
# print(c)


# In[19]:


# import numpy as np
# a1 = np.array([[[1, 2, 3], [4, 5, 6]], [7, 8, 9], [[10, 11, 12]]])
# a2 = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
# print(a1.shape)
# print(a2.shape)
# c = np.concatenate((a1, a2), axis=1)
# print(c.shape)
# print(c)


# In[ ]:




