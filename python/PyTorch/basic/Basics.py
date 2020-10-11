#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np


# In[3]:


x = torch.empty(5,3)
print(x)


# In[4]:


x = torch.rand(5,3)
print(x)


# In[6]:


torch.zeros(5,3, dtype=torch.long)


# In[7]:


torch.tensor([1.23,3])


# In[8]:


x = x.new_ones(5,3, dtype=torch.double)
print(x)


# In[10]:


x = torch.randn_like(x,dtype=torch.float)
print(x)


# In[12]:


x = torch.arange(0,19,2)
print(x)


# In[14]:


x = torch.linspace(0,3,4)
print(x)


# In[15]:


x = torch.FloatTensor([5,6,7])
print(x)
print(x.dtype)


# In[16]:


x = torch.tensor([5,6,7],dtype=torch.int)


# In[17]:


print(x.dtype)


# In[18]:


torch.manual_seed(42)
x = torch.rand(2,3)
print(x)


# In[20]:


torch.manual_seed(42)
x = torch.rand(2,3)
print(x)


# In[21]:


x.dtype


# In[22]:


x.size()


# In[24]:


x.shape


# In[26]:


x = torch.rand(5,3)
print(x)


# In[29]:


y = torch.rand(5,3)
print(y)


# In[30]:


print(x + y)


# In[31]:


result = torch.empty(5,3)
torch.add(x,y, out=result)
print(result)


# In[32]:


y.add_(x)
print(y)


# In[33]:


print(y[:,1])


# In[34]:


print(y[0,:])


# In[35]:


print(y[2:,1:])


# In[37]:


x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())


# In[40]:


x = torch.randn(1)
print(x)
print(x.item())


# In[5]:


a = torch.ones(3)
print(a)


# In[7]:


b = a.numpy()
print(b)


# In[8]:


a.add_(3)
print(a)
print(b)


# In[10]:


a = np.ones(3)
print(a)


# In[11]:


b = torch.from_numpy(a)
print(b)


# In[15]:


a = torch.arange(1,4,1,dtype=torch.float)
print(a)


# In[16]:


a.mean()


# In[17]:


a.sum()


# In[18]:


print(a.max())
print(a.min())


# In[19]:


a = torch.tensor([1,2,3],dtype=torch.float)
b = torch.tensor([4,5,6],dtype=torch.float)

print(torch.add(a,b).sum())


# In[21]:


import math
a = torch.tensor([math.radians(30), math.radians(60), math.radians(90)])


# In[22]:


print(torch.sin(a))
print(torch.cos(a))
print(torch.tan(a))
print(torch.asin(a))
print(torch.tanh(a))


# In[23]:


a = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
b = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float)


# In[24]:


torch.mm(a,b)


# In[25]:


print(a@b)


# In[ ]:





# In[45]:


x = torch.ones(2,2, requires_grad = True)


# In[46]:


y = x + 2


# In[47]:


z = (y**2).mean()


# In[48]:


z.backward()
print(x.grad)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




