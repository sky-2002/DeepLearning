import torch

batch_size = 10
features = 25
x = torch.rand(size=(batch_size,features)) # 10 rows, 25 columns
xx = torch.rand(size=(2,3,1)) # This is like two 3x1 matrices
# print(xx)
# print(xx[0].shape) # gives 3x1

# print(x[2,:10]) # prints the first 10 features of the 3rd row/example

# Fancy indexing
x = torch.arange(10)
print(x)
inds = [2,5,8]
print(x[inds])

# Advanced indexing
x = torch.arange(8)
print(x[(x<2) | (x>3)]) # gives [0 1 4 5]
print(x[x.remainder(2)==0])

# Useful operations
print(x)
print(torch.where(x>5,x,x*2)) # if element greter than 5, keep as it is, else make it twice
print(x.numel()) # Counting number of elements in x
# getting unique values
print(torch.tensor([0,0,2,2,1,5,6,6]).unique())
