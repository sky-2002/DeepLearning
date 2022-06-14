import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1) # This will have type float32
# print(z1,z1.dtype)
z2 = torch.add(x,y) # This will have type int64
# print(z2,z2.dtype)
z3 = x+y # This will have type int64
# print(z3,z3.dtype)

# Subtraction
z4 = x-y

# Division
z5 = torch.true_divide(x,y)
# If shape of x and y is same, then each element of x is divided
# by the corresponding element of y. If y is an integer, every
# element of x will be divided by y
# print(z5)

# Inplace operations
t = torch.zeros(3)
t.add_(x) # Whenever there is an underscore, they are inplace operations
# t+=x will also do the same thing

# Exponentiation
print(x.pow(2),x.__pow__(2),x**2) # All give the same result

# Comparison
print(x<0) # compares element wise

# Matrix multiplication
x1 = torch.rand(size=(2,5))
x2 = torch.rand(size=(5,3))
x3 = torch.mm(x1,x2) # 2x3, can also written as x1.mm(x2)
# print(x3)

# Matrix exponentiation
mat = torch.rand(size=(3,3))
print(torch.matrix_exp(mat))

# element wise multiplication is x*y

# Dot product
print(torch.dot(x,y))

# Batch matrix multiplication (matrix multiplication of last two dimensions)
batch = 32
n = 10
m = 20
p = 5
t1 = torch.rand(size=(batch,n,m))
t2 = torch.rand(size=(batch,m,p))
# print(torch.bmm(t1,t2).shape)

# Broadcasting
x1 = torch.rand(size=(5,5))
x2 = torch.rand(size=(1,5))
# print(x1-x2) # It matches the dimension of the other one
# In case of x1**x2 it will broadcast it first, and then element by element exponentiation will be done

# Some useful operations
sm = torch.sum(x,dim=0) # x is [1 2 3]
# print(sm)
values, indexes = torch.max(x,dim=0)
absl = torch.abs(x) # takes abs for all elements
print(values,indexes) # gives tensor(3) and tensor(2)

# print(torch.argmax(x)) # gives tensor(2)

# We also have torch.mean but the vector needs to be float, so pass x.float
# torch.eq(x,y) compares elementwise
# We can also use torch.sort and sort a particular dimension
print(torch.sort(x,descending=True)) # Returns values and indices eg [3,2,1] and [2,1,0]
print(x)

# We can use torch.clamp
print(torch.clamp(x,min=0)) # values less than 0 will be set to 0