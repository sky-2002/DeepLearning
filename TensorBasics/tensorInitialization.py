import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.tensor takes in a list of rows in the tensor
# we can pass a dtype argument also
# we can also add the type of device
# we can add a require_grad parameter as well
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, device=device, requires_grad=True)

# tensor attributes
print(my_tensor)
# 1 2 3
# 4 5 6 , this is the matrix/tensor we have created
print(my_tensor.dtype)
print(my_tensor.device)  # gives cpu
print(my_tensor.requires_grad)
print(my_tensor.shape)  # row,column

# Other common initialization methods
x = torch.empty(size=(3, 3))  # these values need not be necessarily 0
x = torch.zeros(size=(3, 3))
x = torch.rand(size=(3, 3))  # uniform distribution 0 to 1
x = torch.ones(size=(3, 3))
x = torch.eye(5, 5)  # identity matrix
x = torch.arange(start=0, end=6, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # The data will be from a normal distribution with N(0,1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
x = torch.diag(torch.tensor([[1, 2, 3], [4, 5, 6]]))  # gives 1, 5 ie the main diagonal
# print(x)

# initialize and convert to other types
tensor = torch.arange(4)
print(tensor.dtype)
print(tensor.bool())
print("short", tensor.short())  # int16
print("half", tensor.half())  # float16
print("long", tensor.long())  # int64 (important)
print("float", tensor.float())  # float32 (important)
print("double", tensor.double())  # float64

# Array to tensor conversion
import numpy as np

arr = np.zeros(shape=(5, 5))
tensor = torch.from_numpy(arr)  # Convert to tensor
print(tensor)
arr_again = tensor.numpy()  # Get back the numpy array
print(arr_again)
