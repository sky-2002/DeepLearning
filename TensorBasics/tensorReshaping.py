import torch

x = torch.arange(9)

print(x.view(3,3)) # view acts on contiguous tensors ie stored in memory contiguously
print(x.reshape(3,3)) # It does not matter if it is not contiguous

print(x.reshape(3,3).t())
print(x.reshape(3,3).t().reshape(1,9))

# using view on the viewed and transposed x will result in error
# print(x.view(3,3).t().view(9))
# ---------------------------------------------------------------------------------
# ERROR: RuntimeError: view size is not compatible with input tensor's
# size and stride (at least one dimension spans across two contiguous subspaces).
# Use .reshape(...) instead.
# ---------------------------------------------------------------------------------
# To avoid the above error , we can use .contiguous() method
y = x.view(3,3).t()
print(y.contiguous().view(9)) # This works fine

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.concat((x1,x2),dim=0)) # this one is by row, 4x5
print(torch.concat((x1,x2),dim=1)) # this one is by column, 2x10

print("Unrolling x1:",x1.view(-1)) # Unrolls/Flattens x1

batch = 64
z = torch.rand((batch,2,5))
# print(z.view(batch,-1)) # 64 rows with 10 columns each

z = z.permute(0,2,1)
# print(z.shape) # gives 64 5 2

x = torch.arange(10)
print(x.unsqueeze(0).shape) # gives 1 10
print(x.unsqueeze(1).shape) # gives 10 1

x = torch.rand(3)
print(x)
print(x.unsqueeze(0))
print(x.unsqueeze(1))
print(x.unsqueeze(0).unsqueeze(1))
