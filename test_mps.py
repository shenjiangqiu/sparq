# %%
import torch

a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)
c = a @ b
print(c)

# %%
a_mps = a.to("mps")
b_mps = b.to("mps")
c_mps = a_mps @ b_mps
print(c_mps)

# %%
