# %%
import datasets

c4 = datasets.load_dataset("allenai/c4", "en", split="train",streaming=True)

data = c4.take(10)
for d in data:
    print(d)
# %%
