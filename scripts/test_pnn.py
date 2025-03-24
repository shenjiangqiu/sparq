# %%
import datasets

data = datasets.load_dataset("ptb-text-only/ptb_text_only",trust_remote_code=True)

print(data)
# %%
for i in range(10):
    print(len(data["train"][i]["sentence"]))
# %%
import llminference

from llminference.tasks.bpc import *

data = PnnTree.data()
print(data)
# %%
