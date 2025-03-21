# %%


from datasets import load_dataset

ds = load_dataset("yahma/alpaca-cleaned")
print(ds["train"])
# %%
train = ds["train"]
display(train[0])


# %%
def compose_text(d):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{d["instruction"]}

### Input:
{d["input"]}

### Response:
{d["output"]}
"""


compose_text(train[0])
# %%
from typing import Any, Dict
def map_dataset(input: Dict[str, Any]) -> Dict[str, Any]:
    text = compose_text(input)
    # split prefill before ### Response:

    prefill_index = text.find("### Response:")
    prefill = text[:prefill_index]
    reference = text[prefill_index:]
    return {"prefill": prefill, "reference": reference}

map_dataset(train[0])

# %%
