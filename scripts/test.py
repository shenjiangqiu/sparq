# %%
# float 32 vs float 16
import llminference as L
import llminference.experiments as xp
import torch

a = torch.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1, 2, 9]], dtype=torch.float16
)
score = a.softmax(dim=-1)
print(score[0][1] / score[0][0])
print(score[3][1] / score[3][0])
print("score float16:", score)
# %%
soreted_score, idx = score.sort(dim=-1, descending=True)
print("sorted score float16:", soreted_score)
# %%
accumulate_sum = soreted_score.cumsum(dim=-1)
print("accumulate sum float16:", accumulate_sum)
# %%
masked_sum = accumulate_sum >= 0.9
print("masked sum float16:", masked_sum)
# %%
max_idx = masked_sum.long().argmax(dim=-1)
print("max idx float16:", max_idx)
# %%
