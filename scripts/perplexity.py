# %%
import torch
from torcheval.metrics.text import Perplexity

# %%
metric = Perplexity()
input = torch.tensor(
    [
        [[0.3659, 0.7025, 0.3104]],
        [[0.0097, 0.6577, 0.1947]],
        [[0.5659, 0.0025, 0.0104]],
        [[0.9097, 0.0577, 0.7947]],
    ]
)
target = torch.tensor([[2], [1], [2], [1]])
metric.update(input, target)
metric.compute()

# %%
metric = Perplexity()
input = torch.tensor(
    [
        [
            [0.7025, 0.3659, 0.4104],
            [0.0097, 0.6577, 0.1947],
            [0.5659, 0.0025, 0.0104],
            [0.9097, 0.0577, 0.7947],
            [0.9097, 0.0577, 0.7947],
            [0.9097, 0.0577, 0.7947],
            [0.9097, 0.0577, 0.7947],
            [0.9097, 0.0577, 0.7947],
        ]
    ]
)
target = torch.tensor([[2, 1, 2, 1, 1, 1, 1, 1]])
metric.update(input, target)
metric.compute()

# %%
