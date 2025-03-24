
# float 32 vs float 16
import llminference as L
import llminference.experiments as xp
import torch
torch.set_num_threads(32)
out = xp.run_one(xp.Experiment(
    "test",
    task=xp.Task("wikitext_bpc", shots=0, samples=10,confusion_contexts=0),
    model="EleutherAI/pythia-410m",
    execution=xp.Execution(device="cuda", dtype="float32", batch_size=10, pipeline_stages=1, wandb=False),
    sparsity=xp.Sparsity("dense"),
))
display({k: v for k, v in out.items() if k not in {"model_config", "results"}})
del out

out = xp.run_one(xp.Experiment(
    "test",
    task=xp.Task("wikitext_bpc", shots=0, samples=10,confusion_contexts=0),
    model="EleutherAI/pythia-410m",
    execution=xp.Execution(device="cuda", dtype="float16", batch_size=10, pipeline_stages=1, wandb=False),
    sparsity=xp.Sparsity("dense"),
))
display({k: v for k, v in out.items() if k not in {"model_config", "results"}})
del out