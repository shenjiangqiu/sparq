# %%
import llminference as L
import llminference.experiments as xp

# float 32 vs float 16

# out = xp.run_one(xp.Experiment(
#     "test",
#     task=xp.Task("wikitext_bpc", shots=0, samples=10,confusion_contexts=0),
#     model="EleutherAI/pythia-410m",
#     execution=xp.Execution(device="cuda", dtype="float32", batch_size=10, pipeline_stages=1, wandb=False),
#     sparsity=xp.Sparsity("dense"),
# ))
# print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
# del out
# %%
out = xp.run_one(
    xp.Experiment(
        "test",
        task=xp.Task("wikitext_bpc", shots=0, samples=10, confusion_contexts=0),
        model="EleutherAI/pythia-410m",
        execution=xp.Execution(
            device="cuda",
            dtype="float32",
            batch_size=10,
            pipeline_stages=1,
            wandb=False,
        ),
        sparsity=xp.Sparsity(
            "exp",
            k=64,
            local_k=16,
            score="sparse_q",
            reallocate_to_mean_value=True,
            valid_bits=1,
        ),
    )
)

print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
del out

# %%
