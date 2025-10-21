# %%
import llminference.experiments as xp

global_results = []

out = xp.run_one(
    xp.Experiment(
        "test",
        task=xp.Task("wikitext_bpc", shots=0, samples=100, confusion_contexts=0),
        model="EleutherAI/pythia-410m",
        execution=xp.Execution(
            device="cuda:0",
            dtype="float16",
            batch_size=1,
            pipeline_stages=1,
            wandb=False,
        ),
        sparsity=xp.Sparsity(
            "dense",
        ),
    )
)

print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
print("bpc:", out["bpc"])
del out

# %%
