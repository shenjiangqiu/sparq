# %%
import llminference as L
import llminference.experiments as xp


# %%
out = xp.run_one(
    xp.Experiment(
        "test",
        task=xp.Task("wikitext_bpc", shots=0, samples=1, confusion_contexts=0),
        model="EleutherAI/pythia-410m",
        execution=xp.Execution(
            device="cpu",
            dtype="float32",
            batch_size=1,
            pipeline_stages=1,
            wandb=False,
        ),
        sparsity=xp.Sparsity(
            "alisa",
            k=64,
            local_k=16,
            score="sparse_q",
            reallocate_to_mean_value=True,
            valid_bits=1,
            last_score=2,
            sparsity=0.8,
        ),
    )
)

print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
del out

# %%
