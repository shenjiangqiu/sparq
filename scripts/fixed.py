# %%
import llminference.experiments as xp

global_results = []
# %%
for sparsity in [i / 100 for i in range(98, 70, -2)]:
    out = xp.run_one(
        xp.Experiment(
            "test",
            task=xp.Task("wikitext_bpc", shots=0, samples=20, confusion_contexts=0),
            model="EleutherAI/pythia-410m",
            execution=xp.Execution(
                device="mps",
                dtype="float16",
                batch_size=2,
                pipeline_stages=1,
                wandb=False,
            ),
            sparsity=xp.Sparsity(
                "fixed",
                k=64,
                local_k=16,
                score="sparse_q",
                rank=16,
                reallocate_to_mean_value=True,
                sparsity=sparsity,
            ),
        )
    )

    print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
    global_results.append((sparsity, out["bpc"]))
    del out

# %%
print(global_results)
