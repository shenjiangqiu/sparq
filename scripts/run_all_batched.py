# %%
import llminference as L
import llminference.experiments as xp
import torch

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
TASKS = [
    "triviaqa",
    "squad",
    "cnn_dailymail",
    "wikitext_bpc",
    "repetition",
]
models = ["EleutherAI/pythia-410m"]
sparsity_num = [0.9, 0.8, 0.6, 0.4]
sparsity = [
    xp.Sparsity("dense"),
    xp.Sparsity(
        "ann",
        k=64,
        local_k=16,
        score="sparse_q",
        rank=16,
        reallocate_to_mean_value=True,
        sparsity=0.8,
    ),
]
for last_score in [2, 4]:
    sparsity.append(
        xp.Sparsity(
            "alisa",
            k=64,
            local_k=16,
            score="sparse_q",
            reallocate_to_mean_value=False,
            valid_bits=1,
            last_score=last_score,
            sparsity=0.8,
        )
    )
for valid_bits in [ 2,3,  4 ]:
    sparsity.append(
        xp.Sparsity(
            "exp",
            k=64,
            local_k=16,
            score="sparse_q",
            reallocate_to_mean_value=False,
            valid_bits=valid_bits,
            sparsity=0.8,
        )
    )
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "float16" if device == "cuda" else "float32"
print("running on" ,device)
results = []
for t in TASKS:
    for m in models:
        for s_num in sparsity_num:
            for s in sparsity:
                if hasattr(s, "sparsity"):
                    s.sparsity = s_num
                out = xp.run_one(
                    xp.Experiment(
                        "test",
                        task=xp.Task(t, shots=0, samples=100, confusion_contexts=0),
                        model=m,
                        execution=xp.Execution(
                            device=device,
                            dtype=dtype,
                            batch_size=20,
                            pipeline_stages=1,
                            wandb=False,
                        ),
                        sparsity=s,
                    )
                )

                result = {
                    k: v for k, v in out.items() if k not in {"model_config", "results"}
                }
                del out
                results.append({"m": m, "t": t, "s": s_num, "result": result})
                import json
                with open("results_batched.json","w") as f:
                    json.dump(results,f)

                    


# save results
import json

with open("results_batched.json", "w") as f:
    json.dump(results, f)
# %%
