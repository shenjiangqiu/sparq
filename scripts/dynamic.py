# %%
import llminference.experiments as xp
import argparse

max_k = None
local_k = None

parser = argparse.ArgumentParser()
parser.add_argument("--max-k", type=int, default=None)
parser.add_argument("--local-k", type=int, default=None)
parser.add_argument("--reallocate", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()

max_k = args.max_k
local_k = args.local_k
reallocate = args.reallocate
gpu = args.gpu
enable_max_k = max_k is not None
enable_local_k = local_k is not None
import sys

print(
    f"max_k: {max_k}, local_k: {local_k}, reallocate: {reallocate}, gpu: {gpu}",
    file=sys.stderr,
)

global_results = []
for sparsity in [i / 1000 for i in range(700, 500, -20)]:
    print("Running dynamic with sparsity:", sparsity)
    global_stats = {"n_selected": 0, "total_tokens": 0}
    out = xp.run_one(
        xp.Experiment(
            "test",
            task=xp.Task("wikitext_bpc", shots=0, samples=100, confusion_contexts=0),
            model="EleutherAI/pythia-410m",
            execution=xp.Execution(
                device="cuda:0" if gpu else "cpu",
                dtype="float16",
                batch_size=1,
                pipeline_stages=1,
                wandb=False,
            ),
            sparsity=xp.Sparsity(
                "dynamic",
                k=64,
                score="sparse_q",
                rank=16,
                enable_vectorized=True,
                reallocate_to_mean_value=reallocate,
                enable_max_k=enable_max_k,
                enable_local_k=enable_local_k,
                max_k=max_k,
                local_k=local_k,
                sparsity=sparsity,
                global_stats=global_stats,
            ),
        )
    )

    print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
    print(global_stats)
    print(
        "real_sparsity:", 1 - global_stats["n_selected"] / global_stats["total_tokens"]
    )
    real_sparsity = 1 - global_stats["n_selected"] / global_stats["total_tokens"]
    global_results.append((real_sparsity, out["bpc"]))
    print(global_results)
    del out

print(global_results)

# %%
