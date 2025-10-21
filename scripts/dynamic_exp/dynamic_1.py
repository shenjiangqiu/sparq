# %%
def run_dynamic_1(sparsity):
    import llminference.experiments as xp

    global_results = []

    global_stats = {"n_selected": 0, "total_tokens": 0}
    out = xp.run_one(
        xp.Experiment(
            "test",
            task=xp.Task("wikitext_bpc", shots=0, samples=1, confusion_contexts=0),
            model="EleutherAI/pythia-410m",
            execution=xp.Execution(
                device="cuda:0",
                dtype="float16",
                batch_size=1,
                pipeline_stages=1,
                wandb=False,
            ),
            sparsity=xp.Sparsity(
                "dynamic",
                k=64,
                local_k=16,
                score="sparse_q",
                rank=16,
                reallocate_to_mean_value=True,
                sparsity=sparsity,
                global_stats=global_stats,
            ),
        )
    )

    print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
    print(global_stats)
    # %%
    print(
        "real_sparsity:", 1 - global_stats["n_selected"] / global_stats["total_tokens"]
    )
    real_sparsity = 1 - global_stats["n_selected"] / global_stats["total_tokens"]
    print("bpc:", out["bpc"])
    # global_results.append((sparsity, out["bpc"], global_stats))
    bpc = out["bpc"]
    # %%
    print(global_results)
    """

    real_sparsity: 0.6388994666198116
    bpc: 1.794921875

    """
    del out
    return bpc, real_sparsity


if __name__ == "__main__":
    run_dynamic_1()
