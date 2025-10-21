# %%
def run_dense_1():
    import llminference.experiments as xp

    global_results = []
    # %%

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
                "dense",
            ),
        )
    )

    # %%
    print({k: v for k, v in out.items() if k not in {"model_config", "results"}})
    print("bpc:", out["bpc"])
    bpc = out["bpc"]
    del out

    # %%
    """
    bpc: 1.7724609375
    """
    return bpc


if __name__ == "__main__":
    run_dense_1()
