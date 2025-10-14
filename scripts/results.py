# %%
dynamic_results = [
    (0.7, 1.8564453125, {"n_selected": 990967, "total_tokens": 39611520}),
    (0.75, 1.8173828125, {"n_selected": 1271452, "total_tokens": 39611520}),
    (0.8, 1.8359375, {"n_selected": 1660428, "total_tokens": 39611520}),
    (0.85, 1.82421875, {"n_selected": 2229373, "total_tokens": 39611520}),
    (0.9, 1.798828125, {"n_selected": 3193979, "total_tokens": 39611520}),
    (0.95, 1.7978515625, {"n_selected": 5206437, "total_tokens": 39611520}),
]
sparsity = [
    1 - stats["n_selected"] / stats["total_tokens"] for _, _, stats in dynamic_results
]
print(sparsity)

fixed_results = [
    (0.974, 1.82421875),
    (0.9679019638731359, 1.8203125),
    (0.9580821942707576, 1.8125),
    (0.943719, 1.7958984375),
    (0.919, 1.7958984375),
    (0.868562, 1.8017578125),
]

sparq = [
    (0.974, 1.9765625),
    (0.9679019638731359, 1.9775390625),
    (0.9580821942707576, 1.9794921875),
    (0.943719, 1.9248046875),
    (0.919, 1.9482421875),
    (0.868562, 1.90625),
]
# ...existing code...
import os
import matplotlib.pyplot as plt


# %%
def plot_valid_vs_ppl():
    # dynamic: valid rate = n_selected / total_tokens
    dyn_valid = [
        stats["n_selected"] / stats["total_tokens"] for _, _, stats in dynamic_results
    ]
    dyn_ppl = [ppl for _, ppl, _ in dynamic_results]

    fixed_valid = [1 - valid for valid, _ in fixed_results]
    fixed_ppl = [ppl for _, ppl in fixed_results]

    sparq_valid = [1 - valid for valid, _ in sparq]
    sparq_ppl = [ppl for _, ppl in sparq]

    plt.figure(figsize=(7, 4.5))
    plt.plot(dyn_valid, dyn_ppl, marker="o", label="dynamic")
    plt.plot(fixed_valid, fixed_ppl, marker="s", label="fixed")
    plt.plot(sparq_valid, sparq_ppl, marker="^", label="sparq")

    plt.xlabel("valid rate (1 - sparsity)")
    plt.ylabel("ppl")
    plt.title("Valid rate vs PPL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "valid_vs_ppl.png")
    plt.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_valid_vs_ppl()

# %%
