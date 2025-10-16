# %%
dynamic_results = [
    (0.98, 1.8017578125, {"n_selected": 8602535, "total_tokens": 39611520}),
    (0.96, 1.7744140625, {"n_selected": 6070678, "total_tokens": 39611520}),
    (0.94, 1.7822265625, {"n_selected": 4770621, "total_tokens": 39611520}),
    (0.92, 1.7880859375, {"n_selected": 3941289, "total_tokens": 39611520}),
]
dynamic = 1.72802734375
dense = 1.7744140625

sparsity = [
    1 - stats["n_selected"] / stats["total_tokens"] for _, _, stats in dynamic_results
]
print(sparsity)

fixed_results = [
    (0.98, 1.90947265625),
    (0.96, 1.869287109375),
    (0.94, 1.85927734375),
    (0.92, 1.85419921875),
    (0.9, 1.85146484375),
    (0.88, 1.848681640625),
    (0.86, 1.851318359375),
    (0.84, 1.84814453125),
    (0.82, 1.84892578125),
    (0.8, 1.84765625),
    (0.78, 1.846484375),
    (0.76, 1.8490234375),
    (0.74, 1.84560546875),
    (0.72, 1.845751953125),
]


sparq = [
    (0.98, 2.077294921875),
    (0.96, 1.996728515625),
    (0.94, 1.962548828125),
    (0.92, 1.945654296875),
    (0.9, 1.930615234375),
    (0.88, 1.915234375),
    (0.86, 1.912548828125),
    (0.84, 1.910498046875),
    (0.82, 1.907177734375),
    (0.8, 1.90654296875),
    (0.78, 1.903369140625),
    (0.76, 1.89404296875),
    (0.74, 1.8998046875),
    (0.72, 1.895849609375),
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

    # add horizontal line for dense ppl
    plt.axhline(
        y=dense, color="k", linestyle="--", linewidth=1.2, label=f"dense = {dense:.6f}"
    )

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
