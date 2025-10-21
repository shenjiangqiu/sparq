# %%
# ...existing code...
import os
import matplotlib.pyplot as plt


dynamic_back = [
    (0.6585542211529136, 1.8501953125),
    (0.7677675198084811, 1.846142578125),
    (0.8143381552418629, 1.847119140625),
    (0.8458122171724369, 1.847607421875),
    (0.945640463863051, 1.85751953125),
    (0.9488175716739701, 1.858740234375),
    (0.951660125895851, 1.86279296875),
    (0.9543265958911487, 1.861328125),
    (0.9566399894927561, 1.86435546875),
    (0.9589472473386382, 1.8650390625)
]
dense = 1.844677734375


fixed_results = [
    (0.98, 1.876708984375),
    (0.96, 1.854443359375),
    (0.94, 1.853857421875),
    (0.92, 1.847216796875),
    (0.9, 1.852099609375),
    (0.88, 1.848876953125),
    (0.86, 1.848388671875),
    (0.84, 1.849755859375),
    (0.82, 1.8458984375),
    (0.8, 1.848828125),
    (0.78, 1.847998046875),
    (0.76, 1.846923828125),
    (0.74, 1.848828125),
    (0.72, 1.846630859375),
]


# %%
def plot_valid_vs_ppl():
    # dynamic: valid rate = n_selected / total_tokens
    dyn_valid = [1 - valid for valid, _ in dynamic_back]
    dyn_ppl = [ppl for _, ppl in dynamic_back]
    fixed_valid = [1 - valid for valid, _ in fixed_results]
    fixed_ppl = [ppl for _, ppl in fixed_results]

    # sparq_valid = [1 - valid for valid, _ in sparq]
    # sparq_ppl = [ppl for _, ppl in sparq]

    plt.figure(figsize=(7, 4.5))
    plt.plot(dyn_valid, dyn_ppl, marker="o", label="dynamic")
    plt.plot(fixed_valid, fixed_ppl, marker="s", label="fixed")
    # plt.plot(sparq_valid, sparq_ppl, marker="^", label="sparq")

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
