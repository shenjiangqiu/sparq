# %%
dynamic_results = [
    (0.98, 1.84833984375, {"n_selected": 216413538, "total_tokens": 1060050432}),
    (0.96, 1.848388671875, {"n_selected": 150808834, "total_tokens": 1060050432}),
    (0.94, 1.846728515625, {"n_selected": 116208385, "total_tokens": 1060050432}),
    (0.92, 1.850634765625, {"n_selected": 94176696, "total_tokens": 1060050432}),
    (0.9, 1.8541015625, {"n_selected": 78723182, "total_tokens": 1060050432}),
    (0.88, 1.856689453125, {"n_selected": 67080427, "total_tokens": 1060050432}),
    (0.86, 1.859033203125, {"n_selected": 57948232, "total_tokens": 1060050432}),
    (0.84, 1.86494140625, {"n_selected": 50744080, "total_tokens": 1060050432}),
    (0.82, 1.86875, {"n_selected": 44865320, "total_tokens": 1060050432}),
    (0.8, 1.87470703125, {"n_selected": 39854969, "total_tokens": 1060050432}),
    (0.78, 1.875732421875, {"n_selected": 35569635, "total_tokens": 1060050432}),
    (0.76, 1.885888671875, {"n_selected": 31996334, "total_tokens": 1060050432}),
    (0.74, 1.8919921875, {"n_selected": 28859786, "total_tokens": 1060050432}),
    (0.72, 1.9017578125, {"n_selected": 26122896, "total_tokens": 1060050432}),
]

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
