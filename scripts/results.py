# %%
# ...existing code...
import os
import matplotlib.pyplot as plt

dynamic_new_only_sparsity = [
    (0.9443807817127845, 1.882099609375),
    (0.9476726836110394, 1.883291015625),
    (0.9505444596359117, 1.8828125),
    (0.9532419783567039, 1.885224609375),
    (0.9556295081553889, 1.88671875),
    (0.9579692094868152, 1.887138671875),
    (0.9600042525585346, 1.888720703125),
    (0.9619446022163406, 1.890986328125),
    (0.9636903734909279, 1.891474609375),
    (0.9654392139161787, 1.89783203125),
    (0.9669818605229863, 1.896806640625),
    (0.968402716079728, 1.900048828125),
    (0.9698357837998478, 1.9015234375),
    (0.9710976631619818, 1.90546875),
    (0.9723512591108872, 1.906796875),
]
dynamic_old_only_sparsity = [
    (0.9443432015452518, 1.882216796875),
    (0.9476428787070152, 1.881474609375),
    (0.950495663556666, 1.882841796875),
    (0.9532145249748337, 1.885517578125),
    (0.9556142172265982, 1.886904296875),
    (0.9579396703180915, 1.88826171875),
    (0.9599774709816888, 1.889345703125),
    (0.9619533119516369, 1.89203125),
    (0.9637328492023205, 1.89154296875),
    (0.9654838046678106, 1.8947265625),
    (0.9670041845384805, 1.896513671875),
    (0.9684267875247342, 1.9004296875),
    (0.9698389173897874, 1.9020703125),
    (0.9711042631758362, 1.90375),
    (0.9723656286603145, 1.90646484375),
]

dynamic_new_withmean = [
    (0.9480591390421096, 1.885556640625),
    (0.9513739442956359, 1.88599609375),
    (0.9542218880168156, 1.888486328125),
    (0.9569403360007708, 1.890556640625),
    (0.9593322726276632, 1.892353515625),
    (0.961644584402788, 1.895419921875),
    (0.9636575051641758, 1.898369140625),
    (0.9656235439996712, 1.900810546875),
    (0.9673274274945467, 1.9042578125),
    (0.9690348897217896, 1.90685546875),
    (0.9705557096206591, 1.911376953125),
    (0.9719566961960406, 1.91587890625),
    (0.9733259925582922, 1.919365234375),
    (0.9745823332112203, 1.92392578125),
    (0.9758002758292322, 1.92798828125),
]


dynamic_old_sparsity = [
    (0.9443432015452518, 1.882216796875),
    (0.9476428787070152, 1.881474609375),
    (0.950495663556666, 1.882841796875),
    (0.9532145249748337, 1.885517578125),
    (0.9556142172265982, 1.886904296875),
    (0.9579396703180915, 1.88826171875),
    (0.9599774709816888, 1.889345703125),
    (0.9619533119516369, 1.89203125),
    (0.9637328492023205, 1.89154296875),
    (0.9654838046678106, 1.8947265625),
    (0.9670041845384805, 1.896513671875),
    (0.9684267875247342, 1.9004296875),
    (0.9698389173897874, 1.9020703125),
    (0.9711042631758362, 1.90375),
    (0.9723656286603145, 1.90646484375),
]
dense = 1.872275390625


fixed_results = [
    (0.98, 1.92162109375),
    (0.96, 1.89328125),
    (0.94, 1.884169921875),
    (0.92, 1.879306640625),
    (0.9, 1.8786328125),
    (0.88, 1.8762109375),
    (0.86, 1.876337890625),
    (0.84, 1.875078125),
    (0.82, 1.8740625),
    (0.8, 1.873544921875),
    (0.78, 1.87365234375),
    (0.76, 1.874453125),
    (0.74, 1.874423828125),
    (0.72, 1.874013671875),
]


# %%
def plot_valid_vs_ppl():
    # dynamic: valid rate = n_selected / total_tokens
    # 老的实现，没有mean
    dyn_old_sparsity_valid = [1 - valid for valid, _ in dynamic_old_sparsity]
    dyn_old_sparsity_ppl = [2**bpc for _, bpc in dynamic_old_sparsity]
    # 再次验证
    dynamic_old_only_sparsity_valid = [
        1 - valid for valid, _ in dynamic_old_only_sparsity
    ]
    dynamic_old_only_sparsity_ppl = [2**bpc for _, bpc in dynamic_old_only_sparsity]
    # 新的没有mean
    dynamic_new_only_sparsity_valid = [
        1 - valid for valid, _ in dynamic_new_only_sparsity
    ]
    dynamic_new_only_sparsity_ppl = [2**bpc for _, bpc in dynamic_new_only_sparsity]
    # 新的加上 mean
    dynamic_new_withmean_valid = [1 - valid for valid, _ in dynamic_new_withmean]
    dynamic_new_withmean_ppl = [2**bpc for _, bpc in dynamic_new_withmean]

    fixed_valid = [1 - valid for valid, _ in fixed_results]
    fixed_ppl = [2**bpc for _, bpc in fixed_results]

    # sparq_valid = [1 - valid for valid, _ in sparq]
    # sparq_ppl = [ppl for _, ppl in sparq]

    plt.figure(figsize=(7, 4.5))

    plt.plot(
        dyn_old_sparsity_valid,
        dyn_old_sparsity_ppl,
        marker="o",
        label="old impl without mean",
    )
    plt.plot(fixed_valid, fixed_ppl, marker="s", label="fixed")
    plt.plot(
        dynamic_old_only_sparsity_valid,
        dynamic_old_only_sparsity_ppl,
        marker="^",
        label="old impl without mean(validated)",
    )
    plt.plot(
        dynamic_new_only_sparsity_valid,
        dynamic_new_only_sparsity_ppl,
        marker="v",
        label="new impl without mean",
    )
    plt.plot(
        dynamic_new_withmean_valid,
        dynamic_new_withmean_ppl,
        marker="D",
        label="new impl with mean",
    )
    # plt.plot(sparq_valid, sparq_ppl, marker="^", label="sparq")
    dense_ppl = 2**dense
    # add horizontal line for dense ppl
    plt.axhline(
        y=dense_ppl,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label=f"dense = {dense:.6f}",
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
