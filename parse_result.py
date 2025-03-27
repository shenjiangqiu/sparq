# %%
import json

file = "results_batched.json"
results = None
with open(file) as f:
    results = json.load(f)


# %%
display(results)


# %%
def update_final_results(final_results, result):
    if (
        result["m"],
        result["t"],
        # result["s"],
        result["result"]["task"]["name"],
    ) not in final_results:
        final_results[(result["m"], result["t"], result["result"]["task"]["name"])] = []
    final_results[(result["m"], result["t"], result["result"]["task"]["name"])].append(
        (result["s"], result["result"])
    )


final_results = {}
for result in results:
    update_final_results(final_results, result)
# %%
display(final_results)


# %%
def compose_sparsity(sparsity) -> str:
    name = sparsity["name"]
    if name == "dense":
        return "dense"
    elif name == "ann":
        return "ann"
    elif name == "alisa":
        return f"alisa_{sparsity['last_score']}"
    elif name == "exp":
        return f"exp_{sparsity['valid_bits']}"
    else:
        raise ValueError(f"Unknown sparsity name {name}")


for r in final_results:
    print(r)
    for sparsity,test in final_results[r]:
        # print(test)
        sparsity = compose_sparsity(test["sparsity"])

        if r[1] == "triviaqa":
            score = test["match"]
        elif r[1] == "squad":
            score = test["match"]
        elif r[1] == "cnn_dailymail":
            score = test["rougeL"]
        elif r[1] == "wikitext_bpc":
            score = test["bpc"]
        elif r[1] == "repetition":
            score = test["match_length_char"]
        print("Sparsity: ", sparsity, "Score: ", score)

# %%
