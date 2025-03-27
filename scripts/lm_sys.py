# %%
import datasets

data = datasets.load_dataset("lmsys/lmsys-chat-1m")["train"]
# %%
print(data[0])


# %%
def compose_task(data_item):
    """
    Compose a task from a data item by formatting the conversation into a single string.

    Args:
        data_item (dict): A dictionary containing the conversation data.

    Returns:
        str: A formatted string representing the conversation.
    """
    conversation = data_item["conversation"]
    formatted_conversation = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        formatted_conversation += f"###{role}: {content}\n"
    return formatted_conversation


print(compose_task(data[0]))
# %%
