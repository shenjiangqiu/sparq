# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Method for evaluating models for bits-per-character.

For example:

    adapter = ...
    prefill_len = 6000
    reference_len = 400
    num_examples = 10

    data = bpc.WikiText.data(prefill_len, reference_len)
    results = list(bpc.evaluate(
        adapter,
        [data[i] for i in range(num_examples)],
        batch_size=5,
    ))
    print(results[0]["bpc"])  # => 1.234
    print(results[9]["bpc"])  # => 5.678
"""
import math
import re
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets
import torch
from torch import Tensor
from tqdm import tqdm

from .. import utility
from ..eval_adapter import Adapter, ModelContext
from ..utility import AnyDict


def compose_text(d):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{d["instruction"]}

### Input:
{d["input"]}

### Response:
{d["output"]}
"""


def map_dataset2(input: Dict[str, Any]) -> Dict[str, Any]:
    text = compose_text(input)
    # split prefill before ### Response:
    # assert False
    prefill_index = text.find("### Response:")
    prefill = text[:prefill_index]
    reference = text[prefill_index : prefill_index + 400]
    # print(len(reference))
    return {"prefill": prefill, "reference": reference}
    # None
    pass


def filter_len_larger_than_6400(example):
    return len(compose_text(example)) > 6400


class Alpaca:
    """Filtered version of wikitext-103-v1 (training set) from HuggingFace
    EleutherAI/wikitext_document_level"""

    @classmethod
    def data(
        cls,
        shuffle_seed: int = 2353669,
        prefill_len: int = 6000,
        reference_len: int = 400,
    ) -> datasets.Dataset:
        ds = datasets.load_dataset("yahma/alpaca-cleaned")["train"]
        # print("mapping!!!------------------")
        ds_mapped = ds.map(
            map_dataset2, batched=False, remove_columns=ds.column_names
        ).filter(lambda x: len(x["prefill"]) >= 300)
        # for i in range(100):
        #     print("------")
        #     print(ds_mapped[i]["reference"])
        return ds_mapped


class WikiText:
    """Filtered version of wikitext-103-v1 (training set) from HuggingFace
    EleutherAI/wikitext_document_level"""

    @staticmethod
    def preprocess(
        d: Dict[str, Any], prefill_len: int, reference_len: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a summarisation example from wikitext-103-v1.

        Examples will be split into "prefill" and "reference" sub-strings, where the
        prefill ends just before the first whitespace character occurring after
        `prefill_len` characters, and the reference string ends just before the first
        whitespace character or end-of-line occurring after `reference_len` characters.
        Should the dataset string not match this format, it will be filtered out
        (i.e. nothing returned here).

        Yields: {"prefill": str, "reference": str}
        """
        # Explanation of regex: https://regex101.com/r/yigOBu/1
        # (note: {{{ in the f-string should be read as { in the regex)
        filter_regex = rf"^(.{{{prefill_len}}}.*?)(\s.{{{reference_len-1}}}.*?)(?:\s|$)"
        m = re.search(filter_regex, d["page"], flags=re.S)
        if m:
            assert len(m.groups()) == 2, (
                "WikiText filter regex should always have 2 groups,"
                f"but has {len(m.groups())} on string '{d['page']}'"
            )
            return dict(prefill=m.group(1), reference=m.group(2))

    @classmethod
    def data(
        cls,
        shuffle_seed: int = 2353669,
        prefill_len: int = 6000,
        reference_len: int = 400,
    ) -> datasets.Dataset:
        return utility.map_and_filter(
            datasets.load_dataset(
                "EleutherAI/wikitext_document_level",
                "wikitext-103-raw-v1",
                trust_remote_code=True,
            )["train"],
            partial(
                cls.preprocess, prefill_len=prefill_len, reference_len=reference_len
            ),
        ).shuffle(shuffle_seed)


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


class LmSysChat:
    """Filtered version of wikitext-103-v1 (training set) from HuggingFace
    EleutherAI/wikitext_document_level"""

    @staticmethod
    def preprocess(
        d: Dict[str, Any], prefill_len: int, reference_len: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a summarisation example from wikitext-103-v1.

        Examples will be split into "prefill" and "reference" sub-strings, where the
        prefill ends just before the first whitespace character occurring after
        `prefill_len` characters, and the reference string ends just before the first
        whitespace character or end-of-line occurring after `reference_len` characters.
        Should the dataset string not match this format, it will be filtered out
        (i.e. nothing returned here).

        Yields: {"prefill": str, "reference": str}
        """

        data = compose_task(d)
        # Explanation of regex: https://regex101.com/r/yigOBu/1
        # (note: {{{ in the f-string should be read as { in the regex)
        filter_regex = rf"^(.{{{prefill_len}}}.*?)(\s.{{{reference_len-1}}}.*?)(?:\s|$)"
        m = re.search(filter_regex, data, flags=re.S)
        if m:
            assert len(m.groups()) == 2, (
                "WikiText filter regex should always have 2 groups,"
                f"but has {len(m.groups())} on string '{data}'"
            )
            return dict(prefill=m.group(1), reference=m.group(2))

    @classmethod
    def data(
        cls,
        shuffle_seed: int = 2353669,
        prefill_len: int = 6000,
        reference_len: int = 400,
    ) -> datasets.Dataset:
        return utility.map_and_filter(
            datasets.load_dataset(
                "lmsys/lmsys-chat-1m",
                trust_remote_code=True,
            )[
                "train"
            ].select(range(4000)),
            partial(
                cls.preprocess, prefill_len=prefill_len, reference_len=reference_len
            ),
        ).shuffle(shuffle_seed)


class PnnTree:
    """Filtered version of wikitext-103-v1 (training set) from HuggingFace
    EleutherAI/wikitext_document_level"""

    @staticmethod
    def preprocess(
        d: Dict[str, Any], prefill_len: int, reference_len: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a summarisation example from wikitext-103-v1.

        Examples will be split into "prefill" and "reference" sub-strings, where the
        prefill ends just before the first whitespace character occurring after
        `prefill_len` characters, and the reference string ends just before the first
        whitespace character or end-of-line occurring after `reference_len` characters.
        Should the dataset string not match this format, it will be filtered out
        (i.e. nothing returned here).

        Yields: {"prefill": str, "reference": str}
        """
        # Explanation of regex: https://regex101.com/r/yigOBu/1
        # (note: {{{ in the f-string should be read as { in the regex)
        filter_regex = rf"^(.{{{prefill_len}}}.*?)(\s.{{{reference_len-1}}}.*?)(?:\s|$)"
        m = re.search(filter_regex, d["sentence"], flags=re.S)
        if m:
            assert len(m.groups()) == 2, (
                "WikiText filter regex should always have 2 groups,"
                f"but has {len(m.groups())} on string '{d['page']}'"
            )
            return dict(prefill=m.group(1), reference=m.group(2))

    @classmethod
    def data(
        cls,
        shuffle_seed: int = 2353669,
        prefill_len: int = 6000,
        reference_len: int = 400,
    ) -> datasets.Dataset:
        return utility.map_and_filter(
            datasets.load_dataset(
                "ptb-text-only/ptb_text_only",
                trust_remote_code=True,
            )["train"],
            partial(
                cls.preprocess, prefill_len=prefill_len, reference_len=reference_len
            ),
        ).shuffle(shuffle_seed)


def calc_bpc(nll: Tensor, chars_per_seq: Tensor) -> Tensor:
    """Calculates the bits-per-character of a tensor of negative log likelihoods.

    Expects a tensor of shape (..., sequence_length), and reduces across the
    sequence dimension.
    """
    nll = nll * math.log2(math.e)  # convert from base 'e' to base 2
    return nll.sum(-1) / chars_per_seq


def calculate_perlexity(nll: Tensor, chars_per_seq: Tensor) -> Tensor:
    """Calculates the bits-per-character of a tensor of negative log likelihoods.

    Expects a tensor of shape (..., sequence_length), and reduces across the
    sequence dimension.
    """

    return torch.exp(nll.sum(-1) / chars_per_seq)


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    max_reference_tokens: int = 256,
    generation_context: Optional[ModelContext] = None,
    progress: Union[bool, str] = True,
) -> Iterable[AnyDict]:
    """Evaluate bits-per-character with respect to a list of example sequences.

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model.
        examples (List[AnyDict]): Summarisation dataset containing:
        {prefill: str, reference: str}.
        batch_size (int): Batch examples for greedy sample steps.
        max_reference_tokens (int): Maximium number of generated reference tokens.
        generation_context (ModelContext, optional): Override model
        during generation.
        progress (bool|str, optional): enable tqdm (True) and set description
        (str), or disable (False).

    Yields:
        {bpc: int, prefill_length: int, reference_length: int}: For each input.
    """
    for batch in tqdm(
        list(utility.batches(examples, batch_size)),
        desc=(
            progress
            if isinstance(progress, str)
            else f"Evaluating {adapter.model.name_or_path}"
        ),
        disable=progress is False,
    ):
        prefill = [b["prefill"] for b in batch]
        reference = [b["reference"] for b in batch]

        prefill_lengths = [len(adapter.tok_encode(p)) for p in prefill]
        reference_lengths = [len(adapter.tok_encode(r)) for r in reference]
        reference_char_lengths = torch.tensor([len(r) for r in reference])

        nll_batch = adapter.forced_sample(
            prefill,
            reference,
            generation_context=generation_context,
            max_reference_tokens=max_reference_tokens,
        )
        bpcs = calc_bpc(nll_batch, reference_char_lengths)
        for p, pr_len, g_len in zip(bpcs, prefill_lengths, reference_lengths):
            yield dict(
                bpc=p.item(),
                prefill_length=pr_len,
                reference_length=g_len,
            )
