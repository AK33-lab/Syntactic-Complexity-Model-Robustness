import json
import os
import random
from datasets import load_dataset
from collections import Counter

DEFAULT_PERTURBATIONS = ("adjrel", "passive", "pp")


def _read_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _has_expanded_format(records):
    if not records:
        return False
    required = {"premise", "hypothesis", "label", "source_split", "perturbation_method"}
    return required.issubset(records[0].keys())


def _build_base_data(sample_per_split=7500):
    random.seed(67)

    dataset = load_dataset("multi_nli")
    matched_dev = list(dataset["validation_matched"])
    mismatched_dev = list(dataset["validation_mismatched"])

    matched_sample = random.sample(matched_dev, sample_per_split)
    mismatched_sample = random.sample(mismatched_dev, sample_per_split)

    data = []
    for idx, ex in enumerate(matched_sample):
        data.append(
            {
                "example_id": idx,
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"],
                "source_split": "matched",
            }
        )

    start_idx = len(data)
    for offset, ex in enumerate(mismatched_sample):
        data.append(
            {
                "example_id": start_idx + offset,
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"],
                "source_split": "mismatched",
            }
        )

    random.shuffle(data)
    return data


def _expand_with_perturbations(base_data, perturbations):
    from perturbations import (
        adj_to_relative_clause,
        passive_voice,
        prepositional_phrase_insertion,
    )

    perturb_map = {
        "adjrel": adj_to_relative_clause,
        "passive": passive_voice,
        "pp": prepositional_phrase_insertion,
    }

    expanded = []
    ordered_perturbations = ["original"] + list(dict.fromkeys(perturbations))

    for ex in base_data:
        for perturbation_method in ordered_perturbations:
            new_ex = dict(ex)
            if perturbation_method == "original":
                new_ex["premise"] = ex["premise"]
            else:
                new_ex["premise"] = perturb_map[perturbation_method](ex["premise"])
            new_ex["perturbation_method"] = perturbation_method
            expanded.append(new_ex)

    return expanded


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for ex in records:
            json.dump(ex, f)
            f.write("\n")


def load_data(
    save_path="data.jsonl",
    rebuild=False,
    perturbations=DEFAULT_PERTURBATIONS,
    sample_per_split=7500,
):
    requested_methods = {"original", *list(dict.fromkeys(perturbations))}

    if os.path.exists(save_path) and not rebuild:
        cached_data = _read_jsonl(save_path)
        if cached_data and _has_expanded_format(cached_data):
            cached_methods = {ex["perturbation_method"] for ex in cached_data}
            if requested_methods.issubset(cached_methods):
                print(f"Loaded expanded data from {save_path}: {len(cached_data)} examples")
                print(f"Perturbation counts: {Counter(ex['perturbation_method'] for ex in cached_data)}")
                print(f"Label distribution: {Counter(ex['label'] for ex in cached_data)}")
                return cached_data

        print("Rebuilding data.jsonl with expanded perturbation rows...")

    base_data = _build_base_data(sample_per_split=sample_per_split)
    data = _expand_with_perturbations(base_data, perturbations)

    _write_jsonl(save_path, data)

    print(f"Saved expanded data to {save_path}")
    print(f"Total examples: {len(data)}")
    print(f"Perturbation counts: {Counter(ex['perturbation_method'] for ex in data)}")
    print(f"Label distribution: {Counter(ex['label'] for ex in data)}")

    return data