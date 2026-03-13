import random
import json
from datasets import load_dataset
from collections import Counter

def load_data(save_path="data.jsonl"):
    random.seed(67)

    dataset = load_dataset("multi_nli")
    matched_dev = list(dataset["validation_matched"])
    mismatched_dev = list(dataset["validation_mismatched"])

    # Sample 7,500 from each split
    matched_sample = random.sample(matched_dev, 7500)
    mismatched_sample = random.sample(mismatched_dev, 7500)

    # Add split tracking: matched vs mismatched
    for ex in matched_sample:
        ex["source_split"] = "matched"
    for ex in mismatched_sample:
        ex["source_split"] = "mismatched"

    # Combine and shuffle
    data = matched_sample + mismatched_sample
    random.shuffle(data)

    # Save to data.jsonl
    with open(save_path, "w") as f:
        for ex in data:
            json.dump({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"],
                "source_split": ex["source_split"]
            }, f)
            f.write("\n")

    print(f"Total examples: {len(data)}")
    print(f"Label distribution: {Counter(ex['label'] for ex in data)}")

    return data