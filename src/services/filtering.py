# You must `pip install langdetect dataset` in order to run the following code:
import argparse
import json
import os
from typing import Any, Dict, List

from langdetect import detect

from datasets import load_dataset

DEFAULT_STORE_DIR = "datasets/filtered"


# Define the filter strategy interface
class FilterStrategy:
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class BasicFilterStrategy(FilterStrategy):
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [d for d in data if d["instruction"] and d["output"]]


class LengthFilterStrategy(FilterStrategy):
    def __init__(self, min_chars: int = 10):
        self.min_chars = min_chars

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            d
            for d in data
            if len(d["instruction"]) > self.min_chars
            and len(d["output"]) > self.min_chars
        ]


class LanguageFilterStrategy(FilterStrategy):
    def __init__(self, lang: str):
        self.lang = lang

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_data = []
        for d in data:
            try:
                result = detect(d["instruction"])
                if result == self.lang:
                    filtered_data.append(d)
            except:
                continue
        return filtered_data


class Filter:
    def __init__(self, strategy: FilterStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FilterStrategy):
        self.strategy = strategy

    def filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.strategy.apply(data)

    def save_and_upload(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        push_to_hub: bool = True,
        hf_token: str = None,
    ):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        if push_to_hub:
            dataset = load_dataset("json", data_files=filename)
            hub_name = filename.split("/")[-1].split(".")[0]
            dataset.push_to_hub(hub_name, token=hf_token, private=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument("--filter_lang", type=str, default="en")
    parser.add_argument("--min_chars", type=int, default=10)
    parser.add_argument(
        "--filter_strategies",
        choices=["basic", "length", "language"],
        nargs="+",
        default=["basic", "length", "language"],
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    if args.push_to_hub and args.hf_token is None:
        print("Please provide a HuggingFace API token to push the dataset to the Hub.")
        exit(1)

    file_name = args.file_name
    print(f"Filtering file: {file_name}")
    print(f"Filtering language: {args.filter_lang}")
    print(f"Filtering strategies: {args.filter_strategies}")

    os.makedirs(DEFAULT_STORE_DIR, exist_ok=True)

    dataset = [json.loads(line) for line in open(file_name, "r")]

    filtering_strategies = {
        "basic": BasicFilterStrategy(),
        "length": LengthFilterStrategy(),
        "language": LanguageFilterStrategy(args.filter_lang),
    }

    for strategy in args.filter_strategies:
        print(f"Starting {strategy} filtering...")
        filter = Filter(filtering_strategies[strategy])
        dataset = filter.filter(dataset)
        print(f"Filtered data count: {len(dataset)}")

    # save the filtered data
    final_num_examples = len(dataset)
    if final_num_examples == 0:
        print("No examples left after filtering. Exiting...")
        exit(0)
    else:
        print("Saving filtered data and uploading to the Hub...")
        filtered_file_name = (
            file_name.replace(".json", f"_{final_num_examples}_filtered.json")
            .replace(":", "-")
            .split("/")[-1]
        )
        print(f"Filtered file name vale: {filtered_file_name}")
        filter.save_and_upload(
            dataset,
            f"{DEFAULT_STORE_DIR}/{filtered_file_name}",
            push_to_hub=args.push_to_hub,
            hf_token=args.hf_token,
        )


if __name__ == "__main__":
    main()
