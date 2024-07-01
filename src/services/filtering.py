# You must `pip install langdetect dataset` in order to run the following code:
import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset
from langdetect import detect


# Define the filter strategy interface
class FilterStrategy:
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class BasicFilterStrategy(FilterStrategy):
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [d for d in data if d["instruction"] and d["output"]]


class LengthFilterStrategy(FilterStrategy):
    def __init__(self, min_chars: int = 10):
        self.num_chars = min_chars

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
        dataset = load_dataset("json", data_files=filename)
        dataset.push_to_hub(filename.split(".")[0], token=hf_token, private=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", type=str, default="dataset_llama3_5000_samples_es.json"
    )
    parser.add_argument("--filter_lang", type=str, default="es")
    parser.add_argument("--min_chars", type=int, default=10)
    parser.add_argument(
        "--filter_strategies",
        choices=["basic", "length", "language"],
        nargs="+",
        default=["basic", "length", "language"],
    )
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    file_name = args.file_name
    print(f"Filtering file: {file_name}")
    print(f"Filtering language: {args.filter_lang}")
    print(f"Filtering strategies: {args.filter_strategies}")

    dataset = []
    with open(file_name, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {file_name} before filtering")

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
    print("Saving filtered data and uploading to the Hub...")
    filtered_file_name = file_name.replace(
        ".json", f"_{final_num_examples}_filtered.json"
    ).replace(":", "-")
    filter.save_and_upload(
        dataset, filtered_file_name, push_to_hub=True, hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
