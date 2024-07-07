# Make sure have your ollama server runing
# and pip install tqdm datasets

import argparse
import json
import os
import urllib.request

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_STORE_DIR = "datasets/raw"

# ollama default URL
URL = "http://localhost:11434/api/chat"

query_templates = {
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
    "phi3": "<s><|user|>",  # phi3:mini
    "phi3:medium": "<s><|user|>",  # phi3:medium
}

# add more languages as needed
lang_dict = {"en": "", "es": "spanish", "de": "deutsch"}


def make_query_template(model, lang):
    return f"{query_templates[model]}{lang_dict[lang]}:"


def query_model(prompt, model, url=URL, role="user"):
    data = {
        "model": model,
        "seed": 676,
        "temperature": 1.0,
        "top_p": 1,
        "messages": [{"role": role, "content": prompt}],
    }
    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Print each generated sample",
    )
    parser.add_argument(
        "--lang",
        choices=[lang for lang in lang_dict.keys()],
        default="en",
        help="Language to generate samples in",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push the dataset to the HuggingFace Hub",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for pushing the dataset to the Hub",
    )
    args = parser.parse_args()

    if args.push_to_hub and args.hf_token is None:
        print("Please provide a HuggingFace API token to push the dataset to the Hub.")
        exit(1)

    os.makedirs(DEFAULT_STORE_DIR, exist_ok=True)

    output_file_name = (
        f"{DEFAULT_STORE_DIR}/{args.model}_{args.num_samples}_samples_{args.lang}.json"
    )

    print("Creating dataset with the following parameters:")
    print(f"MODEL: {args.model}")
    print(f"Total Samples: {args.num_samples}")
    print(f"Language: {args.lang}")
    print(f"Verbose Mode: {args.display}")
    print(f"Output file: {output_file_name}")
    query_template = make_query_template(args.model, args.lang)
    print(f"Query Template: {query_template}")

    with open(output_file_name, "a") as f:  # Open file in append mode from the start
        for i in tqdm(range(args.num_samples), desc="Generating Samples"):
            result = query_model(
                query_template,
                model=args.model,
                role="assistant",
            )
            instruction = extract_instruction(result)
            response = query_model(instruction, model=args.model, role="user")
            entry = {
                "instruction": instruction,
                "output": response,
                "model": args.model,
            }
            json.dump(entry, f)
            f.write("\n")  # Newline to separate entries

            if args.display:
                print(f"Sample {i+1}")
                print(f"Instruction: {instruction}")
                print(f"Response: {response[:100]}\n")

    if args.push_to_hub:
        dataset = load_dataset("json", data_files=output_file_name)
        hub_name = output_file_name.split("/")[-1].split(".")[0]
        dataset.push_to_hub(hub_name, token=args.hf_token, private=True)


if __name__ == "__main__":
    main()
