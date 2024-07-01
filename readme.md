# 📚 Synthetic Instruction Dataset Generation
This repo will allow you to create *multilingual* synthetic instructions datasets using the [MAGPIE](https://arxiv.org/abs/2406.08464) method and `ollama`.

**⚠️Important Note:** The instruction datasets created here are for educational purposes. However, it is the users' duty to ensure that their use adheres to the terms of the relevant licensing agreements with Meta AI's Llama 3.

## 🔧 Prerequisites
- Git
- Python 3.8 or higher
- Poetry
- ollama


## 🛠️ Installation
1. Clone this repo
```bash
git clone https://mrm8488/synthetic-instructions-dataset-generation
```

2. Install the requirements
```bash
poetry install
```

3. Download the `ollama` model
```bash
ollama run llama3
```

4. Create a server with the `ollama` model
```bash
ollama server llama3
```


## 🚀 Example of usage
```bash
python src/dataset_gen.py --model llama3 --lang es --num_samples 1000 --push_to_hub --hf_token <YOUR_HUGGINGFACE_TOKEN>
```


## 🔍 Filtering the generated dataset
```bash
python src/services/filtering.py --filter_lang es --push_to_hub --hf_token <YOUR_HUGGINGFACE_TOKEN>
```

## License
```text
MIT License
````

Contributions are welcome!  🎉

### Acknowledgements
Sebastian Raschka, PhD for his post and base script:

https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/05_dataset-generation/llama3-ollama.ipynb
https://www.linkedin.com/feed/update/urn:li:activity:7210982019751661568/
