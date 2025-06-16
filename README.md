# 🎯 Formalizer - GPT-2 for Formality Translation

Formalizer is a fine-tuned GPT-2 language model that translates informal text to formal language using few-shot prompting techniques. This project demonstrates how to apply efficient fine-tuning methods to create a specialized text style transfer model.

## 📋 Overview

The Formalizer model transforms casual, conversational text into polished, formal language suitable for professional communication, academic writing, or business correspondence. It leverages few-shot prompting to guide the model in performing style transfer while preserving the original meaning.

### Features

- **Few-Shot Prompting** - Uses carefully selected example pairs to guide the translation
- **LoRA Fine-Tuning** - Efficient parameter-efficient fine-tuning with Low-Rank Adaptation
- **Dynamic Example Selection** - Selects the most relevant examples based on input similarity
- **Comprehensive Evaluation** - Multiple metrics including BLEU, METEOR, and formality indicators

## 🚀 Quick Start

### Setup Environment

```bash
# Create and activate a virtual environment (Windows)
python -m venv formalizer-venv
formalizer-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Translator

```python
# Example code to use the fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

# Load the model and tokenizer
model_path = "./formality_translator_model/best_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load few-shot examples
with open("./formality_translator_model_20250616_173812/few_shot_examples.json", "r") as f:
    few_shot_examples = json.load(f)

# Create a generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Function to translate informal text to formal
def translate_to_formal(informal_text, few_shot_examples):
    # Create prompt with examples
    prompt = create_formality_prompt(few_shot_examples, informal_text)
    
    # Generate formal text
    output = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.4,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    # Extract the formal translation
    generated_text = output[0]["generated_text"]
    formal_part = generated_text.split("Formal:")[-1].strip()
    
    return formal_part

# Example usage
informal_text = "Hey, what's up with that project?"
formal_text = translate_to_formal(informal_text, few_shot_examples)
print(f"Informal: {informal_text}")
print(f"Formal: {formal_text}")
```

## 📊 Dataset

This project uses a curated dataset of informal-formal text pairs (`valentin_dataset.csv`). The dataset is split into training and validation sets (80/20 split).

## 🧠 Model Architecture

- Base model: **GPT-2 Medium** (355M parameters)
- Fine-tuning: **LoRA** (Low-Rank Adaptation)
  - Rank: 32
  - Alpha: 64
  - Target modules: `c_attn`, `c_proj`
  - Dropout: 0.1

## 📈 Evaluation

The model is evaluated using several metrics:
- **BLEU score** - Measures translation quality
- **METEOR score** - Evaluates translation semantics
- **Formality indicators** - Tracks formal expressions added
- **Informality reduction** - Measures reduction in casual language
- **Length ratio** - Analyzes output length compared to input

## 📒 Notebooks

- **formalizer-gpt2.ipynb** - The main notebook for training and evaluating the formality translation model
- **scifi-poetry-gpt2.ipynb** - A complementary notebook exploring other creative applications

## 💼 Project Structure

```
formalizer/
├── formalizer-gpt2.ipynb         # Main training notebook
├── formality_dataset.jsonl       # Dataset in JSONL format
├── valentin_dataset.csv          # Original dataset
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
└── formality_translator_model/   # Fine-tuned model and artifacts
    └── best_model/               # Saved model checkpoint
```

## 📚 Requirements

See `requirements.txt` for the complete list of dependencies.

Main requirements:
- Python 3.8+
- PyTorch
- Transformers
- PEFT
- TRL (Transformer Reinforcement Learning)
- Datasets
- Accelerate
- scikit-learn
- pandas

## 🔮 Future Work

- Web API for online formality translation
- Integration with writing tools and editors
- Multilingual support for formality translation
- Enhanced dynamic prompt selection algorithms

## 📄 License

MIT License

## 🙏 Acknowledgements

This project builds upon several open-source libraries, especially Hugging Face's Transformers and PEFT.

---

Created on June 16, 2025
