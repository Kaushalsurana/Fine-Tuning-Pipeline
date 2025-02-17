import os
import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get Hugging Face Token
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Use the token while loading models/tokenizers
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

def create_prompt_completion_pairs(text):
    """Splits long text into training prompt-completion pairs."""
    lines = text.split("\n")
    pairs = []
    
    for i in range(len(lines) - 1):
        prompt = lines[i].strip()
        completion = lines[i + 1].strip()
        
        if prompt and completion:
            pairs.append({"prompt": prompt, "completion": completion})
    
    return pairs

def tokenize_function(example, tokenizer):
    """Tokenizes prompt-completion pairs."""
    return tokenizer(
        example["prompt"], 
        text_target=example["completion"], 
        truncation=True, 
        max_length=512
    )

def prepare_dataset(texts):
    """Creates a dataset with train/validation/test splits."""
    all_data = []
    
    for text in texts:
        all_data.extend(create_prompt_completion_pairs(text))
    
    random.shuffle(all_data)
    
    split1 = int(0.8 * len(all_data))
    split2 = int(0.9 * len(all_data))
    
    dataset = DatasetDict({
        "train": Dataset.from_list(all_data[:split1]),
        "validation": Dataset.from_list(all_data[split1:split2]),
        "test": Dataset.from_list(all_data[split2:])
    })
    
    return dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

if __name__ == "__main__":
    # ✅ Load preprocessed text instead of extracting again
    with open("./data/preprocessed_text.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    dataset = prepare_dataset(documents)
    
    # ✅ Save dataset to disk
    os.makedirs("./data", exist_ok=True)
    dataset.save_to_disk("./data/processed_dataset")
    
    print("✅ Dataset saved successfully!")
