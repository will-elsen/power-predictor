import os
import pandas as pd
import numpy as np
import predictor
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from datasets import Dataset

# login to Hugging Face Hub
from huggingface_hub import login
login()

# Configuration
print("configuring model...")
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "./mtg_rating_model"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
EPOCHS = 3
MAX_LENGTH = 512


# Sample data structure (replace with your actual data)
# This is just an example - you would load your actual data
def load_sample_data():
    """Load sample data or replace with your actual data loading code"""
    cards = [
        "Adventuring Gear",
        "Basilisk Collar",
        "Bloodthorn Flail",
        "Carnelian Orb of Dragonkind",
        "Carrot Cake",
        "Cori-Steel Cutter",
        "Gilded Lotus",
        "Golden Argosy",
        "Monument to Endurance",
        "Perilous Snare",
        "Racers' Scoreboard",
        "Rope",
        "Runaway Boulder",
        "Abhorrent Oculus",
        "Ajani's Pridemate",
        "Ash, Party Crasher",
        "Ball Lightning",
        "Beza, the Bounding Spring",
        "Bloodghast",
        "Boulderborn Dragon",
        "Brightblade Stoat",
        "Defiler of Vigor",
        "Diregraf Ghoul",
        "Edgewall Pack",
        "Elvish Archdruid",
        "Essence Channeler",
        "Evolved Sleeper",
        "Fang Guardian",
        "Fangkeeper's Familiar",
        "Friendly Teddy",
        "Fynn, the Fangbearer",
        "Greedy Freebooter",
        "Halo-Charged Skaab",
        "Haughty Djinn",
        "Heartfire Hero",
        "Hinterland Sanctifier",
        "Ingenious Leonin",
        "Iridescent Vinelasher",
        "Jolly Gerbils",
        "Kiora, the Rising Tide",
        "Knight-Errant of Eos",
        "Kraul Whipcracker",
        "Llanowar Elves",
        "Manifold Mouse",
        "Mintstrosity",
        "Nurturing Pixie",
        "Overlord of the Hauntwoods",
        "Overlord of the Boilerbilges",
        "Pride of the Road",
        "Rankle and Torbran",
        "Savage Ventmaw",
        "Savannah Lions",
        "Screaming Nemesis",
        "Severance Priest",
        "Sire of Seven Deaths",
        "Skirmish Rhino",
        "Spiteful Hexmage",
        "Tangled Colony",
        "Up the Beanstalk",
        "Caretaker's Talent",
        "Colossification",
        "Disturbing Mirth",
        "Leyline of Resonance",
        "Lost in the Maze",
        "Nahiri's Resolve",
        "Nowhere to Run",
        "Phyrexian Arena",
        "Tribute to the World Tree",
        "Monstrous Rage",
        "Abrade",
        "Aetherize",
        "Bite Down",
        "Flame Lash",
        "Get Out",
        "Get Lost",
        "This Town Ain't Big Enough",
        "Negate",
        "On the Job",
        "Opt",
        "Rat Out",
        "Refute",
        "Ride's End",
        "Slick Sequence",
        "Steer Clear",
        "Torch the Tower",
        "Abuelo's Awakening",
        "Boltwave",
        "Captain's Call",
        "Deathmark",
        "Excavation Explosion",
        "Exorcise",
        "Feed the Swarm",
        "Jailbreak Scheme",
        "Lunar Insight",
        "Maelstrom Pulse",
        "Pyroclasm",
        "Rankle's Prank",
        "Zombify",
        "Slime Against Humanity",
        "Sunfall"
    ]
    
    # fetch data from scryfall and append to cards_with_data for use in training
    cards_with_data = []
    for card in cards:
        card_info = predictor.card_utils.get_card_info(card)
        with_data = f"Name: {card_info['name']}\nMana Cost: {card_info['mana_cost']}\nTypes: {card_info['types']}\nOracle Text: {card_info['oracle_text']}\nPower/Toughness: {card_info['power']}/{card_info['toughness']}\nLoyalty: {card_info['loyalty']}\n\n."
        cards_with_data.append(with_data)
        
    # Random ratings for the example
    ratings = [1, 2, 1, 2, 3, 4, 2, 2, 4, 3, 1, 2, 1, 4, 2, 2, 2, 4, 3, 1, 2, 3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 3, 2, 3, 4, 3, 1, 3, 1, 3, 4, 2, 4, 3, 2, 3, 3, 4, 1, 2, 2, 2, 3, 2, 2, 3, 2, 2, 4, 1, 3, 4, 1, 1, 3, 3, 3, 5, 2, 3, 2, 1, 3, 3, 5, 3, 1, 3, 2, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 2, 1, 2, 3, 3, 2, 2, 4, 3]
    
    return pd.DataFrame({"card_with_data": cards_with_data, "rating": ratings})

# Format the prompt and completion
def format_examples(df):
    """Format each example as a prompt-completion pair"""
    formatted_data = []
    
    for _, row in df.iterrows():
        # Prompt - asking about a specific card
        prompt = f"Rate the Magic: The Gathering card '{row['card_with_data']}' on a scale from 1 to 5 where 1 is irrelevant to the current standard format and 5 is format-warping.\n\nRating:"
        
        # Completion - just the rating
        completion = f" {int(row['rating'])}"
        
        formatted_data.append({
            "prompt": prompt,
            "completion": completion
        })
    
    return pd.DataFrame(formatted_data)

# Tokenize the data
def tokenize_function(examples, tokenizer):
    """Tokenize and format examples for training"""
    prompt_tokens = tokenizer(examples["prompt"], truncation=True, max_length=MAX_LENGTH)
    completion_tokens = tokenizer(examples["completion"], truncation=True, max_length=20)
    
    # Full sequence = prompt + completion
    input_ids = []
    labels = []
    attention_mask = []
    
    for i in range(len(prompt_tokens["input_ids"])):
        # Combine prompt and completion
        prompt_ids = prompt_tokens["input_ids"][i]
        completion_ids = completion_tokens["input_ids"][i]
        
        # Create full sequence
        sequence_ids = prompt_ids + completion_ids[1:]  # Skip the BOS token
        
        # For labels, we set prompt tokens to -100 (ignored in loss)
        sequence_labels = [-100] * len(prompt_ids) + completion_ids[1:]
        
        # Create attention mask
        sequence_attention = [1] * len(sequence_ids)
        
        input_ids.append(sequence_ids)
        labels.append(sequence_labels)
        attention_mask.append(sequence_attention)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def main():
    # Load and prepare data
    print("Loading sample data...")
    df = load_sample_data()
    
    print(f"Loaded {len(df)} card ratings.")
    
    # Format the examples
    formatted_df = format_examples(df)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(formatted_df, test_size=0.2, random_state=42)
    
    # Create HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load tokenizer and model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_lin", "k_lin", "v_lin", "o_lin"]  # Typical attention modules
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Tokenize datasets
    print("Preparing datasets...")
    
    # Apply tokenization function to datasets
    def tokenize_dataset(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_train = train_dataset.map(
        tokenize_dataset,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = val_dataset.map(
        tokenize_dataset,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        fp16=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")

    # Demonstration of how to use the fine-tuned model
    print("\nExample of using the fine-tuned model:")
    print("---------------------------------------")
    
    # Load the fine-tuned model
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Example card to evaluate
    example_card = "Sol Ring"
    prompt = f"Rate the Magic: The Gathering card '{example_card}' on a scale from 1 to 5 where 1 is very weak and 5 is extremely powerful.\n\nRating:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)
    
    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            num_return_sequences=1
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Card: {example_card}")
    print(f"Generated output: {result}")
    
    # Extract just the rating
    try:
        rating = result.split("Rating:")[-1].strip()
        print(f"Predicted rating: {rating}")
    except:
        print("Couldn't parse rating from output.")

if __name__ == "__main__":
    main()