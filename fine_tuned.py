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
)
from peft import LoraConfig, get_peft_model, TaskType
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
EPOCHS = 1
MAX_LENGTH = 512

# Sample data structure (replace with your actual data)
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
        
    # Ratings for example (1-5 scale)
    ratings = [1, 2, 1, 2, 3, 4, 2, 2, 4, 3, 1, 2, 1, 4, 2, 2, 2, 4, 3, 1, 2, 3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 3, 2, 3, 4, 3, 1, 3, 1, 3, 4, 2, 4, 3, 2, 3, 3, 4, 1, 2, 2, 2, 3, 2, 2, 3, 2, 2, 4, 1, 3, 4, 1, 1, 3, 3, 3, 5, 2, 3, 2, 1, 3, 3, 5, 3, 1, 3, 2, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 2, 1, 2, 3, 3, 2, 2, 4, 3]
    
    return pd.DataFrame({"card_text": cards_with_data, "rating": ratings})

# Tokenize the data for sequence classification
def tokenize_function(examples, tokenizer):
    """Tokenize examples for sequence classification"""
    # Format the examples as input text
    texts = [
        f"Rate the Magic: The Gathering card '{card_text}' on a scale from 1 to 5 where 1 is irrelevant to the current standard format and 5 is format-warping."
        for card_text in examples["card_text"]
    ]
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        texts, 
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Convert ratings to labels (subtract 1 to make labels 0-4 instead of 1-5)
    tokenized["labels"] = [label - 1 for label in examples["rating"]]
    
    return tokenized

def main():
    # Load and prepare data
    print("Loading sample data...")
    df = load_sample_data()
    
    print(f"Loaded {len(df)} card ratings.")
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load tokenizer and model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with num_labels=5 for the 1-5 rating scale
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,  # 5 classes (ratings 1-5)
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
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
    
    # Process datasets with batched=True for efficiency
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
        fp16=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")
    
main()