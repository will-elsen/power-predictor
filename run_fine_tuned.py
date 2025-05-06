import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import predictor  # Your card utility module

def load_model(model_path="./mtg_rating_model"):
    """
    Load the fine-tuned model and tokenizer from the specified path
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        model: The loaded sequence classification model
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model from {model_path}...")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5,  # 5 classes (ratings 1-5)
        device_map="auto"  # Automatically use GPU if available
    )
    
    
    from peft import PeftModel, PeftConfig

    peft_model = PeftModel.from_pretrained(
        model, model_path
    )
    
    
    
    print("Model loaded successfully!")
    return peft_model, tokenizer

def predict_card_rating(card_name, model, tokenizer, max_length=512):
    """
    Predict the rating for a specific Magic: The Gathering card
    
    Args:
        card_name: Name of the card to rate
        model: The fine-tuned model
        tokenizer: The tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        rating: Predicted rating (1-5)
        confidence: Confidence scores for each class
    """
    # Get card information
    try:
        card_info = predictor.card_utils.get_card_info(card_name)
        card_text = f"Name: {card_info['name']}\nMana Cost: {card_info['mana_cost']}\nTypes: {card_info['types']}\nOracle Text: {card_info['oracle_text']}\nPower/Toughness: {card_info['power']}/{card_info['toughness']}\nLoyalty: {card_info['loyalty']}\n\n."
    except Exception as e:
        print(f"Error fetching card info: {e}")
        return None, None
    
    # Format the prompt as it was during training
    prompt = f"Rate the Magic: The Gathering card '{card_text}' on a scale from 1 to 5 where 1 is irrelevant to the current standard format and 5 is format-warping."
    
    # Tokenize the input
    inputs = tokenizer(
        prompt, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class and confidence scores
    logits = outputs.logits
    print(f"Logits: {logits}")
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Convert back to 1-5 scale (since model was trained on 0-4 labels)
    rating = predicted_class + 1
    
    # Convert probabilities to a regular list
    confidence_scores = probabilities.cpu().numpy().tolist()
    
    return rating, confidence_scores

def main():
    # Load the model
    model, tokenizer = load_model()
    
    # Example cards to predict
    test_cards = [
    ]
    
    for i in range(0, 1):
        card_name = input(f"Enter the name of card {i+1}: ")
        test_cards.append(card_name)
    
    print("\nPredicting card ratings:")
    print("-----------------------")
    
    for card in test_cards:
        rating, confidence = predict_card_rating(card, model, tokenizer)
        
        if rating is not None:
            # Format confidence scores as percentages
            conf_percentages = [f"{conf*100:.1f}%" for conf in confidence]
            
            print(f"Card: {card}")
            print(f"Predicted Rating: {rating}/5")
            print(f"Confidence: {conf_percentages}")
            print("-----------------------")

if __name__ == "__main__":
    main()