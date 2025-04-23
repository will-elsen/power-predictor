import requests
import sys

def get_card_info(card_name):
    """
    Fetch Magic: The Gathering card information from Scryfall API
    
    Args:
        card_name: Name of the MTG card to search for
        
    Returns:
        Dictionary with card information or None if card not found
    """
    # URL encode the card name for the API request
    url = f"https://api.scryfall.com/cards/named?exact={card_name}"
    
    try:
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            card_data = response.json()
            
            # Extract the requested information
            card_info = {
                'name': card_data.get('name'),
                'mana_cost': card_data.get('mana_cost'),
                'types': card_data.get('type_line'),
                'oracle_text': card_data.get('oracle_text')
            }
            
            return card_info
        else:
            print(f"Error: Could not find card named '{card_name}'")
            return None
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def display_card_info(card_info):
    """
    Display the card information in a readable format
    """
    if card_info:
        print(f"Name: {card_info['name']}")
        print(f"Mana Value: {card_info['mana_cost'].replace('{', '').replace('}', '')}")
        print(f"Types: {card_info['types']}")
        print(f"Oracle Text: {card_info['oracle_text']}")

def main():
    # Get card name from command line or prompt user
    if len(sys.argv) > 1:
        card_name = ' '.join(sys.argv[1:])
    else:
        card_name = input("Enter card name: ")
    
    # Get and display card information
    card_info = get_card_info(card_name)
    if card_info:
        display_card_info(card_info)

if __name__ == "__main__":
    main()