import requests
import sys

# William Elsen: 4/23/35. 
# Code written by Claude AI, edited by William Elsen

class card_utils:
    @staticmethod
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
            response = requests.get(url).json()
            
            # Check if the request was successful
            if response:
                card_data = response
                
                # Extract the requested information
                card_info = {
                    'image_uris': card_data.get('image_uris'),
                    'name': card_data.get('name'),
                    'mana_cost': card_data.get('mana_cost'),
                    'types': card_data.get('type_line'),
                    'oracle_text': card_data.get('oracle_text'),
                    'power': card_data.get('power'),
                    'toughness': card_data.get('toughness'),
                    'loyalty': card_data.get('loyalty')
                }
                
                return card_info
            else:
                print(f"Error: Could not find card named '{card_name}'")
                return None
                
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def display_card_info(card_info):
        """
        Display the card information in a readable format
        """
        if card_info:
            print(f"Name: {card_info['name']}")
            print(f"Mana Value: {card_info['mana_cost']}")
            print(f"Types: {card_info['types']}")
            if "Creature" in card_info['types']:
                print(f"Power/Toughness: {card_info['power']}/{card_info['toughness']}")
            print(f"Oracle Text: {card_info['oracle_text']}")