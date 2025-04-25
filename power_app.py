import streamlit as st
import openai
import re
import predictor
import requests
from PIL import Image
from io import BytesIO

# Set page config
st.set_page_config(page_title="MTG Card Power Rater", page_icon="✨", layout="wide")

# App title and description
st.title("Magic: The Gathering Card Power Rater")
st.markdown("""
This app evaluates Magic: The Gathering cards based on their mana cost, types, and oracle text.
Enter the card name you wish to evaluate to get a response.
""")

# API Key input
api_key = "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"

card_info = None

# Card information form
with st.form("card_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        card_name = st.text_input("Card Name")
        
    if card_name:
        card_info = predictor.card_utils.get_card_info(card_name)
        # print(card_info)
        
    with col2:
        st.subheader("Card Image")
        if card_name:
            card_image_uri = card_info["image_uris"]['png']
            
            
            image = requests.get(card_image_uri)
            image = Image.open(BytesIO(image.content))
            st.image(image, use_container_width=True)
    
    submitted = st.form_submit_button("Analyze Card")


# Function to analyze card
def analyze_card(card_info):
    system_prompt = """
    You are a Magic: The Gathering expert specializing in evaluating cards for Standard format play.
    Analyze the provided card based on its mana cost, types, and oracle text.
    Rate the card on a scale of 1-10 where:
    
    1-3: Weak or very situational cards that rarely see play
    4-5: Average cards that might see some play in specific decks
    6-7: Strong cards that are likely to see regular play in certain archetypes
    8-10: Format-defining cards that are powerful in multiple decks or create new archetypes
    
    Consider factors like:
    - Mana efficiency
    - Card advantage
    - Board impact
    - Synergies with other cards in Standard
    - Flexibility and utility
    
    Your analysis should include:
    1. Power Rating (1-10)
    2. Strengths of the card
    3. Weaknesses or limitations
    4. Types of decks where this card would fit
    5. Comparison to similar cards in Standard (if applicable)
    
    Be specific and refer to current Standard format meta knowledge as of April 2025.
    """
    
    # try:
    #     client = openai.OpenAI(api_key=api_key)
    #     response = client.chat.completions.create(
    #         model="gpt-4-turbo",
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": system_prompt
    #             },
    #             {
    #                 "role": "user",
    #                 "content": f"Please analyze this Magic: The Gathering card:\n\nName: {card_info['name']}\nMana Cost: {card_info['mana_cost']}\nTypes: {card_info['types']}\nOracle Text: {card_info['oracle_text']}\nPower/Toughness: {card_info['power']}/{card_info['toughness']}\nLoyalty: {card_info['loyalty']}\n\nProvide a detailed analysis of its power level in the current Standard format (April 2025)."
    #             }
    #         ],
    #         temperature=0.7,
    #         max_tokens=1000
    #     )
    #     return response.choices[0].message.content
    # except Exception as e:
    #     return f"Error: {str(e)}"
    return "yes"

# Process form submission
if submitted and api_key:
    if not card_name or not card_info:
        st.error("Please provide at least the card name and oracle text.")
    else:
        with st.spinner("Analyzing card..."):
            
            # Get analysis
            analysis = analyze_card(card_info)
            
            # Display results
            st.subheader(f"Analysis for {card_name}")
            st.markdown(analysis)
            
            # Extract rating from analysis (assuming it follows the format)
            try:
                rating_match = re.search(r'Power Rating:?\s*(\d+(?:\.\d+)?)', analysis)
                if rating_match:
                    rating = float(rating_match.group(1))
                    st.progress(rating/10, text=f"Power Level: {rating}/10")
            except:
                pass

elif submitted and not api_key:
    st.error("Please enter your OpenAI API key.")

with st.expander("How the Rating System Works"):
    st.markdown("""
    ### Rating System Explanation
    
    The card analysis uses these criteria:
    
    - **Mana Efficiency**: How much value you get for the mana cost
    - **Card Advantage**: Whether the card replaces itself or generates additional resources
    - **Board Impact**: How the card affects the game state when played
    - **Synergies**: How well it works with other cards in Standard
    - **Flexibility**: How useful the card is in different situations or deck types
    
    The 1-10 scale breaks down as:
    
    - **1-3**: Weak cards that rarely see competitive play
    - **4-5**: Average cards that might see play in specific decks
    - **6-7**: Strong cards that regularly see play in certain archetypes
    - **8-10**: Format-defining cards that shape the meta
    """)

# Add footer
st.markdown("---")
st.markdown("Developed for Magic: The Gathering card analysis. Not affiliated with Wizards of the Coast.")