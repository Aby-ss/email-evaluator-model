import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from rich.traceback import install
install(show_locals=True)

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

def read_email(file_path):
    """
    Reads the content of a text file.
    
    Args:
    - file_path (str): Path to the text file.
    
    Returns:
    - str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        email_content = file.read()
    return email_content

def preprocess_email(email_content):
    """
    Preprocesses email content by tokenizing, lowercasing, 
    and removing stopwords and non-alphabetic tokens.
    
    Args:
    - email_content (str): The content of the email as a string.
    
    Returns:
    - list: A list of preprocessed tokens.
    """
    # Tokenize the email content
    tokens = word_tokenize(email_content.lower())
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    
    return tokens

def extract_features(email_content):
    """
    Extracts features from the email content.
    
    Args:
    - email_content (str): The content of the email as a string.
    
    Returns:
    - dict: A dictionary containing feature values.
    """
    # Length of the email (in characters and words)
    length_characters = len(email_content)
    length_words = len(word_tokenize(email_content))
    
    # Conciseness (ratio of non-stopwords to total words)
    tokens = preprocess_email(email_content)
    num_non_stopwords = len(tokens)
    num_words = len(word_tokenize(email_content))
    conciseness = num_non_stopwords / num_words if num_words > 0 else 0
    
    # Clear CTA (simple keyword-based approach)
    cta_keywords = ['call to action', 'contact us', 'buy now', 'subscribe', 'learn more', 'get started', 'call', 'discuss']
    cta_present = any(keyword in email_content.lower() for keyword in cta_keywords)
    cta_score = 1 if cta_present else 0
    
    features = {
        'length_characters': length_characters,
        'length_words': length_words,
        'conciseness': conciseness,
        'cta_score': cta_score
    }
    
    return features

def main(file_path):
    """
    Main function to read email, preprocess, and extract features.
    
    Args:
    - file_path (str): Path to the text file.
    """
    # Read the email
    email_content = read_email(file_path)
    print("Original Email Content:\n", email_content)
    
    # Extract features
    features = extract_features(email_content)
    
    # Display extracted features
    print("\nExtracted Features:\n", features)

if __name__ == '__main__':
    # Define the file path
    file_path = '/Users/raoabdul/Documents/GitHub/email-evaluator-model/data/raw/email1.txt'  # Update this path as needed
    
    # Run the main function
    main(file_path)

