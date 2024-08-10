import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from rich.traceback import install
install(show_locals=True)

# Make sure to download the required NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

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

# Define the file path
file_path = '/Users/raoabdul/Documents/GitHub/email-evaluator-model/data/raw/email1.txt'

# Read the email
email_content = read_email(file_path)
print("Original Email Content:\n", email_content)

# Preprocess the email
preprocessed_tokens = preprocess_email(email_content)
print("\nPreprocessed Email Tokens:\n", preprocessed_tokens)

