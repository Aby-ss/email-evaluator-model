import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from rich import box
from rich import text
from rich import print

from rich.panel import Panel
from rich.traceback import install
install(show_locals=True)

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

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

def cta_analysis(email_content):
    """
    Analyzes the presence and sentiment of CTAs in the email content.
    
    Args:
    - email_content (str): The content of the email as a string.
    
    Returns:
    - dict: A dictionary containing CTA presence and sentiment score.
    """
    # Expanded CTA keywords list
    cta_keywords = [
        'call to action', 'contact us', 'buy now', 'subscribe', 'learn more', 'get started',
        'sign up', 'register', 'download', 'join', 'click here', 'request a demo', 'schedule a call',
        'get your free', 'take advantage', 'order now', 'reserve your spot', 'get it now', 'limited time offer'
    ]
    
    # Check if any CTA keywords are present
    cta_present = any(keyword in email_content.lower() for keyword in cta_keywords)
    
    # Use NLTK's SentimentIntensityAnalyzer to assess the sentiment of the CTA sentences
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(email_content.lower())
    
    # Filter sentences containing CTA keywords
    cta_sentences = [sent for sent in sentences if any(keyword in sent for keyword in cta_keywords)]
    
    # Calculate average sentiment score for CTA sentences
    if cta_sentences:
        sentiment_scores = [sia.polarity_scores(sent)['compound'] for sent in cta_sentences]
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
    else:
        avg_sentiment_score = 0
    
    return {
        'cta_present': cta_present,
        'cta_sentiment_score': avg_sentiment_score
    }

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
    conciseness = (num_non_stopwords / num_words * 100) if num_words > 0 else 0
    
    # Analyze CTA presence and sentiment
    cta_info = cta_analysis(email_content)
    
    # Assign a CTA score based on presence and sentiment
    cta_score = (1 if cta_info['cta_present'] else 0) + cta_info['cta_sentiment_score']
    
    features = {
        'length_characters': length_characters,
        'length_words': length_words,
        'conciseness_percentage': conciseness,
        'cta_score': cta_score
    }
    
    return features

def process_directory(directory_path):
    """
    Processes all text files in the specified directory and extracts features for each one.
    
    Args:
    - directory_path (str): Path to the directory containing text files.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            print(Panel(f"Processing file: {filename}", border_style="bold", box=box.SQUARE))
            
            # Read the email content
            email_content = read_email(file_path)
            
            # Extract features
            features = extract_features(email_content)
            
            # Display extracted features
            print(Panel(f"Extracted Features:\n{features}", border_style="bold yellow", box=box.SQUARE))

if __name__ == '__main__':
    # Define the directory path
    directory_path = '/Users/raoabdul/Documents/GitHub/email-evaluator-model/data/raw/'  # Update this path as needed
    
    # Process all files in the directory
    process_directory(directory_path)

