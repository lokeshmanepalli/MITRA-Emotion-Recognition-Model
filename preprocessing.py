"""
Text preprocessing module for MITRA
Author: Lokesh Manepalli
"""

import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def tokenize_text(text):
    return word_tokenize(text)


def preprocess_pipeline(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    return tokens
