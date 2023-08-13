import re
import subprocess
import unicodedata
from typing import List, Optional

import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer

from src.contractions import CONTRACTION_MAP

# Download the models used
nltk.download("stopwords")
nltk.download("punkt")
subprocess.run(["spacy", "download", "en_core_web_sm"])

nltk.download('averaged_perceptron_tagger')

# Load NLP models
tokenizer = ToktokTokenizer()
nlp = spacy.load("en_core_web_sm")
stopword_list = nltk.corpus.stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def remove_html_tags(text: str) -> str:
    """
    Remove html tags from text like <br/> , etc.
    """
    texto = BeautifulSoup(text, 'html.parser') 
    return texto.get_text()


def stem_text(text: str) -> str:
    """
    Stem input string.
    """
    # Tokenize text into words
    tokens = word_tokenize(text)

    # Initialize Porter's stemmer
    stemmer = PorterStemmer()

    # Get the stem of each word and join them in a text string
    stemmed_words = [stemmer.stem(word) for word in tokens]
    output_text = ' '.join(stemmed_words)
    return output_text

def lemmatize_text(text: str) -> str:
    """
    Lemmatize input string, tokenizing first and extracting lemma from each text after.
    """
    # Tokenize the text using SpaCy's nlp model
    doc = nlp(text)

    # Extract the lemmas from each token and join them in a text string
    lemmas = [token.lemma_ for token in doc]
    output_text = ' '.join(lemmas)

    return output_text


def remove_accented_chars(text: str) -> str:
    """
    Remove accents from input string.
    """
    # Normalize text to separate accented characters into their components
    normalized_text = unicodedata.normalize('NFKD', text)

    # Filter out characters that are combination markers (accents)
    output_text = ''.join(c for c in normalized_text if not unicodedata.combining(c))

    return output_text


def remove_special_chars(text: str, remove_digits: Optional[bool] = False) -> str:
    """
    Remove non-alphanumeric characters from input string.
    """
    if remove_digits:
        filtered_doc = re.sub(r'[^A-Za-z\s]+', '',text)
    else:
        filtered_doc = re.sub(r'[^A-Za-z0-9\s]+', '',text)
        
    return filtered_doc
    

def remove_stopwords(
    text: str,
    is_lower_case: Optional[bool] = False,
    stopwords: Optional[List[str]] = stopword_list,
) -> str:
    """
    Remove stop words using list from input string.
    """
    # Initialize the tokenizer
    tokenizer = ToktokTokenizer()

    # Tokenize the text.
    tokens = tokenizer.tokenize(text)

    # Apply lowercase if necessary
    if not is_lower_case:
        tokens = [token.lower() for token in tokens]

    # Filter out tokens that are not stopwords if the list of stopwords is provided
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Bind the filtered tokens into a text string
    output_text = ' '.join(filtered_tokens)

    return output_text


def remove_extra_new_lines(text: str) -> str:
    """
    Remove extra new lines or tab from input string.
    """
    # Remove extra lines and tabs
    cleaned_text = ' '.join(text.splitlines())

    return cleaned_text


def remove_extra_whitespace(text: str) -> str:
    """
    Remove any whitespace from input string.
    """
    # Remove extra whitespace
    cleaned_text = ' '.join(text.split())

    return cleaned_text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP) -> str:
    """
    Expand english contractions on input string.
    """
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", expanded_text)

    return text


def normalize_corpus(
    corpus: List[str],
    html_stripping: Optional[bool] = True,
    contraction_expansion: Optional[bool] = True,
    accented_char_removal: Optional[bool] = True,
    text_lower_case: Optional[bool] = True,
    text_stemming: Optional[bool] = False,
    text_lemmatization: Optional[bool] = False,
    special_char_removal: Optional[bool] = True,
    remove_digits: Optional[bool] = True,
    stopword_removal: Optional[bool] = True,
    stopwords: Optional[List[str]] = stopword_list,
) -> List[str]:
    """
    Normalize list of strings (corpus)

    Args:
        corpus : List[str]
            Text corpus.
        html_stripping : bool
            Html stripping,
        contraction_expansion : bool
            Contraction expansion,
        accented_char_removal : bool
            accented char removal,
        text_lower_case : bool
            Text lower case,
        text_stemming : bool
            Text stemming,
        text_lemmatization : bool
            Text lemmatization,
        special_char_removal : bool
            Special char removal,
        remove_digits : bool
            Remove digits,
        stopword_removal : bool
            Stopword removal,
        stopwords : List[str]
            Stopword list.

    Return:
        List[str]
            Normalized corpus.
    """

    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)

        # Remove extra newlines
        doc = remove_extra_new_lines(doc)

        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # Expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)

        # Remove special chars and\or digits
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits=remove_digits)

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc, is_lower_case=text_lower_case, stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
