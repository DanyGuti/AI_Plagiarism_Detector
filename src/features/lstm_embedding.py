'''
LSTM-based token embedding feature extraction.
'''
from gensim.models import Word2Vec
import keras
import tensorflow as tf
import numpy as np

VOCABULARY_SIZE = 300
EMBEDDING_DIM = 10

def read_tokens_from_file(file_path: str) -> list[tuple[str, str]]:
    '''
    Reads tokens from a file and returns a list of tuples containing token text and type.
    Args:
        file_path (str): Path to the file containing tokens.
    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains \
            the token text and its type.
    '''
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                token_text, token_type = parts
                tokens.append((token_text, token_type))
    return tokens

# 1. Prepare the corpus from the tokens
def prepare_corpus(tokens: list[tuple[str, str]]) -> list[list[str]]:
    '''
    Prepares a corpus from the list of tokens for training the Word2Vec model.
    Args:
        tokens (List[Tuple[str, str]]): List of tuples containing token text and type.
    Returns:
        List[List[str]]: A list of lists where each inner list contains token texts.
    '''
    corpus = []
    for token_text, token_type in tokens:
        corpus.append([f"{token_text}_{token_type}"])
    return corpus

# 2. Train the Word2Vec model
def train_word2vec_model(corpus: list[list[str]]) -> Word2Vec:
    '''
    Trains a Word2Vec model on the provided corpus.
    Args:
        corpus (List[List[str]]): List of lists where each inner list contains token texts.
    Returns:
        Word2Vec: A trained Word2Vec model.
    '''
    model = Word2Vec(
        sentences=corpus,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=1,
        workers=4,
        # TODO try to use sg=1 (skip-gram) instead of CBOW
    )
    return model

# 3. Generate embedding vectors for the tokens
def generate_embedding_vector(tokens: list[tuple[str, str]], model: Word2Vec):
    '''
    Given a list of tokens and a trained Word2Vec model,
    generates embedding vectors for each token.
    Args:
        tokens (List[Tuple[str, str]]): List of tuples containing token text and type.
        model (Word2Vec): Trained Word2Vec model.
    Returns:
        np.ndarray: An array of embedding vectors for the tokens.
    '''
    tokens = [f"{token_text}_{token_type}" for token_text, token_type in tokens]
    embedding_dim = model.vector_size
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embedding = model.wv[token]
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(embedding_dim))

    return np.array(embeddings)
