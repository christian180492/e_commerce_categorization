import redis
import pickle
import nltk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

# We read the outputs to have a reference of the output classes.
y = pd.read_csv('ejemplo_salida.csv')

# We get the stopwords, tokenizer and pretrained word2vec model to proces entrance data.
from src import text_normalizer
stop_words = nltk.corpus.stopwords.words("english")

from nltk.tokenize.toktok import ToktokTokenizer

import gensim.downloader
pretrained_w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')

# Path to get the model.
ruta_modelo = '/app/modelos/best_moc.pkl' 
modelo = None

def normalizacion(oracion):
    oracion_normalizada = text_normalizer.normalize_corpus(oracion, stopwords=stop_words, text_lemmatization=True)
    return oracion_normalizada

def generate_ngrams(tokens, n):
    if n == 1:
        return tokens
    else:
        return [' '.join(ngram) for ngram in ngrams(tokens, n)]

def toktok(oraciones, ninit=1, nfin=1):
    '''
    oraciones: List of sentences to tokenize.
    ninit: Minimun amount of ngrams to get from tokens.
    nfin: Maximun amount of ngrams to get from tokens.
    '''

    tokenizer = ToktokTokenizer()

    oraciones_tokenizadas_ngramas = []
    
    for oracion in oraciones:
        ngram_tokens = [generate_ngrams(tokenizer.tokenize(oracion), n) for n in range(ninit, nfin+1)]
        ngram_tokens_flattened = [token for ngram in ngram_tokens for token in ngram]
        oraciones_tokenizadas_ngramas.append(ngram_tokens_flattened)
        
    return oraciones_tokenizadas_ngramas

def vectorizer_pretrained(corpus, model, num_features: int=100):
    corpus_size = len(corpus)
    corpus_vectors = np.zeros((corpus_size, num_features), dtype=np.float32)

    for i, document in enumerate(corpus):
        word_count = 0
        doc_vector = np.zeros(num_features, dtype=np.float32)

        for word in document:
            if word in model:
                doc_vector += model.word_vec(word)
                word_count+=1
        
        if word_count > 0:
            doc_vector /= word_count
        
        corpus_vectors[i] = doc_vector
    
    return corpus_vectors

def cargar_modelo(ruta_modelo):
    global modelo  # We use the global variable modelo.
    with open(ruta_modelo, 'rb') as file:
        modelo = pickle.load(file)
    print(f"Model uploaded successfully.\n")

def procesar_datos(nombre, descripcion, imagen):
    # Verify if the model is loaded.
    if modelo is None:
        print("Charging the model.")
        cargar_modelo(ruta_modelo)

    # Logic to process the received data and obtain the categories using the model
    descripcion_normalizada = normalizacion([descripcion])
    descripcion_tokenizada = toktok(descripcion_normalizada, 1, 1)
    descripcion_vectorizada = vectorizer_pretrained(descripcion_tokenizada, pretrained_w2v_model, 300)
    
    # We get the probability values of each category
    predict_proba = modelo.predict_proba(descripcion_vectorizada)

    # We  filter the categories to obtain the ones with probability greater than 0.3
    prediccion = []
    for ele in predict_proba:
        prediccion.append(ele[0][1])
    categoria_binaria = [[1 if ele > 0.3 else 0 for ele in fila] for fila in [prediccion]]

    # We obtain the categories
    clave_valor = zip(y.columns, categoria_binaria[0])
    diccionario = dict(clave_valor)

    categorias = []
    for clave, valor in diccionario.items():
        if valor != 0:
            categorias.append(clave)

    return categorias

if __name__ == '__main__':
    # Connect to Redis
    redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

    # Wait for data to be sent from Flask to Redis.
    while not redis_client.exists('producto'):
        pass

    # Get data from Redis.
    datos = redis_client.hgetall('producto')
    nombre = datos.get(b'nombre').decode('utf-8')
    descripcion = datos.get(b'descripcion').decode('utf-8')
    imagen = datos.get(b'imagen').decode('utf-8') if datos.get(b'imagen') else None

    # Process the data and get the categories
    categorias = procesar_datos(nombre, descripcion, imagen)

    # Save categories in Redis
    redis_client.set('categorias', ','.join(categorias))