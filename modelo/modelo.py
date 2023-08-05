import redis
import pickle
import nltk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

y = pd.read_csv('ejemplo_salida.csv')

from src import text_normalizer
stop_words = nltk.corpus.stopwords.words("english")

from nltk.tokenize.toktok import ToktokTokenizer

import gensim.downloader
pretrained_w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')

# Cargar el modelo al iniciar el script
ruta_modelo = '/app/modelos/best_moc.pkl'  # Ruta del modelo dentro del contenedor
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
    oraciones: La lista de oraciones que se desea tokenizar.
    ninit: Cantidad de ngramas mínimo que uno desea obtener como tokens.
    nfin: Cantidad de ngramas máximo que uno desea obtener como tokens.
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
    global modelo  # Utiliza la variable global del modelo
    with open(ruta_modelo, 'rb') as file:
        modelo = pickle.load(file)
    print(f"Modelo cargado exitosamente.\n Estos son sus mejores parámetros: ¡¡¡{modelo.best_params_}!!!")

def procesar_datos(nombre, descripcion, imagen):
    # Verificar si el modelo ya ha sido cargado
    if modelo is None:
        print("El modelo aún no ha sido cargado. Cargando...")
        cargar_modelo(ruta_modelo)

    # Lógica para procesar los datos recibidos y obtener las categorías usando el modelo
    descripcion_normalizada = normalizacion([descripcion])
    print(f'\n\nESTA ES LA DESCRIPCIÓN NORMALIZADA {descripcion_normalizada}\n\n')
    descripcion_tokenizada = toktok(descripcion_normalizada, 1, 1)
    print(f'\n\nESTA ES LA DESCRIPCIÓN TOKENIZADA {descripcion_tokenizada}\n\n')
    descripcion_vectorizada = vectorizer_pretrained(descripcion_tokenizada, pretrained_w2v_model, 300)
    categoria_binaria = modelo.predict(descripcion_vectorizada)
    print(categoria_binaria)

    clave_valor = zip(y.columns, categoria_binaria[0])
    diccionario = dict(clave_valor)

    categorias = []
    for clave, valor in diccionario.items():
        if valor != 0:
            categorias.append(clave)

    categorias = ['Categoria 1', 'Categoria 2', 'Categoria 3']  # Ejemplo: categorías predichas por el modelo
    return categorias

if __name__ == '__main__':
    # Conectar a Redis
    redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

    # Esperar a que se envíen los datos desde Flask a Redis
    while not redis_client.exists('producto'):
        pass

    # Obtener los datos desde Redis
    datos = redis_client.hgetall('producto')
    nombre = datos.get(b'nombre').decode('utf-8')
    descripcion = datos.get(b'descripcion').decode('utf-8')
    imagen = datos.get(b'imagen').decode('utf-8') if datos.get(b'imagen') else None

    # Procesar los datos y obtener las categorías
    categorias = procesar_datos(nombre, descripcion, imagen)

    # Guardar las categorías en Redis
    redis_client.set('categorias', ','.join(categorias))

