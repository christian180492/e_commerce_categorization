#  Final Project  - Christian Ojeda
> e-Commerce Products Categorization

## Structure of the project.

|- api/
    |- Dockerfile
    |- app.py
    |- static/
        |- imagenes/
    |- templates/
        |- historial.html
        |- index.html
        |- productos.html
    requirements.txt
|- dataset/
    |- categorias_buleanas.csv
    |- categories_normalized.csv
    |- categories.json
    |- prod_descri_normalized.csv
    |- prod_names_normalized.csv
    |- products.json
    |- reporte_descripciones_df.csv
    |- same_name_dif_IDs.csv
    |- stores.json
|- modelo/
    |- Dockerfile
    |- ejemplo_salida.csv
    |- modelo.py
    |- requirements.txt
|- modelos/
    |- best_mlp_model.h5
    |- best_moc.pkl
    |- descri_lstm_mlp.h5
    |- descri_mlp_base.h5
    |- descri_mlp_elim.h5
    |- modelo_w2v.model
|- src
    |- contractions.py
    |- text_normalizer.py
|- docker-compose.yml
|- EDA.ipynb
|- model_train.ipynb
|- README.md
|- requirements.txt

## EDA
In the `EDA`, we create visualizations of the amount of characters, words, sentences, and special characters in the descriptions and names of each product. 
Another thing we do is analyze the most frequent categories used in each example and the least common ones. Since the dataset is not balanced, we had to combine some of the categories. All the categories that had fewer than 100 appearances became one category named 'others,' and the categories with fewer than 1000 appearances were combined with categories that represent them well enough using a word2vec model. 
After this, we ended up with 123 categories to classify. Following this step, we implemented a `RandomForest` with `MultiOutputClassifier` to detect multiple categories from descriptions, and the same type of model was used to detect multiple categories from the names of each product.
These two models became a first try and the next searching of models were done in the `model_train.ipynb`.

## Implementation of `model_train.ipynb`
In the `model_train.ipynb` notebook, we use some insights we've gained from the `EDA` to transform the text. We normalize it by eliminating unnecessary words, lemmatize it, and perform several other steps to prepare the text for different models for training.
In this notebook, we trained a couple of MLP models and also an LSTM-MLP model. However, none of them yielded better results than the implementation of the `RandomForest` with the `MultiOutputClassifier`.

## Deployment of the API.

There is a `docker-compose.yml` that reads the Dockerfile from the `api` and `modelo` folders to build them and create the containers. For this purpose, there is also a `requirements.txt` document in each folder to provide the needed libraries for each of the containers.


