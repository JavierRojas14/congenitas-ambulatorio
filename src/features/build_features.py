from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

STOPWORDS_ESPANOL = set(stopwords.words("spanish"))


def crear_matriz_sparse_para_texto(serie_texto):
    vectorizador = CountVectorizer()
    matriz_sparse = vectorizador.fit_transform(serie_texto)

    return vectorizador, matriz_sparse
