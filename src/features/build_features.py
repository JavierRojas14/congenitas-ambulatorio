import Levenshtein
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

STOPWORDS_ESPANOL = set(stopwords.words("spanish"))


def crear_matriz_sparse_para_texto(serie_texto):
    vectorizador = CountVectorizer()
    matriz_sparse = vectorizador.fit_transform(serie_texto)

    return vectorizador, matriz_sparse


def calcular_distancia_levenshtein_columna_texto(serie_texto):
    return serie_texto.apply(lambda x: [Levenshtein.distance(x, y) for y in serie_texto]).tolist()


def agrupar_textos_en_columna(serie_texto):
    matriz_levenshtein = calcular_distancia_levenshtein_columna_texto(serie_texto)
    distancias = squareform(matriz_levenshtein)

    linkage_matrix = linkage(distancias, method="average")
    threshold = 3
    clusters = fcluster(linkage_matrix, threshold, criterion="distance")

    df_resultado = pd.DataFrame({serie_texto.name: serie_texto, "cluster": clusters}).sort_values(
        "cluster"
    )

    return df_resultado