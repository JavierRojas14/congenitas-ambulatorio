# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import os
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import unidecode
import hashlib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv(find_dotenv())

COLS_A_PREPROCESAR_TEXTO = os.environ.get("COLS_A_PREPROCESAR_TEXTO").split(",")
COLS_INFO_SENSIBLE = os.environ.get("COLS_INFO_SENSIBLE").split(",")


def filtrar_palabras_stopword(texto, idioma_palabras_stopword):
    stopwords_elegidas = set(stopwords.words(idioma_palabras_stopword))

    tokens = texto.split()
    filtro_stop_words = [palabra for palabra in tokens if palabra not in stopwords_elegidas]
    texto_juntado = " ".join(filtro_stop_words)

    return texto_juntado


def lematizar_texto(texto):
    tokens = texto.split()
    motor = WordNetLemmatizer()
    palabras_reducidas = [motor.lemmatize(palabra, "v") for palabra in tokens]
    texto_juntado = " ".join(palabras_reducidas)

    return texto_juntado


def preprocesar_columna_texto(serie_texto):
    serie_limpia = serie_texto.copy()

    serie_limpia = (
        serie_limpia.dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: filtrar_palabras_stopword(x, "spanish"))
        .apply(unidecode.unidecode)
        .apply(lematizar_texto)
    )

    return serie_limpia


def hashear_columna_texto(serie_texto):
    serie_hasheada = serie_texto.copy()

    serie_hasheada = serie_hasheada.astype(str).apply(
        lambda x: hashlib.sha512(x.encode()).hexdigest()
    )

    return serie_hasheada


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    df = pd.read_excel(input_filepath)
    df = df.dropna(how="all")
    df.loc[:, COLS_A_PREPROCESAR_TEXTO] = df.loc[:, COLS_A_PREPROCESAR_TEXTO].apply(
        preprocesar_columna_texto
    )
    df.loc[:, COLS_INFO_SENSIBLE] = df.loc[:, COLS_INFO_SENSIBLE].apply(hashear_columna_texto)

    df.to_csv(output_filepath, encoding="latin-1", index=False, sep=";", errors="replace")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
