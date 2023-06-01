# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def filtrar_palabras_stopword(texto, idioma_palabras_stopword):
    stopwords_elegidas = set(stopwords.words(idioma_palabras_stopword))

    tokens = texto.split()
    filtro_stop_words = [palabra for palabra in tokens if palabra not in stopwords_elegidas]
    texto_juntado = " ".join(filtro_stop_words)

    return texto_juntado


def stemmear_o_lematizar_texto(texto, stem_o_lema):
    tokens = texto.split()
    if stem_o_lema == "stem":
        motor = PorterStemmer()
        palabras_reducidas = [motor.stem(palabra, "v") for palabra in tokens]

    elif stem_o_lema == "lema":
        motor = WordNetLemmatizer()
        palabras_reducidas = [motor.lemmatize(palabra, "v") for palabra in tokens]

    return palabras_reducidas


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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
