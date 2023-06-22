# -*- coding: utf-8 -*-
import hashlib
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import unidecode
from dotenv import find_dotenv, load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv(find_dotenv())

COLS_A_PREPROCESAR_TEXTO = os.environ.get("COLS_A_PREPROCESAR_TEXTO").split(",")
COLS_INFO_SENSIBLE = os.environ.get("COLS_INFO_SENSIBLE").split(",")
RUT_EN_FECHA = os.environ.get("RUT_EN_FECHA")

TRANSFORMACION_FECHAS_PRIMERA_CONSULTA = {
    "25-8-2020 (Presencial)": "25/08/2020",
    "21 julio 2020 (Politelefonico por Contingencia)": "21/07/2020",
    "25 agosto 2020 (Presencial)": "25/08/2020",
    "5/082014": "05/08/2014",
    "04/08/2020 (poli telefonico)": "04/08/2020",
    "30/09/2020 (Presencial)": "30/09/2020",
    "09/002/2016": "09/02/2016",
    "16-3-2022 hospitalizada": "16/03/2022",
    "31/07/20188": "31/07/2018",
    "26-05-202": np.nan,
    "8-092020 (oli Presencial)": "08/09/2020",
    "27/10/2020 (Poli Presencial)": "27/10/2020",
    "08/04/2003": "08/04/2003",
    "0705/2019": "07/05/2019",
    "22 sept 2020 (Presencial)": "22/09/2020",
    "06/11/25018": "06/11/2018",
    "11/10/21016": "11/10/2016",
    "27/11/0212": "27/11/2012",
    "30/09/2020 (Preencial)": "30/09/2020",
    "13/'07/2010": "13/07/2010",
    "12-10-2021.": "12/10/2021",
    "14 julio 2020 (Contingencia COVID)": "14/07/2020",
    "1-9-2020 (Presencial)": "01/09/2020",
    "24-05-202": np.nan,
    "13/09/20146": "13/09/2014",
    "19 MAYO 2020 (POLI TELEFONICO CONTINGENCIA covid": "19/05/2020",
    "29/9/2020 (P.Presencial)": "29/09/2020",
    "24/03/009": "24/03/2009",
    "002/05/2017": "02/05/2017",
    "29/9/2020 (P. Presencial)": "29/09/2020",
    "22-09-2020 (Presencial)": "22/09/2020",
    "19-70-2022": "19/07/2022",
    "8/09/2020 (Poli Presencial)": "08/09/2020",
    "13/016/2011": "13/01/2011",
    RUT_EN_FECHA: np.nan,
    "27 Octubre 2020. ": "27/10/2020",
    "15/0/2013": np.nan,
    "19/03/201": np.nan,
    "29/10/2020 (Cateterismo)": "29/10/2020",
    "20-010-2023": np.nan,
    "2/09/2020 (Poli Presencial)": "02/09/2020",
    "15/10/013": "15/10/2013",
}

TRANSFORMACION_SEXO = {
    "df": "f",
    "b": "m",
}


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


def formatear_columnas_fecha_primera_evaluacion(df):
    tmp = df.copy()

    serie_fecha = df["FECHA 1º evaluación"]

    fechas_reemplazadas = pd.to_datetime(serie_fecha.replace(TRANSFORMACION_FECHAS_PRIMERA_CONSULTA), dayfirst=True)
    anio_primera_evaluacion = fechas_reemplazadas.dt.year
    mes_primera_evaluacion = fechas_reemplazadas.dt.month

    tmp["FECHA 1º evaluación"] = fechas_reemplazadas
    tmp["ANIO_PRIMERA_EVALUACION"] = anio_primera_evaluacion
    tmp["MES_PRIMERA_EVALUACION"] = mes_primera_evaluacion

    return tmp


def agregar_cie_para_glosa(df):
    traductor_glosa_cie = pd.read_excel("data/external/Trabajo Javier_V1_AH.xlsx").drop(
        columns="cluster"
    )
    union = pd.merge(df, traductor_glosa_cie, how="left", on="DIAGNOSTICO PRINCIPAL")

    return union


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
    df["SEXO"] = df["SEXO"].replace(TRANSFORMACION_SEXO)
    df = formatear_columnas_fecha_primera_evaluacion(df)
    df = agregar_cie_para_glosa(df)

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
