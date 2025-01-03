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

COLS_A_PREPROCESAR_TEXTO = [
    "diagnostico_principal",
    "hospital",
    "sexo",
    "prevision",
    "centro_referencia",
    # "cie10",
    # "cie11",
    "region",
    "clasificacion",
    "procedimiento",
    "complejidad",
]

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

TRANSFORMACION_FECHAS_NACIMIENTO = {
    "26/002/1961": "1961/02/26",
    "01/11991": "1991/01/01",
    "...": np.nan,
}

TRANSFORMACION_SEXO = {
    "df": "f",
    "b": "m",
}

STOPWORDS_ESPANOL = set(stopwords.words("spanish"))


def filtrar_palabras_stopword(texto, stopwords_elegidas):
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
    if serie_texto.name != "PREVISION":
        palabras_filtro = STOPWORDS_ESPANOL

    else:
        palabras_filtro = STOPWORDS_ESPANOL - {"a"}

    serie_limpia = (
        serie_limpia.dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: filtrar_palabras_stopword(x, palabras_filtro))
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


def formatear_fecha_primera_evaluacion(serie_fecha):
    # Creamos una copia de la serie
    serie_fecha = serie_fecha.copy()

    # Limpiamos la columna, reemplazando fechas y luego casteandolas
    serie_fecha = serie_fecha.replace(TRANSFORMACION_FECHAS_PRIMERA_CONSULTA)
    serie_fecha = pd.to_datetime(serie_fecha)

    return serie_fecha


def formatear_fecha_nacimiento(serie_fecha):
    # Creamos una copia de la serie
    serie_fecha = serie_fecha.copy()

    # Limpiamos la columna, reemplazando fechas y luego casteandolas
    serie_fecha = serie_fecha.replace(TRANSFORMACION_FECHAS_NACIMIENTO)
    serie_fecha = pd.to_datetime(serie_fecha, yearfirst=True)

    return serie_fecha


def formatear_columnas_fecha_nacimiento(df):
    tmp = df.copy()

    serie_fecha = df["F NAC"]

    fechas_reemplazadas = pd.to_datetime(
        serie_fecha.replace(TRANSFORMACION_FECHAS_NACIMIENTO), yearfirst=True
    )

    tmp["F NAC"] = fechas_reemplazadas

    return tmp


def recodificar_cols_dict_de_congenitas(df):
    tmp = df.copy()

    traductor_congenitas = pd.ExcelFile("data/external/Trabajo Javier_V1_AH.xlsx")
    cols_a_recodificar = [
        "diagnostico_principal",
        "region",
        "clasificacion",
        "complejidad",
        "prevision",
    ]
    for col in cols_a_recodificar:
        df_traductor = pd.read_excel(traductor_congenitas, sheet_name=col).drop(columns="cluster")
        diccionario = df_traductor.set_index(col)["validacion"].to_dict()
        tmp[col] = tmp[col].replace(diccionario)

    return tmp


def agregar_info_codigo_cie(df, columna_con_cie):
    cie = pd.read_excel("data/external/CIE-10.xlsx")

    union = pd.merge(df, cie, how="left", left_on=columna_con_cie, right_on="Código")

    return union


def clean_column_names(df):
    """
    Cleans the column names of a DataFrame by converting to lowercase, replacing spaces with
    underscores, ensuring only a single underscore between words, and removing miscellaneous symbols.

    :param df: The input DataFrame.
    :type df: pandas DataFrame

    :return: The DataFrame with cleaned column names.
    :rtype: pandas DataFrame
    """
    tmp = df.copy()

    # Clean and transform the column names
    cleaned_columns = (
        df.columns.str.lower()
        .str.normalize("NFD")
        .str.encode("ascii", "ignore")
        .str.decode("utf-8")
        .str.replace(
            r"[^\w\s]", "", regex=True
        )  # Remove all non-alphanumeric characters except spaces
        .str.replace(r"\s+", "_", regex=True)  # Replace spaces with underscores
        .str.replace(r"_+", "_", regex=True)  # Ensure only a single underscore between words
        .str.strip("_")
    )

    # Assign the cleaned column names back to the DataFrame
    tmp.columns = cleaned_columns

    return tmp


def procesar_base_de_congenitas(input_filepath):
    # Carga la base de datos
    ruta_archivo = f"{input_filepath}/Base datos configurada INT (Actualizada 26-05-2023).xls"
    df = pd.read_excel(ruta_archivo)

    # Limpieza de la base de datos
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)

    # Limpia los nombres de las columnas
    df = clean_column_names(df)

    # Preprocesamiento de texto
    df.loc[:, COLS_A_PREPROCESAR_TEXTO] = df.loc[:, COLS_A_PREPROCESAR_TEXTO].apply(
        preprocesar_columna_texto
    )

    # Convierte los CIE a mayuscula
    # df["cie10"] = df["cie10"].str.upper()
    # df["cie11"] = df["cie11"].str.upper()

    # Cambia glosas de sexo
    df["sexo"] = df["sexo"].replace(TRANSFORMACION_SEXO)

    # Formatea columna de primera evaluacion
    df["fecha_1_evaluacion"] = formatear_fecha_primera_evaluacion(df["fecha_1_evaluacion"])
    df["anio_1_evaluacion"] = df["fecha_1_evaluacion"].dt.year

    # Formatea columna de fecha de nacimiento
    df["f_nac"] = formatear_fecha_nacimiento(df["f_nac"])

    # Recodifica columnas con diccionarios
    df = recodificar_cols_dict_de_congenitas(df)

    # Elimina registros sin diagnostico principal
    df = df.dropna(subset="diagnostico_principal")

    # Filtra solamente las columnas a utilizar
    df = df[
        ["rut"]
        + COLS_A_PREPROCESAR_TEXTO
        + [
            "f_nac",
            "fecha_1_evaluacion",
            "anio_1_evaluacion",
            "edad_1_evaluacion_op",
        ]
    ]

    # Ordena por la fecha de la primera atencion
    df = df.sort_values("fecha_1_evaluacion")

    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Lee la base de congenitas y la procesa
    df = procesar_base_de_congenitas(input_filepath)

    # Guarda la base de datos procesada
    ruta_output = f"{output_filepath}/df_procesada.csv"
    df.to_csv(ruta_output, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
