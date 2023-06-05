"""
Auxiliary functions for analyzing data distribution in a DataFrame.

This module provides functions to analyze the distribution of categorical and numerical variables,
analyze missing values, visualize distribution plots, and explore correlations between variables.

Functions:
- separar_df_a_numericas_categoricas: Separates a DataFrame into numerical and categorical
  variables.
- graficar_distribucion_variable_numerica: Plots the distribution of a numerical variable.
- analizar_distr_todas_las_variables_numericas: Analyzes the distribution of all numerical
  variables.
- graficar_distribucion_variable_categorica: Plots the distribution of a categorical variable.
- analizar_dist_todas_las_variables_categoricas: Analyzes the distribution of all categorical
  variables.
- analizar_valores_faltantes: Analyzes missing values in a set of variables.
- analizar_distribucion_y_faltantes_variables: Analyzes the distribution and missing values of
  all variables in a DataFrame.
- mostrar_perdida_de_datos: Displays the percentage of data lost when removing missing values
  from a DataFrame.
- analizar_correlacion_todas_las_variables: Analyzes the correlation between all variables in a
  DataFrame.
"""
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns


def separar_df_a_numericas_categoricas(df):
    """
    Separates a DataFrame into numerical and categorical variables.

    :param df: The DataFrame to be separated.
    :type df: pd.DataFrame

    :return: A tuple containing a DataFrame with numerical variables and another with categorical
        variables.
    :rtype: tuple
    """
    numericas = df.select_dtypes("number")
    categoricas = df.select_dtypes("object")
    return numericas, categoricas


def graficar_distribucion_variable_numerica(serie_numerica, nombre_grafico):
    """
    Plots the distribution of a numerical variable.

    :param serie_numerica: The numerical series to be plotted.
    :type serie_numerica: pd.Series

    :param nombre_grafico: The name of the plot.
    :type nombre_grafico: str

    :return: None
    """
    fig, axis = plt.subplots(1, 2)
    print("-----------------------------------------")
    sns.histplot(data=serie_numerica, ax=axis[0])
    axis[0].axvline(serie_numerica.mean(), color="tomato")
    sns.boxplot(data=serie_numerica, ax=axis[1])

    plt.title(nombre_grafico)
    plt.show()
    print("-----------------------------------------")


def analizar_distr_todas_las_variables_numericas(df_numericas):
    """
    Analyzes the distribution of all numerical variables in a DataFrame.

    :param df_numericas: The DataFrame with the numerical variables to be analyzed.
    :type df_numericas: pd.DataFrame

    :return: None
    """
    if not (df_numericas.empty):
        print("Analizando todas las variables numericas \n")
        display(df_numericas.describe())

        for columna_numerica, serie_numerica in df_numericas.items():
            graficar_distribucion_variable_numerica(serie_numerica, columna_numerica)


def graficar_distribucion_variable_categorica(serie_categorica, nombre_grafico):
    """
    Plots the distribution of a categorical variable.

    :param serie_categorica: The categorical series to be plotted.
    :type serie_categorica: pd.Series
    :param nombre_grafico: The name of the plot.
    :type nombre_grafico: str

    :return: None
    """
    frecuencias = serie_categorica.value_counts()
    porcentajes = serie_categorica.value_counts("%")
    total = pd.DataFrame(
        {"Frecuencia": frecuencias, "Porcentaje": porcentajes}, index=frecuencias.index
    )
    print("-----------------------------------------")
    display(total)

    sns.countplot(y=serie_categorica, order=frecuencias.index)
    plt.title(nombre_grafico)
    plt.show()
    print("-----------------------------------------")


def analizar_dist_todas_las_variables_categoricas(df_categoricas):
    """
    Analyzes the distribution of all categorical variables in a DataFrame.

    :param df_categoricas: The DataFrame with the categorical variables to be analyzed.
    :type df_categoricas: pd.DataFrame

    :return: None
    """
    print("Analizando todas las variables categoricas \n")
    for columna_categorica, serie_categorica in df_categoricas.items():
        graficar_distribucion_variable_categorica(serie_categorica, columna_categorica)


def analizar_valores_faltantes(variables_a_analizar):
    """
    Analyzes missing values in a set of variables.

    :param variables_a_analizar: The DataFrame with the variables to be analyzed.
    :type variables_a_analizar: pd.DataFrame

    :return: None
    """
    valores_faltantes = variables_a_analizar.isnull().sum()
    porcentaje_faltantes = round(valores_faltantes * 100 / len(variables_a_analizar), 2)

    faltantes_resumen = pd.DataFrame(
        {"cantidad_na": valores_faltantes, "porcentaje_na": porcentaje_faltantes}
    )
    display(faltantes_resumen)

    msno.matrix(variables_a_analizar)


def analizar_distribucion_y_faltantes_variables(df):
    """
    Analyzes the distribution and missing values of all variables in a DataFrame.

    :param df: The DataFrame to be analyzed.
    :type df: pd.DataFrame

    :return: None
    """
    numericas, categoricas = separar_df_a_numericas_categoricas(df)
    analizar_distr_todas_las_variables_numericas(numericas)
    analizar_dist_todas_las_variables_categoricas(categoricas)
    analizar_valores_faltantes(df)


def mostrar_perdida_de_datos(df_completa):
    """
    Displays the percentage of data lost when removing missing values from a DataFrame.

    :param df_completa: The complete DataFrame.
    :type df_completa: pd.DataFrame

    :return: None
    """
    cantidad_valores_originales = len(df_completa)
    cantidad_valores_droppeados = len(df_completa.dropna())
    porcentaje_droppeo = cantidad_valores_droppeados / cantidad_valores_originales
    cambio = round((1 - porcentaje_droppeo) * 100, 2)

    print(
        f"Al droppear todos los valores faltantes en la DataFrame se pierde el {cambio}% "
        f"de los datos totales"
    )


def analizar_correlacion_todas_las_variables(df_variables):
    """
    Analyzes the correlation between all variables in a DataFrame.

    :param df_variables: The DataFrame with the variables to be analyzed.
    :type df_variables: pd.DataFrame

    :return: None
    """
    corr = df_variables.corr()
    sns.heatmap(corr, cmap="Blues", annot=True)
    plt.show()
