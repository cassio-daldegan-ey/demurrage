import os
import glob
import json
import calendar
import datetime as dt
from functools import reduce
from datetime import date
from datetime import timedelta
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from prophet import Prophet


def TotalEmbarcado(
    indices1: pd.DataFrame,
    indices2: pd.DataFrame,
    indices3: pd.DataFrame,
    indices4: pd.DataFrame,
    indices5: pd.DataFrame,
    indices6: pd.DataFrame,
    indices7: pd.DataFrame,
    indices8: pd.DataFrame,
    indices9: pd.DataFrame,
    indices10: pd.DataFrame,
):

    """A funcao acima faz o cálculo do Volume Total Embarcado por porto e por pier."""
    for pier_porto in (
        [indices1, "1", "Ponta Madeira"],
        [indices2, "3n", "Ponta Madeira"],
        [indices3, "3s", "Ponta Madeira"],
        [indices4, "4n", "Ponta Madeira"],
        [indices5, "4s", "Ponta Madeira"],
        [indices6, "Total", "Sepetiba"],
        [indices7, "Total", "Guaiba"],
        [indices8, "2", "Tubarao"],
        [indices9, "1n", "Tubarao"],
        [indices10, "1s", "Tubarao"],
    ):
        pier_porto[0]["Dia"] = (
            pd.to_datetime(pier_porto[0]["DATA_MIN"], dayfirst=True)
        ).dt.normalize()
        pier_porto[0]["Pier"] = pier_porto[1]
        pier_porto[0]["Port"] = pier_porto[2]
        pier_porto[0] = pier_porto[0][["Dia", "Pier", "Port", "CAPACIDADE"]]

    pm_total = pd.concat(
        [indices1, indices2, indices3, indices4, indices5]
    ).reset_index(drop=True)
    gs_total = pd.concat([indices6, indices7]).reset_index(drop=True)
    t_total = pd.concat([indices8, indices9, indices10]).reset_index(drop=True)

    pm_total["Pier"] = "Total"
    gs_total["Pier"] = "Total"
    gs_total["Port"] = "Guaiba e Sepetiba"
    t_total["Pier"] = "Total"

    dfs = [
        pm_total,
        indices1,
        indices2,
        indices3,
        indices4,
        indices5,
        gs_total,
        indices7,
        indices6,
        t_total,
        indices8,
        indices9,
        indices10,
    ]

    dataset_total_embarcado = pd.concat(dfs).reset_index(drop=True)
    dataset_total_embarcado.columns = dataset_total_embarcado.columns.str.replace(
        "CAPACIDADE", "Total Embarcado"
    )

    dataset_total_embarcado["Total Embarcado"] = dataset_total_embarcado[
        "Total Embarcado"
    ].astype(float)
    dataset_total_embarcado = dataset_total_embarcado.groupby(["Dia", "Port", "Pier"])[
        "Total Embarcado"
    ].agg("sum")
    dataset_total_embarcado = dataset_total_embarcado.reset_index()
    dataset_total_embarcado.columns = dataset_total_embarcado.columns.str.replace(
        "Port", "Porto"
    )
    datas_diarias = pd.DataFrame(
        pd.date_range(
            dataset_total_embarcado["Dia"].min(),
            dataset_total_embarcado["Dia"].max(),
            freq="D",
        ),
        columns=["Dia"],
    )
    dataset_total_embarcado = pd.merge(
        dataset_total_embarcado, datas_diarias, on="Dia", how="outer"
    )

    return dataset_total_embarcado


def previsao_prophet(
    df: pd.DataFrame,
    data_inicial_real: str,
    data_inicio_projecoes: str,
    data_final_projecoes: str,
    unidade_da_previsao: str,
):

    """Essa funcao faz a previsao de valores semanais futuros por meio de modelos
    prophet.
    """
    df.reset_index(inplace=True)
    base = df
    base = base[["Day", "Port", "Pier",]]

    colunas = list(df.columns)
    colunas.remove("Day")
    colunas.remove("Port")
    colunas.remove("Pier")

    for variavel in colunas:
        prophet_portos = []
        if variavel == "Numero de navios que chegaram":
            for porto_pier in (
                ("Ponta Madeira", "Total"),
                ("Ponta Madeira", "1"),
                ("Ponta Madeira", "3n"),
                ("Ponta Madeira", "3s"),
                ("Ponta Madeira", "4n"),
                ("Ponta Madeira", "4s"),
                ("Guaiba", "Total"),
                ("Sepetiba", "Total"),
                ("Guaiba e Sepetiba", "Total"),
                ("Tubarao", "Total"),
                ("Tubarao", "2"),
                ("Tubarao", "1n"),
                ("Tubarao", "1s"),
            ):
                data_fila = df[
                    (df["Port"] == porto_pier[0]) & (df["Pier"] == porto_pier[1])
                ][["Day", variavel]]
                data_fila.rename(columns={variavel: "y", "Day": "ds"}, inplace=True)

                fila_semana = Prophet()

                fila_semana.fit(
                    data_fila[
                        (data_fila.ds >= data_inicial_real)
                        & (data_fila.ds < data_inicio_projecoes)
                    ]
                )

                future = fila_semana.make_future_dataframe(
                    periods=len(
                        data_fila[
                            (data_fila.ds >= data_inicio_projecoes)
                            & (data_fila.ds <= data_final_projecoes)
                        ]
                    ),
                    freq=unidade_da_previsao,
                )

                forecast = fila_semana.predict(future)
                pred = forecast[["ds", "yhat"]]
                pred.rename(
                    columns={"ds": "Day", "yhat": variavel + " previsao"}, inplace=True
                )
                pred["Port"] = porto_pier[0]
                pred["Pier"] = porto_pier[1]
                pred = pred[pred["Day"] >= data_inicio_projecoes].reset_index(drop=True)
                data_fila = data_fila[data_fila.ds >= data_inicio_projecoes][
                    ["ds"]
                ].reset_index(drop=True)
                pred = pd.merge(
                    pred, data_fila, left_index=True, right_index=True, how="outer"
                )
                del pred["Day"]
                pred.rename(columns={"ds": "Day"}, inplace=True)
                prophet_portos.append(pred)

            df_previsao = pd.concat(
                prophet_portos, ignore_index=False, axis=0, join="outer"
            )

            df = pd.merge(df, df_previsao, on=["Day", "Port", "Pier"], how="outer")
            df = df.dropna(subset=["Port"])

            if variavel != "Multas por dia":
                df[variavel + " previsao"] = np.where(
                    df[variavel + " previsao"] < 0, 0, df[variavel + " previsao"]
                )

            if variavel in ["DISPONIBILIDADE", "UTILIZACAO", "OEE"]:
                df[variavel + " previsao"] = np.where(
                    df[variavel + " previsao"] > 1, 1, df[variavel + " previsao"]
                )

            df[variavel] = np.where(
                df["Day"] >= data_inicio_projecoes,
                df[variavel + " previsao"],
                df[variavel],
            )

            del df[variavel + " previsao"]

        else:
            for porto in (
                "Ponta Madeira",
                "Guaiba",
                "Sepetiba",
                "Guaiba e Sepetiba",
                "Tubarao",
            ):
                data_fila = df[(df["Port"] == porto) & (df["Pier"] == "Total")][
                    ["Day", variavel]
                ]
                data_fila.rename(columns={variavel: "y", "Day": "ds"}, inplace=True)

                fila_semana = Prophet()

                fila_semana.fit(
                    data_fila[
                        (data_fila.ds >= data_inicial_real)
                        & (data_fila.ds < data_inicio_projecoes)
                    ]
                )

                future = fila_semana.make_future_dataframe(
                    periods=len(
                        data_fila[
                            (data_fila.ds >= data_inicio_projecoes)
                            & (data_fila.ds <= data_final_projecoes)
                        ]
                    ),
                    freq=unidade_da_previsao,
                )

                forecast = fila_semana.predict(future)
                pred = forecast[["ds", "yhat"]]
                pred.rename(
                    columns={"ds": "Day", "yhat": variavel + " previsao"}, inplace=True
                )
                pred["Port"] = porto
                pred["Pier"] = "Total"
                pred = pred[pred["Day"] >= data_inicio_projecoes].reset_index(drop=True)
                data_fila = data_fila[data_fila.ds >= data_inicio_projecoes][
                    ["ds"]
                ].reset_index(drop=True)
                pred = pd.merge(
                    pred, data_fila, left_index=True, right_index=True, how="outer"
                )
                del pred["Day"]
                pred.rename(columns={"ds": "Day"}, inplace=True)
                prophet_portos.append(pred)

            df_previsao = pd.concat(
                prophet_portos, ignore_index=False, axis=0, join="outer"
            )
            df = pd.merge(df, df_previsao, on=["Day", "Port", "Pier"], how="outer")
            df = df.dropna(subset=["Port"])

            if variavel != "Multas por dia":
                df[variavel + " previsao"] = np.where(
                    df[variavel + " previsao"] < 0, 0, df[variavel + " previsao"]
                )

            if variavel in ["DISPONIBILIDADE", "UTILIZACAO", "OEE"]:
                df[variavel + " previsao"] = np.where(
                    df[variavel + " previsao"] > 1, 1, df[variavel + " previsao"]
                )

            df[variavel] = np.where(
                df["Day"] >= data_inicio_projecoes,
                df[variavel + " previsao"],
                df[variavel],
            )

            del df[variavel + " previsao"]

    df["Porcentagem FOB"] = (
        df["Numero de navios FOB"]
        / (df["Numero de navios FOB"] + df["Numero de navios CFR"])
    ).fillna(0)

    df["Porcentagem CFR"] = (
        df["Numero de navios CFR"]
        / (df["Numero de navios FOB"] + df["Numero de navios CFR"])
    ).fillna(0)

    df["Porcentagem SPOT/FOB"] = (
        df["Numero de navios SPOT/FOB"]
        / (
            df["Numero de navios SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/FOB"]
            + df["Numero de navios Frota Dedicada"]
        )
    ).fillna(0)

    df["Porcentagem Frota Dedicada/SPOT/FOB"] = (
        df["Numero de navios Frota Dedicada/SPOT/FOB"]
        / (
            df["Numero de navios SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/FOB"]
            + df["Numero de navios Frota Dedicada"]
        )
    ).fillna(0)

    df["Porcentagem Frota Dedicada/FOB"] = (
        df["Numero de navios Frota Dedicada/FOB"]
        / (
            df["Numero de navios SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/FOB"]
            + df["Numero de navios Frota Dedicada"]
        )
    ).fillna(0)

    df["Porcentagem Frota Dedicada"] = (
        df["Numero de navios Frota Dedicada"]
        / (
            df["Numero de navios SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/SPOT/FOB"]
            + df["Numero de navios Frota Dedicada/FOB"]
            + df["Numero de navios Frota Dedicada"]
        )
    ).fillna(0)

    df["Porcentagem PANAMAX"] = (
        df["Numero de navios PANAMAX"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Porcentagem CAPE"] = (
        df["Numero de navios CAPE"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Porcentagem VLOC"] = (
        df["Numero de navios VLOC"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Porcentagem NEWCASTLE"] = (
        df["Numero de navios NEWCASTLE"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Porcentagem VALEMAX"] = (
        df["Numero de navios VALEMAX"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Porcentagem BABYCAPE"] = (
        df["Numero de navios BABYCAPE"]
        / (
            df["Numero de navios PANAMAX"]
            + df["Numero de navios CAPE"]
            + df["Numero de navios VLOC"]
            + df["Numero de navios NEWCASTLE"]
            + df["Numero de navios VALEMAX"]
            + df["Numero de navios BABYCAPE"]
        )
    ).fillna(0)

    df["Quantity (t)"] = np.where(
        df["Quantity (t)"] > df["Dwt (K) total"],
        df["Dwt (K) total"],
        df["Quantity (t)"],
    )
    df["Quantity / Dwt"] = df["Quantity (t)"] / df["Dwt (K) total"]

    df = pd.merge(df, base, on=["Day", "Port", "Pier"])
    df = df.set_index("Day")
    return df


def contador(lista):
    """A funcao abaixo conta a quantidade de dias em que cada dia aparece
    em cada lista.
    """
    return Counter(i for sublist in lista for i in sublist)


class Hiperparametros:
    """Essa classe transforma os valores de hiperparametros salvos em um arquivo
    json em formato de lista.

    Args:

        path_hiperparametros: Caminho com o endereco do arquivo em json que tem
        os valores de hiperparametros obtidos na última execucao do tuning.

    Returns:

        hiperparametros: Hiperparametros obtidos na ultima vez que executamos o
        tuning em formato de lista para ser utilizado nos modelos prophet caso
        nao queiramos executar outro tuning.

    """

    def __init__(self, path_hiperparametros):
        self.path_hiperparametros = path_hiperparametros

    def lista(self):

        with open(self.path_hiperparametros) as json_file:
            hiperparametros = json.load(json_file)

        hiperparametros = hiperparametros.split("],")
        hiperparametros = [sub.replace("[", "") for sub in hiperparametros]
        hiperparametros = [sub.replace("]", "") for sub in hiperparametros]
        hiperparametros = [sub.replace(" ", "") for sub in hiperparametros]
        hiperparametros = [sub.replace('"', "") for sub in hiperparametros]

        for i in range(len(hiperparametros)):
            hiperparametros[i] = hiperparametros[i].split(",")

        for i in range(len(hiperparametros)):
            hiperparametros[i][2] = int(hiperparametros[i][2])
            for j in range(2):
                hiperparametros[i][j] = float(hiperparametros[i][j])

        return [hiperparametros]


class Importacao:
    """A funcao importacao_bases importa todas as bases de dados que utilizamos
    em nossas previsoes de random forest e prophet. Para funcionar, voce precisa
    colocar as bases de dados necessarias nas pastas indicadas em cada caminho.

    Args:

        demurrage_folder: Pasta onde estao todas as bases de demurrage. Dentre outras
        variaveis, as multas de demurrage e o tempo de estadia.

        op_folder: Pasta onde estao todas as bases de operational desk. Essa base
        tem informacoes sobre, dentre outras variaveis, volume embarcado por pier.

        indicadores_pm_folder: Pasta onde estao as bases de indicadores (OEE,
        DISPONIBILIDADE, ...) para o porto de Ponta Madeira.

        indicadores_g_folder: Pasta onde estao as bases de indicadores (OEE,
        DISPONIBILIDADE, ...) para o porto de Guaiba.

        indicadores_s_folder: Pasta onde estao as bases de indicadores (OEE,
        DISPONIBILIDADE, ...) para o porto de Ponta Sepetiba.

        indicadores_t_folder: Pasta onde estao as bases de indicadores (OEE,
        DISPONIBILIDADE, ...) para o porto de Tubarao.

    Returns:

        bases_demurrage: Uniao de todas as bases de demurrage presentes na pasta.

        bases_op: Uniao de todas as bases de operational desk presentes na pasta.

        indices_PM: Uniao de todas as bases de indicadores de Ponta Madeira presentes
        na pasta.

        indices_S: Uniao de todas as bases de indicadores de Sepetiba presentes
        na pasta.

        indices_G: Uniao de todas as bases de indicadores de Guaiba presentes
        na pasta.

        indices_T: Uniao de todas as bases de indicadores de Tubarao presentes
        na pasta.

    """

    def __init__(
        self,
        demurrage_folder,
        op_folder,
        indicadores_pm_folder,
        indicadores_s_folder,
        indicadores_g_folder,
        indicadores_t_folder,
    ):
        self.demurrage_folder = demurrage_folder
        self.op_folder = op_folder
        self.indicadores_pm_folder = indicadores_pm_folder
        self.indicadores_s_folder = indicadores_s_folder
        self.indicadores_g_folder = indicadores_g_folder
        self.indicadores_t_folder = indicadores_t_folder

    def dados(self):

        # Definimos em qual pasta estao os arquivos de demurrage

        os.chdir(self.demurrage_folder)

        bases_demurrage = []

        # Vamos importar todos os arquivos que tenham as extensoes .xlsx, .xls e .csv
        # pois algumas vezes esses arquivos mudam de formato de um mes para outro.

        files = glob.glob("*.xlsx")
        for i in range(len(files)):
            files[i] = self.demurrage_folder + files[i]

        for file in files:
            f = pd.read_excel(file)
            bases_demurrage.append(f)

        files = glob.glob("*.xls")
        for i in range(len(files)):
            files[i] = self.demurrage_folder + files[i]

        for file in files:
            f = pd.read_excel(file)
            bases_demurrage.append(f)

        files = glob.glob("*.csv")
        for i in range(len(files)):
            files[i] = self.demurrage_folder + files[i]

        for file in files:
            f = pd.read_csv(file, sep=",")
            bases_demurrage.append(f)

        # Vamos acrescentar as colunas de número de estadia em horas

        for base in bases_demurrage:
            if "Estadia em Horas" not in list(base.columns):
                base["Estadia em Horas"] = (
                    (base["Estadia Em Dias"].str[:2].astype(int) * 24)
                    + base["Estadia Em Dias"].str[4:6].astype(int)
                    + (base["Estadia Em Dias"].str[4:6].astype(int) / 60)
                )

        # Definimos em qual pasta estao os arquivos de operational desk

        os.chdir(self.op_folder)

        # Vamos importar todos os arquivos que tenham as extensoes .xlsx, .xls e .csv
        # pois algumas vezes esses arquivos mudam de formato de um mes para outro.

        bases_op = []

        files = glob.glob("*.xlsx")
        for i in range(len(files)):
            files[i] = self.op_folder + files[i]

        for file in files:
            f = pd.read_excel(file)
            bases_op.append(f)

        files = glob.glob("*.xls")
        for i in range(len(files)):
            files[i] = self.op_folder + files[i]

        for file in files:
            f = pd.read_excel(file)
            bases_op.append(f)

        files = glob.glob("*.csv")
        for i in range(len(files)):
            files[i] = self.op_folder + files[i]

        for file in files:
            f = pd.read_csv(file, sep=",")
            bases_op.append(f)

        # Agora vamos importar os dados de indicadores separadamente por porto.

        PM = []
        S = []
        G = []
        T = []

        # Os arquivos com os indicadores normalmente vem separados por porto. Logo, devemos
        # colocar os arquivos em pastas separadas por porto.

        for path in (
            (self.indicadores_pm_folder, PM),
            (self.indicadores_s_folder, S),
            (self.indicadores_g_folder, G),
            (self.indicadores_t_folder, T),
        ):
            os.chdir(path[0])

            files = glob.glob("*.xlsx")
            for i in range(len(files)):
                files[i] = path[0] + files[i]

            for file in files:
                f = pd.read_excel(file)
                path[1].append(f)

            files = glob.glob("*.xls")
            for i in range(len(files)):
                files[i] = path[0] + files[i]

            for file in files:
                f = pd.read_excel(file)
                path[1].append(f)

            files = glob.glob("*.csv")
            for i in range(len(files)):
                files[i] = path[0] + files[i]

            for file in files:
                f = pd.read_csv(file, sep=",")
                path[1].append(f)

            for file in path[1]:
                if len(file.columns) == 1:
                    nomes = list(file.columns)[0].split(",")
                    file[nomes] = file[list(file.columns)[0]].str.split(
                        ",", expand=True
                    )
                    file.columns = file.columns.str.replace('"', "")

        if len(PM) > 1:
            indices_PM = pd.concat(PM, axis=0, ignore_index=True)
        else:
            indices_PM = PM[0]

        if len(S) > 1:
            indices_S = pd.concat(S, axis=0, ignore_index=True)
        else:
            indices_S = S[0]

        if len(G) > 1:
            indices_G = pd.concat(G, axis=0, ignore_index=True)
        else:
            indices_G = G[0]

        if len(T) > 1:
            indices_T = pd.concat(T, axis=0, ignore_index=True)
        else:
            indices_T = T[0]

        return [
            bases_demurrage,
            bases_op,
            indices_PM,
            indices_S,
            indices_G,
            indices_T,
        ]


class Datas_Inicio_Fim_Teste:
    """Essa classe define as datas de início do período de teste do nosso modelo
    preditivo. Para tanto, analisamos todas as datas das nossas bases e
    estabelecemos como fim do período de teste a última data para a qual temos
    dados em todas as bases e para a qual temos um mes completo de dados.
    Se a última data cai em um dia no meio do mes, passamos a considerar como
    fim do período de teste o ultimo dia do mes anterior (precisamos dos meses
    completos para fazer o calculo do mape). Se o último dia é o último dia do
    mes, mantemos ele. Os ultimos 6 meses de dados reais sao considerados para
    o calculo do mape.

    Args:

        dataset: Base de demurrage, que tem, dentre outras informacoes, os
        valores pagos de multa por embarque.

        indices1: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1 de Ponta Madeira.

        indices2: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 3n de Ponta Madeira.

        indices3: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 3s de Ponta Madeira.

        indices4: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 4n de Ponta Madeira.

        indices5: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 4s de Ponta Madeira.

        indices6: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao porto Sepetiba.

        indices7: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao porto Guaiba.

        indices8: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 2 de Tubarao.

        indices9: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1n de Tubarao.

        indices10: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1s de Tubarao.

    Returns:

        DATA_INICIO_TESTE: Data inicial da nossa base de teste.

        DATA_FIM_TESTE: Data final da nossa base de teste.

    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        indices1: pd.DataFrame,
        indices2: pd.DataFrame,
        indices3: pd.DataFrame,
        indices4: pd.DataFrame,
        indices5: pd.DataFrame,
        indices6: pd.DataFrame,
        indices7: pd.DataFrame,
        indices8: pd.DataFrame,
        indices9: pd.DataFrame,
        indices10: pd.DataFrame,
    ):
        self.dataset = dataset
        self.indices1 = indices1
        self.indices2 = indices2
        self.indices3 = indices3
        self.indices4 = indices4
        self.indices5 = indices5
        self.indices6 = indices6
        self.indices7 = indices7
        self.indices8 = indices8
        self.indices9 = indices9
        self.indices10 = indices10

    def datas(self):

        lista_datas = []

        self.dataset["TCA"] = (
            pd.to_datetime(self.dataset["TCA"], dayfirst=True)
        ).dt.normalize()

        lista_datas.append(self.dataset["TCA"].unique().max())

        for base in [
            self.indices1,
            self.indices2,
            self.indices3,
            self.indices4,
            self.indices5,
            self.indices6,
            self.indices7,
            self.indices8,
            self.indices9,
            self.indices10,
        ]:
            base["DATA_MIN"] = (
                pd.to_datetime(base["DATA_MIN"], dayfirst=True)
            ).dt.normalize()

        for base in [
            self.indices1,
            self.indices2,
            self.indices3,
            self.indices4,
            self.indices5,
            self.indices6,
            self.indices7,
            self.indices8,
            self.indices9,
            self.indices10,
        ]:
            lista_datas.append(base["DATA_MIN"].unique().max())

        # Se a ultima data for a ultima do mes, vamos menter. Se nao for, vamos utilizar
        # a ultima data da semana anterior.

        if (
            calendar.monthrange(
                pd.to_datetime(min(lista_datas)).year,
                pd.to_datetime(min(lista_datas)).month,
            )[1] - pd.to_datetime(min(lista_datas)).day
            < 7
        ):
            DATA_FIM_TESTE = pd.to_datetime(min(lista_datas))
        else:
            DATA_FIM_TESTE = pd.to_datetime(min(lista_datas)) - pd.offsets.DateOffset(
                months=1
            )
            DATA_FIM_TESTE = DATA_FIM_TESTE.replace(
                day=calendar.monthrange(DATA_FIM_TESTE.year, DATA_FIM_TESTE.month)[1]
            )

        DATA_INICIO_TESTE = (DATA_FIM_TESTE - pd.offsets.DateOffset(days=165)).replace(
            day=1
        )

        return [str(DATA_INICIO_TESTE.date()), str(DATA_FIM_TESTE.date())]


class MergeDemurrageOperationalDesk:
    """A classe MergeDemurrageOperationalDesk tem o objetivo de juntar as bases
    de demurrage e operational desk (OP) em uma só. Precisamos unir as
    informacoes sobre em qual pier ocorreu cada embarque (OP) com as demais
    variáveis da base de demurrage (entre elas, a multa paga por embarque).

    Args:

        dataset: Base de demurrage, que tem, dentre outras informacoes, os
        valores pagos de multa por embarque.

        operational_desk: Base operational desk, que tem, dentre outras
        informacoes, dados sobre tipos de navio e em qual pier o navio atracou.
        Esses dados também estao disponíveis por embarque.

    Returns:

        data_inicial_real: Data de início do tratamento no formato
        YYYY-MM-DD.

        data_final_real: Data de fim do tratamento no formato YYYY-MM-DD.

        data_final_projecoes: Data para o fim das previsoes no formato
        YYYY-MM-DD.

        dataset_g: Base unificada (demurrage e op), contendo apenas os
        dados do porto Guaiba.

        dataset_s: Base unificada (demurrage e op), contendo apenas
        os dados do porto Sepetiba.

        dataset_gs: Base unificada (demurrage e op), contendo
        apenas os dados dos portos Guaiba e Sepetiba conjuntamente.

        dataset_pm: Base unificada (demurrage e op), contendo
        apenas os dados do porto Ponta Madeira.

        dataset_t: Base unificada (demurrage e op), contendo apenas
        os dados do porto Tubarao.

        dataset: Base unificada (demurrage e op).

        dataset_:  Base unificada (demurrage e op), cópia.

        operational_desk: Base operational desk após tratamento.

    Os portos considerados para análise sao:

        Guaiba, Sepetiba, Ponta da Madeira e Tubarao.
        Obs: consieramos Guaiba e Sepetiva tanto separadamente de maneira
        conjunta, como sundo um único porto.

    Os portos de Ponta da Madeira e Tubarao, por sua vez, contam com os
    os respectivos piers:

        Ponta da Madeira: 1, 3n, 3s, 4n, 4s.

        Tubarao: 2, 1n, 1s.
    """

    def __init__(self, dataset: pd.DataFrame, operational_desk: pd.DataFrame):
        self.dataset = dataset
        self.operational_desk = operational_desk

    def resultados(self):
        """Unificamos as bases."""

        self.dataset = self.dataset.reset_index()
        self.operational_desk = self.operational_desk.reset_index()

        # Vamos transformar nossas variaveis de datas em formato datetime

        for var in ["TCA", "ACE", "Data de Desatracação", "Laytime Expired"]:
            self.dataset[var] = (
                pd.to_datetime(self.dataset[var], dayfirst=True)
            ).dt.normalize()

        # Abaixo, extraimos as datas de inicio e fim da base real e acrescentamos
        # a data de fim das projecos, que seria a data de final da base real
        # acrescida de 1 ano e meio.

        self.data_inicial_real = str(min(self.dataset["ACE"]))[
            0:10
        ]  # data de início da base real
        self.data_final_real = str(max(self.dataset["Data de Desatracação"]))[
            0:10
        ]  # data de fim da base real
        self.data_inicio_projecoes = str(
            max(self.dataset["Data de Desatracação"]) + timedelta(days=1)
        )[
            0:10
        ]  # data de inicio das previsoes

        self.ano_final_projecoes = pd.to_datetime(self.data_inicio_projecoes).year + 2
        self.mes_final_projecoes = 12

        self.dia_final_projecoes = calendar.monthrange(
            self.ano_final_projecoes, self.mes_final_projecoes
        )[1]

        self.data_final_calculo = dt.datetime(
            self.ano_final_projecoes, self.mes_final_projecoes, self.dia_final_projecoes
        )

        self.qtd_dias_final_projecoes = (
            self.data_final_calculo - pd.to_datetime(self.data_inicio_projecoes)
        ).days

        self.data_final_projecoes = str(
            max(self.dataset["Data de Desatracação"])
            + timedelta(days=self.qtd_dias_final_projecoes + 1)
        )[
            0:10
        ]  # data de fim das previsoes

        # Abaixo, vamos tirar o acento das palavras, para nao dar erro

        self.dataset["Porto"].replace("Tubarão", "Tubarao", inplace=True)

        # Vamos filtrar as variaveis que nos interessam, que tem informacoes
        # sobre porto e pier.

        self.operational_desk = self.operational_desk[
            self.operational_desk["Origin Port"].isin(
                ["Guaiba", "Ponta da Madeira", "Sepetiba", "Tubarao"]
            )
        ]
        self.operational_desk = self.operational_desk[
            [
                "Shipment Number",
                "Incoterm",
                "Origin Port",
                "Pier",
                "Vessel Class",
                "Quantity (t)",
                "Dwt (K)",
                "ID",
            ]
        ]
        self.operational_desk["PIER"] = (
            self.operational_desk["Origin Port"] + self.operational_desk["Pier"]
        )
        self.operational_desk["PIER"] = self.operational_desk["PIER"].replace(
            {"Guaiba1N": "Guaiba", "Guaiba1S": "Guaiba", "Sepetiba01": "Sepetiba"}
        )

        # Temos que deletar as linhas repetidas

        # 22-09-2023 - Leonard e Gustavo: Tá sumindo com linha de shipment number fazendo com que o cálculo fique incorreto
        self.operational_desk["excedeu_dwt"] = 0
        self.operational_desk = self.operational_desk.reset_index()
        self.dataset = self.dataset.reset_index()

        # Remove as linhas cujas as somas são equivalentes as outras linhas do Shipment Number
        [
            self.remove_duplicatas(x)
            for x in self.operational_desk["Shipment Number"].unique()
        ]

        # Estamos dropando linhas duplicadas devido a um problema no mës de Janeiro da base OP - abril
        self.operational_desk.drop_duplicates(subset=["ID"], keep="first", inplace=True)

        # Iremos realizar a soma dos shipment numbers pois se não somarmos irá atrapalhar a junção com os valores de demurrage
        self.operational_desk = self.operational_desk.groupby(["Shipment Number"]).agg(
            {
                "Shipment Number": "first",
                "Incoterm": "first",
                "Origin Port": "first",
                "Pier": "first",
                "Vessel Class": "first",
                "Quantity (t)": "sum",
                "Dwt (K)": "mean",
                "PIER": "first",
            }
        )

        # Agora vamos mudar alguns nomes de variaveis e modificar os termos da
        # variavel pier para que os nomes fiquem mais simples, referentes aos
        # nomes dos piers.

        self.operational_desk["Shipment Number"] = self.operational_desk[
            "Shipment Number"
        ].str.replace("/", " ")
        self.operational_desk.rename(
            columns={"Shipment Number": "Embarque"}, inplace=True
        )
        self.operational_desk = self.operational_desk[
            ["Embarque", "PIER", "Incoterm", "Vessel Class", "Quantity (t)", "Dwt (K)"]
        ]

        # 26-09-2023 Gustavo e Cassio: Foi encontrado o problema que realiza a soma de valores extras no Volume
        # O problema ocorria quando existiam 2 ou mais linhas de Demurrage para o mesmo Shipment Number (Embarque)

        self.dataset = self.dataset.groupby(["Embarque"]).agg(
            {
                "Porto": "first",
                "Embarque": "first",
                "Sigla Navio": "first",
                "Nome Navio": "first",
                "Número LayDay Report": "first",
                "Data de Desatracação": "first",
                "Dia": "first",
                "Mês": "first",
                "Ano": "first",
                "Período": "first",
                "Data": "first",
                "Valor de Multa ou Prêmio": "sum",
                "Cons": "first",
                "Vend": "first",
                "Comp": "first",
                "Total Embarcado": "first",
                "Data Receb. SOF": "first",
                "Data Validação LDR": "first",
                "Data Envio do LayTime": "first",
                "Data Envio SAP": "first",
                "Data Criação ND/NC no SAP": "first",
                "Valor ND/NC gerado no SAP": "first",
                "Número ND/NC no SAP": "first",
                "Acertador": "first",
                "Analista responsável": "first",
                "Data Confirmação": "first",
                "Recebimento de Debit Note": "first",
                "Data de envio da SP": "first",
                "Date Settlement": "first",
                "Nº Chamado": "first",
                "Nºdias sem emissão ND/NC": "first",
                "Laytime Expired": "first",
                "ACE": "first",
                "Time Allowed DD HH MM": "first",
                "TA D": "first",
                "TA H": "first",
                "TA M": "first",
                "Time Allowed Horas": "first",
                "TCA": "first",
                "Per Day": "first",
                "Estadia Em Dias": "first",
                "ES D": "first",
                "ES H": "first",
                "ES M": "first",
                "Estadia em Horas": "first",
                "Prêmio/Multa em Dias": "first",
                "P/M D": "first",
                "P/M H": "first",
                "P/M M": "first",
                "Premio/Multa em Horas": "first",
            }
        )

        self.dataset.reset_index(drop=True, inplace=True)

        # 22-09-2023 - Gustavo e Leonard: Quando ocorre o merge sem dropar as linhas de volume duplicado podem ocorrer duplicatas de Demurrage (Aumento muito o valor)
        self.dataset = pd.merge(
            self.dataset,
            self.operational_desk[["Embarque", "PIER", "Incoterm", "Vessel Class"]],
            on="Embarque",
            how="left",
        )

        self.dataset["PIER"] = self.dataset["PIER"].map(
            {
                "Guaiba": "Guaiba",
                "Sepetiba": "Sepetiba",
                "Ponta da Madeira1 ": "1",
                "Ponta da Madeira3N": "3n",
                "Ponta da Madeira3S": "3s",
                "Ponta da Madeira4N": "4n",
                "Ponta da Madeira4S": "4s",
                "Tubarao02": "2",
                "Tubarao1N": "1n",
                "Tubarao1S": "1s",
            }
        )

        self.dataset["Porto"] = np.where(
            self.dataset["PIER"] == "Guaiba",
            "Guaiba",
            np.where(
                self.dataset["PIER"] == "Sepetiba",
                "Sepetiba",
                np.where(
                    self.dataset["PIER"] == "1",
                    "Ponta Madeira",
                    np.where(
                        self.dataset["PIER"] == "3n",
                        "Ponta Madeira",
                        np.where(
                            self.dataset["PIER"] == "3s",
                            "Ponta Madeira",
                            np.where(
                                self.dataset["PIER"] == "4n",
                                "Ponta Madeira",
                                np.where(
                                    self.dataset["PIER"] == "4s",
                                    "Ponta Madeira",
                                    np.where(
                                        self.dataset["PIER"] == "2",
                                        "Tubarao",
                                        np.where(
                                            self.dataset["PIER"] == "1n",
                                            "Tubarao",
                                            np.where(
                                                self.dataset["PIER"] == "1s",
                                                "Tubarao",
                                                self.dataset["Porto"],
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        # Agora acrescentamos esses dados de pier a nossa base dataset, que tem
        # os valores de custo de demurrage.

        self.dataset = pd.merge(
            self.dataset,
            pd.get_dummies(self.dataset["PIER"]),
            left_index=True,
            right_index=True,
        )

        for var in ["Guaiba", "1", "3n", "3s", "4n", "4s", "Sepetiba", "2", "1n", "1s"]:
            self.dataset[var] = self.dataset[var].fillna(-1)
            self.dataset[var] = self.dataset[var].astype(int)
            self.dataset[var] = self.dataset[var].replace("-1", np.nan)

        dataset_ = (
            self.dataset
        )  # vamos salvar a base nesse outro objeto para gerar nossa última variável

        # Vamos separar as bases de dados por porto, para ficar mais simples

        dataset_g = self.dataset[self.dataset["Porto"] == "Guaiba"].reset_index(
            drop=True
        )
        dataset_s = self.dataset[self.dataset["Porto"] == "Sepetiba"].reset_index(
            drop=True
        )
        dataset_gs = self.dataset[
            (self.dataset["Porto"] == "Guaiba") | (self.dataset["Porto"] == "Sepetiba")
        ].reset_index(drop=True)
        dataset_gs["Porto"] = "Guaiba e Sepetiba"
        dataset_pm = self.dataset[self.dataset["Porto"] == "Ponta Madeira"].reset_index(
            drop=True
        )
        dataset_t = self.dataset[self.dataset["Porto"] == "Tubarao"].reset_index(
            drop=True
        )

        resultados = [
            self.data_inicial_real,
            self.data_final_real,
            self.data_inicio_projecoes,
            self.data_final_projecoes,
            dataset_g,
            dataset_s,
            dataset_gs,
            dataset_pm,
            dataset_t,
            self.dataset,
            dataset_,
            self.operational_desk,
        ]

        return resultados

    # Função criada para remover algumas linhas que representam a soma de todos os valores de Quantity (t) (isso acontece em alguns poucos casos)
    def remove_duplicatas(self, shipment_number):

        # 1 - Filtramos a base de operation desk para o Shipment Number
        lines = self.operational_desk[
            self.operational_desk["Shipment Number"] == shipment_number
        ]

        # 2 - Guardamos o valor de Dwt para o Shipment
        dwt_shipment = lines["Dwt (K)"].values[0]

        # A quantidade de linhas deve ser maior do que 3 pois anteriormente já foi verificado a combinação linear de 2 registros
        if lines["Quantity (t)"].count() >= 3:

            # 3 - Calculamos qual a quantitdade maxima de combinações entre linhas teremos
            quantidade_combinacoes = lines["Quantity (t)"].count() - 1

            # 4 - Calculamos o valor total de Quantity (t) para o Shipment Number
            soma_total_linhas = lines["Quantity (t)"].sum()

            # 5 - Utilizamos a biblioteca Itertools para gerarmos todas as combinações possiveis de n elementos
            combinacoes = list(
                combinations(lines["Quantity (t)"], quantidade_combinacoes)
            )

            # 6 - Verificamos se a soma de todos os valores de Quantity (t) é maior que o valor de Dwt (k) (isso indica a existência de uma linha adicional que precisa ser removida)
            if soma_total_linhas > dwt_shipment:

                for i in self.operational_desk[
                    self.operational_desk["Shipment Number"] == shipment_number
                ].index:
                    self.operational_desk.at[i, "excedeu_dwt"] = 1

                # 7 - Para cada uma das combinações nós executamos as seguintes regras
                for combinacao in combinacoes:

                    # 7.1 - Soma de todos os valores da combinação
                    soma = sum(combinacao)

                    # 7.2 - caso a soma das linhas fique igual a algum dos outros valores de Quantity (t) significa que encontramos a linha que possui a soma de todos os valores
                    if soma in lines["Quantity (t)"].values:

                        # vamos remover a linha que representa a soma das outras linhas de um Shipment Number
                        self.operational_desk.drop(
                            self.operational_desk[
                                (
                                    (self.operational_desk["Quantity (t)"] == soma)
                                    & (
                                        self.operational_desk["Shipment Number"]
                                        == shipment_number
                                    )
                                )
                            ].index,
                            inplace=True,
                        )


class ContagemNavios:

    """A classe ContagemNavios permite calcular, tanto para os portos quanto para
    os piers, quantos navios chegaram, quantos estao esperando na fila, quantos
    por data TCA, quantos estao carregando, e quantos desatracaram. Todas essas
    informacoes sao consolidadas por dia.

    Inicialmente criamos dicionários com os dias em que se dao cada uma dessas
    ocorrencias e, em sequencia, contabilizamos a quantidade de ocorrencias
    por dia.

    Como em todas as nossas funcoes, essa também é feita tanto a nível dos
    portos quanto a nível dos piers.

    Args:

        dataset_g: Base tratada apenas com os dados de Guaiba, output
        da funcao MergeDemurrageOperationalDesk.

        dataset_s: Base tratada apenas com os dados de Sepetiba,
        output da funcao MergeDemurrageOperationalDesk.

        dataset_gs: Base tratada apenas com os dados de Guaiba
        e Sepetiba, output da funcao MergeDemurrageOperationalDesk.

        dataset_pm: Base tratada apenas com os dados de Ponta
        Madeira, output da funcao MergeDemurrageOperationalDesk.

        dataset_t: Base tratada apenas com os dados de Tubarao,
        output da funcao MergeDemurrageOperationalDesk.

    Returns:

        navios_esperando: 13 bases de dados, com a quantidade de navios
        esperando na fila por dia. Temos como outputs uma base por cada um dos
        portos (Guaiba,Sepetiba,Ponta Madeira,Tubarao, Guaiba+Sepetiba) e uma
        para cada um dos piers (1, 3n, 3s, 4n, 4s, 2, 1n, 1s).

        navios_tca: 13 bases de dados, com a quantidade de navios por data
        TCA por dia. Temos como outputs uma base por cada um dos portos
        (Guaiba,Sepetiba,Ponta Madeira,Tubarao, Guaiba+Sepetiba) e uma para
        cada um dos piers (1, 3n, 3s, 4n, 4s, 2, 1n, 1s).

        navios_carregando: 13 bases de dados, com a quantidade de navios
        carregando o navio, excluindo os dias de atracacao e desatracacao dos
        navios. Temos como outputs uma base por cada um dos portos (Guaiba,
        Sepetiba,Ponta Madeira,Tubarao,Guaiba+Sepetiba) e uma para cada um dos
        piers (1, 3n, 3s, 4n, 4s, 2, 1n, 1s).

        navios_desatracaram: 13 bases de dados, com a quantidade de navios que
        desatracaram por dia. Temos como outputs uma base por cada um dos
        portos (Guaiba,Sepetiba,Ponta Madeira,Tubarao, Guaiba+Sepetiba) e uma
        para  cada um dos piers (1, 3n, 3s, 4n, 4s, 2, 1n, 1s).

        navios_chegaram: 13 bases de dados, com a quantidade de navios que
        chegaram por dia. Temos como outputs uma base por cada um dos portos
        (Guaiba,Sepetiba,Ponta Madeira,Tubarao, Guaiba+Sepetiba) e uma para
        cada um dos piers (1, 3n, 3s, 4n, 4s, 2, 1n, 1s).

    Obs: para mais informacoes sobre algum método específico, veja as infos
    associadas a cada método.
    """

    def __init__(
        self,
        dataset_g: pd.DataFrame,
        dataset_s: pd.DataFrame,
        dataset_gs: pd.DataFrame,
        dataset_pm: pd.DataFrame,
        dataset_t: pd.DataFrame,
    ):

        self.dataset_g = dataset_g
        self.dataset_s = dataset_s
        self.dataset_gs = dataset_gs
        self.dataset_pm = dataset_pm
        self.dataset_t = dataset_t

    def dias_esperando_porto(self, dataframe: pd.DataFrame, lista: list):
        """O método dias_esperando_porto tem como output uma lista com as datas
        em que cada navio ficou esperando no porto para poder atracar.
        """
        for i in range(len(dataframe)):
            lista.append(
                list(
                    pd.date_range(
                        start=dataframe["ACE"].values[i],
                        end=dataframe["TCA"].values[i],
                        closed="left",
                    )
                )
            )
        return [lista]

    def dias_esperando_pier(self, dataframe: pd.DataFrame, lista: list, pier: str):
        """O método dias_esperando_pier tem como output uma lista com as datas
        em que cada navio ficou esperando para poder atracar, mas segmentado
        por pier.
        """
        for i in range(len(dataframe)):
            if dataframe[pier].values[i] == 1:
                lista.append(
                    list(
                        pd.date_range(
                            start=dataframe["ACE"].values[i],
                            end=dataframe["TCA"].values[i],
                            closed="left",
                        )
                    )
                )
        return [lista]

    def dia_tca_porto(self, dataframe: pd.DataFrame, lista: list):
        """O método dia_tca_porto tem como output uma lista com a data de TCA
        correspondente a cada navio em determinado porto.
        """
        for i in range(len(dataframe)):
            lista.append(
                list(
                    pd.date_range(
                        start=dataframe["TCA"].values[i], end=dataframe["TCA"].values[i]
                    )
                )
            )
        return [lista]

    def dia_tca_pier(self, dataframe: pd.DataFrame, lista: list, pier: str):
        """O método dia_tca_pier tem como output uma lista com a data de TCA
        correspondente a cada navio em determinado pier.
        """
        for i in range(len(dataframe)):
            if dataframe[pier].values[i] == 1:
                lista.append(
                    list(
                        pd.date_range(
                            start=dataframe["TCA"].values[i],
                            end=dataframe["TCA"].values[i],
                        )
                    )
                )
        return [lista]

    def dias_carregando_porto(self, dataframe: pd.DataFrame, lista: list):
        """O método dias_carregando_porto tem como output uma lista com os dias
        no qual em que cada navio ficou exclusivamente carregando o navio no
        porto, excluindo as datas de atracacao e desatracacao.
        """
        for i in range(len(dataframe)):
            if pd.Timestamp(dataframe["TCA"].values[i]) == pd.Timestamp(
                dataframe["Data de Desatracação"].values[i]
            ):
                lista.append([])
            else:
                lista.append(
                    list(
                        pd.date_range(
                            start=str(pd.Timestamp(dataframe["TCA"].values[i])),
                            end=(
                                str(
                                    pd.Timestamp(
                                        dataframe["Data de Desatracação"].values[i]
                                    )
                                )
                            ),
                            closed="left",
                        )
                    )
                )
        return [lista]

    def dias_carregando_pier(self, dataframe: pd.DataFrame, lista: list, pier: str):
        """O método dias_carregando_pier tem como output uma lista com os dias
        no qual em que cada navio ficou exclusivamente carregando o navio no
        pier, excluindo as datas de atracacao e desatracacao.
        """
        for i in range(len(dataframe)):
            if dataframe[pier].values[i] == 1:
                if pd.Timestamp(dataframe["TCA"].values[i]) == pd.Timestamp(
                    dataframe["Data de Desatracação"].values[i]
                ):
                    lista.append([])
                else:
                    lista.append(
                        list(
                            pd.date_range(
                                start=str(pd.Timestamp(dataframe["TCA"].values[i])),
                                end=(
                                    str(
                                        pd.Timestamp(
                                            dataframe["Data de Desatracação"].values[i]
                                        )
                                    )
                                ),
                                closed="left",
                            )
                        )
                    )
        return [lista]

    def dia_desatracou_porto(self, dataframe: pd.DataFrame, lista: list):
        """O método dia_desatracou_porto tem como output uma lista com o dia
        no qual cada navio desatracou por porto.
        """
        for i in range(len(dataframe)):
            lista.append(
                list(
                    pd.date_range(
                        start=dataframe["Data de Desatracação"].values[i],
                        end=dataframe["Data de Desatracação"].values[i],
                    )
                )
            )
        return [lista]

    def dia_desatracou_pier(self, dataframe: pd.DataFrame, lista: list, pier: str):
        """O método dia_desatracou_pier tem como output uma lista com o dia
        no qual cada navio desatracou por pier.
        """
        for i in range(len(dataframe)):
            if dataframe[pier].values[i] == 1:
                lista.append(
                    list(
                        pd.date_range(
                            start=dataframe["Data de Desatracação"].values[i],
                            end=dataframe["Data de Desatracação"].values[i],
                        )
                    )
                )
        return [lista]

    def dia_chegou_porto(self, dataframe: pd.DataFrame, lista: list):
        """O método dia_chegou_porto tem como output uma lista com o dia
        no qual cada navio chegou por porto.
        """
        for i in range(len(dataframe)):
            lista.append(
                list(
                    pd.date_range(
                        start=dataframe["ACE"].values[i], end=dataframe["ACE"].values[i]
                    )
                )
            )
        return [lista]

    def dia_chegou_pier(self, dataframe: pd.DataFrame, lista: list, pier: str):
        """O método dia_chegou_pier tem como output uma lista com o dia
        no qual cada navio chegou por pier.
        """
        for i in range(len(dataframe)):
            if dataframe[pier].values[i] == 1:
                lista.append(
                    list(
                        pd.date_range(
                            start=dataframe["ACE"].values[i],
                            end=dataframe["ACE"].values[i],
                        )
                    )
                )
        return [lista]

    def navios_esperando(self):
        """O método navios_esperando utiliza as listas de datas geradas pelos
        métodos dias_esperando_porto e dias_esperando_pier e contabiliza
        quantas vezes cada data aparece. Esse cálculo é feito para cada porto
        e pier.
        """
        datas_fila_g = []
        datas_fila_s = []
        datas_fila_gs = []
        datas_fila_pm = []
        datas_fila_t = []
        datas_fila_pm1 = []
        datas_fila_pm3n = []
        datas_fila_pm3s = []
        datas_fila_pm4n = []
        datas_fila_pm4s = []
        datas_fila_t2 = []
        datas_fila_t1n = []
        datas_fila_t1s = []

        # Criadas as listas, vamos gerar uma lista com os dias nos quais os
        # navios permaneceram esperando na fila. Tanto por porto quanto por pier.

        for var in (
            (self.dataset_g, datas_fila_g),
            (self.dataset_s, datas_fila_s),
            (self.dataset_gs, datas_fila_gs),
            (self.dataset_pm, datas_fila_pm),
            (self.dataset_t, datas_fila_t),
        ):
            self.dias_esperando_porto(var[0], var[1])

        for var in (
            (self.dataset_pm, datas_fila_pm1, "1"),
            (self.dataset_pm, datas_fila_pm3n, "3n"),
            (self.dataset_pm, datas_fila_pm3s, "3s"),
            (self.dataset_pm, datas_fila_pm4n, "4n"),
            (self.dataset_pm, datas_fila_pm4s, "4s"),
            (self.dataset_t, datas_fila_t2, "2"),
            (self.dataset_t, datas_fila_t1n, "1n"),
            (self.dataset_t, datas_fila_t1s, "1s"),
        ):
            self.dias_esperando_pier(var[0], var[1], var[2])

        # Abaixo, fazemos a contagem de quantos navios permaneceram na fila por
        # dia. Tanto para portos quanto piers.

        fila_g = contador(datas_fila_g)
        fila_s = contador(datas_fila_s)
        fila_gs = contador(datas_fila_gs)
        fila_pm = contador(datas_fila_pm)
        fila_pm1 = contador(datas_fila_pm1)
        fila_pm3n = contador(datas_fila_pm3n)
        fila_pm3s = contador(datas_fila_pm3s)
        fila_pm4n = contador(datas_fila_pm4n)
        fila_pm4s = contador(datas_fila_pm4s)
        fila_t = contador(datas_fila_t)
        fila_t2 = contador(datas_fila_t2)
        fila_t1n = contador(datas_fila_t1n)
        fila_t1s = contador(datas_fila_t1s)

        return [
            fila_g,
            fila_s,
            fila_gs,
            fila_pm,
            fila_pm1,
            fila_pm3n,
            fila_pm3s,
            fila_pm4n,
            fila_pm4s,
            fila_t,
            fila_t2,
            fila_t1n,
            fila_t1s,
        ]

    def navios_tca(self):
        """O método navios_tca utiliza as listas de datas geradas pelos métodos
        dia_tca_porto e dia_tca_pier e contabiliza quantas vezes cada
        data aparece. Esse cálculo é feito para cada porto e pier.
        """
        datas_tca_g = []
        datas_tca_s = []
        datas_tca_gs = []
        datas_tca_pm = []
        datas_tca_t = []

        # Criadas as listas, vamos gerar uma lista com os dias nos quais os
        # navios por data TCA. Tanto por porto quanto por pier.

        for var in (
            (self.dataset_g, datas_tca_g),
            (self.dataset_s, datas_tca_s),
            (self.dataset_gs, datas_tca_gs),
            (self.dataset_pm, datas_tca_pm),
            (self.dataset_t, datas_tca_t),
        ):
            self.dia_tca_porto(var[0], var[1])

        datas_tca_pm1 = []
        datas_tca_pm3n = []
        datas_tca_pm3s = []
        datas_tca_pm4n = []
        datas_tca_pm4s = []
        datas_tca_t2 = []
        datas_tca_t1n = []
        datas_tca_t1s = []

        for var in (
            (self.dataset_pm, datas_tca_pm1, "1"),
            (self.dataset_pm, datas_tca_pm3n, "3n"),
            (self.dataset_pm, datas_tca_pm3s, "3s"),
            (self.dataset_pm, datas_tca_pm4n, "4n"),
            (self.dataset_pm, datas_tca_pm4s, "4s"),
            (self.dataset_t, datas_tca_t2, "2"),
            (self.dataset_t, datas_tca_t1n, "1n"),
            (self.dataset_t, datas_tca_t1s, "1s"),
        ):
            self.dia_tca_pier(var[0], var[1], var[2])

        # Abaixo, fazemos a contagem de quantos navios por data TCA por
        # dia. Tanto para portos quanto piers.

        tca_g = contador(datas_tca_g)
        tca_s = contador(datas_tca_s)
        tca_gs = contador(datas_tca_gs)
        tca_pm = contador(datas_tca_pm)
        tca_pm1 = contador(datas_tca_pm1)
        tca_pm3n = contador(datas_tca_pm3n)
        tca_pm3s = contador(datas_tca_pm3s)
        tca_pm4n = contador(datas_tca_pm4n)
        tca_pm4s = contador(datas_tca_pm4s)
        tca_t = contador(datas_tca_t)
        tca_t2 = contador(datas_tca_t2)
        tca_t1n = contador(datas_tca_t1n)
        tca_t1s = contador(datas_tca_t1s)

        return [
            tca_g,
            tca_s,
            tca_gs,
            tca_pm,
            tca_pm1,
            tca_pm3n,
            tca_pm3s,
            tca_pm4n,
            tca_pm4s,
            tca_t,
            tca_t2,
            tca_t1n,
            tca_t1s,
        ]

    def navios_carregando(self):
        """O método navios_carregando utiliza as listas de datas geradas pelos métodos
        dias_carregando_porto e dias_carregando_pier e contabiliza quantas vezes cada
        data aparece. Esse cálculo é feito para cada porto e pier.
        """
        datas_carregando_g = []
        datas_carregando_s = []
        datas_carregando_gs = []
        datas_carregando_pm = []
        datas_carregando_t = []

        # Criadas as listas, vamos gerar uma lista com os dias nos quais os
        # navios chegaram. Tanto por porto quanto por pier.

        for var in (
            (self.dataset_g, datas_carregando_g),
            (self.dataset_s, datas_carregando_s),
            (self.dataset_gs, datas_carregando_gs),
            (self.dataset_pm, datas_carregando_pm),
            (self.dataset_t, datas_carregando_t),
        ):
            self.dias_carregando_porto(var[0], var[1])

        datas_carregando_pm1 = []
        datas_carregando_pm3n = []
        datas_carregando_pm3s = []
        datas_carregando_pm4n = []
        datas_carregando_pm4s = []
        datas_carregando_t2 = []
        datas_carregando_t1n = []
        datas_carregando_t1s = []

        for var in (
            (self.dataset_pm, datas_carregando_pm1, "1"),
            (self.dataset_pm, datas_carregando_pm3n, "3n",),
            (self.dataset_pm, datas_carregando_pm3s, "3s",),
            (self.dataset_pm, datas_carregando_pm4n, "4n",),
            (self.dataset_pm, datas_carregando_pm4s, "4s",),
            (self.dataset_t, datas_carregando_t2, "2"),
            (self.dataset_t, datas_carregando_t1n, "1n"),
            (self.dataset_t, datas_carregando_t1s, "1s"),
        ):
            self.dias_carregando_pier(var[0], var[1], var[2])

        # Abaixo, fazemos a contagem de quantos navios permaneceram carregando por
        # dia. Tanto para portos quanto piers.

        carregando_g = contador(datas_carregando_g)
        carregando_s = contador(datas_carregando_s)
        carregando_gs = contador(datas_carregando_gs)
        carregando_pm = contador(datas_carregando_pm)
        carregando_pm1 = contador(datas_carregando_pm1)
        carregando_pm3n = contador(datas_carregando_pm3n)
        carregando_pm3s = contador(datas_carregando_pm3s)
        carregando_pm4n = contador(datas_carregando_pm4n)
        carregando_pm4s = contador(datas_carregando_pm4s)
        carregando_t = contador(datas_carregando_t)
        carregando_t2 = contador(datas_carregando_t2)
        carregando_t1n = contador(datas_carregando_t1n)
        carregando_t1s = contador(datas_carregando_t1s)

        return [
            carregando_g,
            carregando_s,
            carregando_gs,
            carregando_pm,
            carregando_pm1,
            carregando_pm3n,
            carregando_pm3s,
            carregando_pm4n,
            carregando_pm4s,
            carregando_t,
            carregando_t2,
            carregando_t1n,
            carregando_t1s,
        ]

    def navios_desatracaram(self):
        """O método navios_desatracaram utiliza as listas de datas geradas pelos métodos
        dias_desatracou_porto e dias_desatracou_pier e contabiliza quantas vezes cada
        data aparece. Esse cálculo é feito para cada porto e pier.
        """
        datas_desatracou_g = []
        datas_desatracou_s = []
        datas_desatracou_gs = []
        datas_desatracou_pm = []
        datas_desatracou_t = []

        # Criadas as listas, vamos gerar uma lista com os dias nos quais os
        # navios desatracaram. Tanto por porto quanto por pier.

        for var in (
            (self.dataset_g, datas_desatracou_g),
            (self.dataset_s, datas_desatracou_s),
            (self.dataset_gs, datas_desatracou_gs),
            (self.dataset_pm, datas_desatracou_pm),
            (self.dataset_t, datas_desatracou_t),
        ):
            self.dia_desatracou_porto(var[0], var[1])

        datas_desatracou_pm1 = []
        datas_desatracou_pm3n = []
        datas_desatracou_pm3s = []
        datas_desatracou_pm4n = []
        datas_desatracou_pm4s = []
        datas_desatracou_t2 = []
        datas_desatracou_t1n = []
        datas_desatracou_t1s = []

        for var in (
            (self.dataset_pm, datas_desatracou_pm1, "1"),
            (self.dataset_pm, datas_desatracou_pm3n, "3n",),
            (self.dataset_pm, datas_desatracou_pm3s, "3s",),
            (self.dataset_pm, datas_desatracou_pm4n, "4n",),
            (self.dataset_pm, datas_desatracou_pm4s, "4s",),
            (self.dataset_t, datas_desatracou_t2, "2"),
            (self.dataset_t, datas_desatracou_t1n, "1n"),
            (self.dataset_t, datas_desatracou_t1s, "1s"),
        ):
            self.dia_desatracou_pier(var[0], var[1], var[2])

        # Abaixo, fazemos a contagem de quantos navios desatracaram por
        # dia. Tanto para portos quanto piers.

        desatracando_g = contador(datas_desatracou_g)
        desatracando_s = contador(datas_desatracou_s)
        desatracando_gs = contador(datas_desatracou_gs)
        desatracando_pm = contador(datas_desatracou_pm)
        desatracando_pm1 = contador(datas_desatracou_pm1)
        desatracando_pm3n = contador(datas_desatracou_pm3n)
        desatracando_pm3s = contador(datas_desatracou_pm3s)
        desatracando_pm4n = contador(datas_desatracou_pm4n)
        desatracando_pm4s = contador(datas_desatracou_pm4s)
        desatracando_t = contador(datas_desatracou_t)
        desatracando_t2 = contador(datas_desatracou_t2)
        desatracando_t1n = contador(datas_desatracou_t1n)
        desatracando_t1s = contador(datas_desatracou_t1s)

        return [
            desatracando_g,
            desatracando_s,
            desatracando_gs,
            desatracando_pm,
            desatracando_pm1,
            desatracando_pm3n,
            desatracando_pm3s,
            desatracando_pm4n,
            desatracando_pm4s,
            desatracando_t,
            desatracando_t2,
            desatracando_t1n,
            desatracando_t1s,
        ]

    def navios_chegaram(self):
        """O método navios_chegaram utiliza as listas de datas geradas pelos métodos
        dias_chegou_porto e dias_chegou_pier e contabiliza quantas vezes cada
        data aparece. Esse cálculo é feito para cada porto e pier.
        """
        datas_chegou_g = []
        datas_chegou_s = []
        datas_chegou_gs = []
        datas_chegou_pm = []
        datas_chegou_t = []

        # Criadas as listas, vamos gerar uma lista com os dias nos quais os
        # navios chegaram. Tanto por porto quanto por pier.

        for var in (
            (self.dataset_g, datas_chegou_g),
            (self.dataset_s, datas_chegou_s),
            (self.dataset_gs, datas_chegou_gs),
            (self.dataset_pm, datas_chegou_pm),
            (self.dataset_t, datas_chegou_t),
        ):
            self.dia_chegou_porto(var[0], var[1])

        datas_chegou_pm1 = []
        datas_chegou_pm3n = []
        datas_chegou_pm3s = []
        datas_chegou_pm4n = []
        datas_chegou_pm4s = []
        datas_chegou_t2 = []
        datas_chegou_t1n = []
        datas_chegou_t1s = []

        for var in (
            (self.dataset_pm, datas_chegou_pm1, "1"),
            (self.dataset_pm, datas_chegou_pm3n, "3n"),
            (self.dataset_pm, datas_chegou_pm3s, "3s"),
            (self.dataset_pm, datas_chegou_pm4n, "4n"),
            (self.dataset_pm, datas_chegou_pm4s, "4s"),
            (self.dataset_t, datas_chegou_t2, "2"),
            (self.dataset_t, datas_chegou_t1n, "1n"),
            (self.dataset_t, datas_chegou_t1s, "1s"),
        ):
            self.dia_chegou_pier(var[0], var[1], var[2])

        # Abaixo, fazemos a contagem de quantos navios chegaram por
        # dia. Tanto para portos quanto piers.

        chegaram_g = contador(datas_chegou_g)
        chegaram_s = contador(datas_chegou_s)
        chegaram_gs = contador(datas_chegou_gs)
        chegaram_pm = contador(datas_chegou_pm)
        chegaram_pm1 = contador(datas_chegou_pm1)
        chegaram_pm3n = contador(datas_chegou_pm3n)
        chegaram_pm3s = contador(datas_chegou_pm3s)
        chegaram_pm4n = contador(datas_chegou_pm4n)
        chegaram_pm4s = contador(datas_chegou_pm4s)
        chegaram_t = contador(datas_chegou_t)
        chegaram_t2 = contador(datas_chegou_t2)
        chegaram_t1n = contador(datas_chegou_t1n)
        chegaram_t1s = contador(datas_chegou_t1s)

        return [
            chegaram_g,
            chegaram_s,
            chegaram_gs,
            chegaram_pm,
            chegaram_pm1,
            chegaram_pm3n,
            chegaram_pm3s,
            chegaram_pm4n,
            chegaram_pm4s,
            chegaram_t,
            chegaram_t2,
            chegaram_t1n,
            chegaram_t1s,
        ]


class NaviosQuantPorcentagem:
    """
    A classe NaviosQuantPorcentagem calcula quantidade e porcentagem de navios
    que se enquadram em cada tipo de embarcacao e em tipo de contrato
    (FOB ou CFR).

    Os tipos considerados sao:

        CFR, FOB, Panamax, Cape, Vloc, Newcastle, Valemax, Babycape,
        Frota Dedicada, SPOT/FOB, Frota Dedicada SPOT/FOB e Frota dedicada FOB.

    Args:

        dataset:  Base unificada entre as bases de demurrage e op. Essa
        base é output da class MergeDemurrageOperatinalDesk. Para mais
        detalhes, veja os comentários da referida classe.

    Returns:

        dataset: Base dataset com tratamentos adicionais modificando os
        nomes dos tipos de navios na variável 'Vessel group'.

        dataset_fob: Base de dados com quantidade e porcentagem de navios
        por tipo de contrato (FOB e CFR).

        dataset_grupos: Base de dados com quantidade e porcentagem de
        navios por tipo de navios Panamax, Cape, VLOC, Newcastle, Valemax,
        Babycape.

        dataset_grupos2: Base de dados com quantidade e porcentagem de
        navios por tipo de navios SPOT/FOB, Frota Dedicada/SPOT/FOB,
        Frota Dedicada/FOB,  Frota Dedicada.
    """

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def resultado(self):
        """Temos entao as informacoes de porcentagem e quantidade de navios
        por tipo.
        """

        # Vamos criar variaveis que filtrem os portos de Ponta Madeira e Tubarao
        # cada um como uma unica unidade quanto para segmentar seus piers de maneira
        # individual.

        self.dataset.rename(columns={"PIER": "Pier"}, inplace=True)
        dataset_pmt = self.dataset[
            (self.dataset["Porto"] == "Ponta Madeira")
            | (self.dataset["Porto"] == "Tubarao")
        ]
        dataset_pmt["Pier"] = "Total"
        self.dataset = pd.concat(
            [self.dataset, dataset_pmt], ignore_index=False, axis=0
        )
        self.dataset["Pier"] = np.where(
            self.dataset["Porto"] == "Guaiba",
            "Total",
            np.where(
                self.dataset["Porto"] == "Sepetiba", "Total", self.dataset["Pier"]
            ),
        )
        dataset_gs = self.dataset[
            (self.dataset["Porto"] == "Guaiba") | (self.dataset["Porto"] == "Sepetiba")
        ]
        dataset_gs["Porto"] = "Guaiba e Sepetiba"
        self.dataset = pd.concat([self.dataset, dataset_gs], ignore_index=False, axis=0)

        # Abaixo, calculamos quantidade e porcentagem de navios com tipos de contrato
        # FOB e CFR por dia, com base na data de chegada.

        def grupo_fob(dataset: pd.DataFrame):

            grupo = pd.DataFrame(
                dataset[dataset["Incoterm"] == "FOB"][
                    ["ACE", "Porto", "Pier", "Incoterm"]
                ]
                .dropna()
                .groupby(["ACE", "Porto", "Pier", "Incoterm"])
                .size()
            ).reset_index()

            grupo.rename(columns={0: "Numero de navios FOB"}, inplace=True)

            grupo_ = pd.DataFrame(
                dataset[
                    (dataset["Incoterm"] == "CFR") | (dataset["Incoterm"] == "FOB")
                ][["ACE", "Porto", "Pier", "Incoterm"]]
                .dropna()
                .groupby(["ACE", "Porto", "Pier"])
                .size()
            ).reset_index()

            grupo_.rename(columns={0: "Navios totais"}, inplace=True)

            grupo = pd.merge(grupo, grupo_, on=["ACE", "Porto", "Pier"], how="left")

            grupo = grupo[grupo["Incoterm"] == "FOB"]

            grupo = grupo[
                ["ACE", "Porto", "Pier", "Numero de navios FOB", "Navios totais"]
            ]

            grupo["Numero de navios CFR"] = (
                grupo["Navios totais"] - grupo["Numero de navios FOB"]
            )

            grupo["Porcentagem FOB"] = (
                grupo["Numero de navios FOB"] / grupo["Navios totais"]
            )

            grupo["Porcentagem CFR"] = 1 - grupo["Porcentagem FOB"]

            grupo = grupo[
                [
                    "ACE",
                    "Porto",
                    "Pier",
                    "Numero de navios FOB",
                    "Numero de navios CFR",
                    "Porcentagem FOB",
                    "Porcentagem CFR",
                ]
            ]
            return grupo

        dataset_fob = grupo_fob(self.dataset)

        # Abaixo criamos uma variavel que segmenta os navios por tipo.

        self.dataset["Vessel group"] = 0
        self.dataset["Vessel group"] = self.dataset["Vessel Class"].map(
            {
                "Panamax": "Panamax",
                "Capesize": "Cape",
                "VLOC": "VLOC",
                "Newcastlemax": "Newcastle",
                "PostPanamax": "Panamax",
                "Valemax G1": "Valemax",
                "Handysize": "Panamax",
                "Babycape": "Babycape",
                "Valemax G2": "Valemax",
                "Handymax": "Panamax",
                np.nan: 0,
                0: 0,
                "Barge": np.nan,
                "Guaibamax": "VLOC",
            }
        )

        # Em sequencia, calculamos quantidade e porcentagem de navios de cada um
        # dos tipos especificados acima.
        # Primeiro calculamos apenas a quantidade de navios por tipo e a quantidade
        # total.

        def grupo_tipo1(dataset: pd.DataFrame):

            grupo = pd.DataFrame(
                dataset[["ACE", "Porto", "Pier", "Vessel group"]]
                .dropna()
                .groupby(["ACE", "Vessel group", "Porto", "Pier"])
                .size()
            ).reset_index()

            grupo.rename(columns={0: "Numero de navios"}, inplace=True)

            grupo_ = pd.DataFrame(
                dataset[["ACE", "Porto", "Pier", "Vessel group"]]
                .dropna()
                .groupby(["ACE", "Porto", "Pier"])
                .size()
            ).reset_index()

            grupo_.rename(columns={0: "Navios totais"}, inplace=True)

            grupo = pd.merge(grupo, grupo_, on=["ACE", "Porto", "Pier"], how="left")

            return grupo

        group = grupo_tipo1(self.dataset)

        # Utilizando os dados de quantidade de navios totais e por tipo, podemos
        # acrscentar também a variavel porcentagem, por tipo, calculada abaixo.

        def nomes_navios(group: pd.DataFrame):

            panamax = group[group["Vessel group"] == "Panamax"]
            cape = group[group["Vessel group"] == "Cape"]
            vloc = group[group["Vessel group"] == "VLOC"]
            newcastle = group[group["Vessel group"] == "Newcastle"]
            valemax = group[group["Vessel group"] == "Valemax"]
            babycape = group[group["Vessel group"] == "Babycape"]

            panamax = panamax[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            cape = cape[["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]]
            vloc = vloc[["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]]
            newcastle = newcastle[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            valemax = valemax[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            babycape = babycape[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]

            panamax["Porcentagem PANAMAX"] = (
                panamax["Numero de navios"] / panamax["Navios totais"]
            )
            cape["Porcentagem CAPE"] = cape["Numero de navios"] / cape["Navios totais"]
            vloc["Porcentagem VLOC"] = vloc["Numero de navios"] / vloc["Navios totais"]
            newcastle["Porcentagem NEWCASTLE"] = (
                newcastle["Numero de navios"] / newcastle["Navios totais"]
            )
            valemax["Porcentagem VALEMAX"] = (
                valemax["Numero de navios"] / valemax["Navios totais"]
            )
            babycape["Porcentagem BABYCAPE"] = (
                babycape["Numero de navios"] / babycape["Navios totais"]
            )

            panamax = panamax[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem PANAMAX"]
            ]
            cape = cape[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem CAPE"]
            ]
            vloc = vloc[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem VLOC"]
            ]
            newcastle = newcastle[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem NEWCASTLE"]
            ]
            valemax = valemax[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem VALEMAX"]
            ]
            babycape = babycape[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem BABYCAPE"]
            ]

            panamax.rename(
                columns={"Numero de navios": "Numero de navios PANAMAX"}, inplace=True
            )
            cape.rename(
                columns={"Numero de navios": "Numero de navios CAPE"}, inplace=True
            )
            vloc.rename(
                columns={"Numero de navios": "Numero de navios VLOC"}, inplace=True
            )
            newcastle.rename(
                columns={"Numero de navios": "Numero de navios NEWCASTLE"}, inplace=True
            )
            valemax.rename(
                columns={"Numero de navios": "Numero de navios VALEMAX"}, inplace=True
            )
            babycape.rename(
                columns={"Numero de navios": "Numero de navios BABYCAPE"}, inplace=True
            )

            return panamax, cape, vloc, newcastle, valemax, babycape

        panamax, cape, vloc, newcastle, valemax, babycape = nomes_navios(group)

        # Vamos unificar todos esses dados por tipo de navios em uma unica base.

        dfs = [panamax, cape, vloc, newcastle, valemax, babycape]

        dataset_grupos = reduce(
            lambda left, right: pd.merge(
                left, right, on=["ACE", "Porto", "Pier"], how="outer"
            ),
            dfs,
        )

        dataset_grupos.fillna(0)

        # Vamos agora repetir a contagem de navios por tipo e calcular suas porcentagens
        # com base em outra forma de agrupamento dos navios, com base nas categorias
        # descritas abaixo.

        self.dataset["Contratação"] = 0
        self.dataset["Contratação"] = self.dataset["Vessel Class"].map(
            {
                "Panamax": "SPOT/FOB",
                "Capesize": "Frota Dedicada/SPOT/FOB",
                "VLOC": "Frota Dedicada/FOB",
                "Newcastlemax": "Frota Dedicada/SPOT/FOB",
                "PostPanamax": "SPOT/FOB",
                "Valemax G1": "Frota Dedicada",
                "Handysize": "SPOT/FOB",
                "Babycape": "SPOT/FOB",
                "Valemax G2": "Frota Dedicada",
                "Handymax": "SPOT/FOB",
                np.nan: 0,
                0: 0,
                "Barge": np.nan,
                "Guaibamax": "Frota Dedicada/FOB",
            }
        )

        # Calculamos a quantidade de navios por tipo e a quantidade de navios
        # totais.

        def grupo_tipo2(dataframe: pd.DataFrame):

            grupo = pd.DataFrame(
                dataframe[["ACE", "Porto", "Pier", "Contratação"]]
                .dropna()
                .groupby(["ACE", "Porto", "Pier", "Contratação"])
                .size()
            ).reset_index()
            grupo.rename(columns={0: "Numero de navios"}, inplace=True)
            grupo_ = pd.DataFrame(
                dataframe[["ACE", "Porto", "Pier", "Contratação"]]
                .dropna()
                .groupby(["ACE", "Porto", "Pier"])
                .size()
            ).reset_index()
            grupo_.rename(columns={0: "Navios totais"}, inplace=True)
            grupo = pd.merge(grupo, grupo_, on=["ACE", "Porto", "Pier"], how="left")

            return grupo

        group2 = grupo_tipo2(self.dataset)

        # Com base nos dados anteriores, calculamos a porcentagem de navios e
        # acrescentamos a base.

        def nomes_navios2(grupo: pd.DataFrame):
            spot_fob = grupo[grupo["Contratação"] == "SPOT/FOB"]
            frota_dedicada_spot_fob = grupo[
                grupo["Contratação"] == "Frota Dedicada/SPOT/FOB"
            ]
            frota_dedicada_fob = grupo[grupo["Contratação"] == "Frota Dedicada/FOB"]
            frota_dedicada = grupo[grupo["Contratação"] == "Frota Dedicada"]

            spot_fob = spot_fob[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            frota_dedicada_spot_fob = frota_dedicada_spot_fob[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            frota_dedicada_fob = frota_dedicada_fob[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]
            frota_dedicada = frota_dedicada[
                ["ACE", "Porto", "Pier", "Numero de navios", "Navios totais"]
            ]

            spot_fob["Porcentagem SPOT/FOB"] = (
                spot_fob["Numero de navios"] / spot_fob["Navios totais"]
            )
            frota_dedicada_spot_fob["Porcentagem Frota Dedicada/SPOT/FOB"] = (
                frota_dedicada_spot_fob["Numero de navios"]
                / frota_dedicada_spot_fob["Navios totais"]
            )
            frota_dedicada_fob["Porcentagem Frota Dedicada/FOB"] = (
                frota_dedicada_fob["Numero de navios"]
                / frota_dedicada_fob["Navios totais"]
            )
            frota_dedicada["Porcentagem Frota Dedicada"] = (
                frota_dedicada["Numero de navios"] / frota_dedicada["Navios totais"]
            )

            spot_fob = spot_fob[
                ["ACE", "Porto", "Pier", "Numero de navios", "Porcentagem SPOT/FOB"]
            ]
            frota_dedicada_spot_fob = frota_dedicada_spot_fob[
                [
                    "ACE",
                    "Porto",
                    "Pier",
                    "Numero de navios",
                    "Porcentagem Frota Dedicada/SPOT/FOB",
                ]
            ]
            frota_dedicada_fob = frota_dedicada_fob[
                [
                    "ACE",
                    "Porto",
                    "Pier",
                    "Numero de navios",
                    "Porcentagem Frota Dedicada/FOB",
                ]
            ]
            frota_dedicada = frota_dedicada[
                [
                    "ACE",
                    "Porto",
                    "Pier",
                    "Numero de navios",
                    "Porcentagem Frota Dedicada",
                ]
            ]

            spot_fob.rename(
                columns={"Numero de navios": "Numero de navios SPOT/FOB"}, inplace=True
            )
            frota_dedicada_spot_fob.rename(
                columns={
                    "Numero de navios": "Numero de navios Frota Dedicada/SPOT/FOB"
                },
                inplace=True,
            )
            frota_dedicada_fob.rename(
                columns={"Numero de navios": "Numero de navios Frota Dedicada/FOB"},
                inplace=True,
            )
            frota_dedicada.rename(
                columns={"Numero de navios": "Numero de navios Frota Dedicada"},
                inplace=True,
            )

            return [
                spot_fob,
                frota_dedicada_spot_fob,
                frota_dedicada_fob,
                frota_dedicada,
            ]

        resultado = nomes_navios2(group2)

        # Vamos, por fim, juntar essas novas variaveis de quantidade de navios
        # em uma unica base.

        spot_fob, frota_dedicada_spot_fob, frota_dedicada_fob, frota_dedicada = (
            resultado[0],
            resultado[1],
            resultado[2],
            resultado[3],
        )

        dfs = [spot_fob, frota_dedicada_spot_fob, frota_dedicada_fob, frota_dedicada]

        dataset_grupos2 = reduce(
            lambda left, right: pd.merge(
                left, right, on=["ACE", "Porto", "Pier"], how="outer"
            ),
            dfs,
        )

        dataset_grupos2.fillna(0)

        return [self.dataset, dataset_fob, dataset_grupos, dataset_grupos2]


class QuantidadeCapacidade:
    """A classe QuantidadeCapacidade calcula a quantidade embarcada por dia e a
    capacidade dos navios que embarcaram por dia, tanto para os portos quanto
    para os piers.

    Args:

        dataset: Base de dados output da classe NaviosQuantPorcentagem.

        operational_desk: Base de dados output da classe MergeDemurrageOperationalDesk.

    Results:

        dataset_quantidade_capacidade: Base de dados com duas variáveis principais:
        soma da quantidade embarcada de todos os navios com base em suas datas de chegada
        e a soma da capacidade de todos os navios com base em suas datas de chegada.
        Tipo dataframe.
    """

    def __init__(self, dataset_: pd.DataFrame, operational_desk: pd.DataFrame):
        self.dataset_ = dataset_
        self.operational_desk = operational_desk

    def valores(self):

        quantidade = self.operational_desk[["Embarque", "Quantity (t)"]]

        # Já realizamos a soma em uma etapa anterior (solução de valores duplicados)
        # quantidade = pd.DataFrame(
        #     quantidade.dropna().groupby(["Embarque"]).sum()
        # ).reset_index()

        capacidade_total = self.operational_desk[["Embarque", "Dwt (K)"]].copy()

        capacidade_total.rename(columns={"Dwt (K)": "Dwt (K) total"}, inplace=True)

        capacidade_media = self.operational_desk[["Embarque", "Dwt (K)"]].copy()

        capacidade_media.rename(columns={"Dwt (K)": "Dwt (K) medio"}, inplace=True)

        quantidade_capacidade = pd.merge(
            quantidade, capacidade_total, on="Embarque", how="left"
        )

        quantidade_capacidade = pd.merge(
            quantidade_capacidade, capacidade_media, on="Embarque", how="left"
        )

        # Unifica o Dataframe de quantidade_capacidade com o dataframe passado
        def merge_qc(dataframe: pd.DataFrame, quantidade_capacidade: pd.DataFrame):

            g_c_porto_pier = pd.merge(
                dataframe, quantidade_capacidade, on="Embarque", how="left"
            )
            gs = g_c_porto_pier[
                (g_c_porto_pier["Porto"] == "Guaiba")
                | (g_c_porto_pier["Porto"] == "Sepetiba")
            ]
            gs["Porto"] = "Guaiba e Sepetiba"

            g_c_porto = pd.concat([g_c_porto_pier, gs], ignore_index=False, axis=0)
            g_c_porto = g_c_porto[
                (g_c_porto["Porto"] == "Guaiba")
                | (g_c_porto["Porto"] == "Sepetiba")
                | (g_c_porto["Porto"] == "Guaiba e Sepetiba")
                | (g_c_porto["Porto"] == "Ponta Madeira")
                | (g_c_porto["Porto"] == "Tubarao")
            ]
            g_c_pier = g_c_porto_pier[
                (g_c_porto_pier["Pier"] == "1")
                | (g_c_porto_pier["Pier"] == "3n")
                | (g_c_porto_pier["Pier"] == "3s")
                | (g_c_porto_pier["Pier"] == "4n")
                | (g_c_porto_pier["Pier"] == "4s")
                | (g_c_porto_pier["Pier"] == "2")
                | (g_c_porto_pier["Pier"] == "1n")
                | (g_c_porto_pier["Pier"] == "1s")
            ]

            g_c_porto["Pier"] = g_c_porto["Porto"].map(
                {
                    "Guaiba": "Total",
                    "Sepetiba": "Total",
                    "Guaiba e Sepetiba": "Total",
                    "Ponta Madeira": "Total",
                    "Tubarao": "Total",
                }
            )

            g_c_pier["Porto"] = g_c_pier["Pier"].map(
                {
                    "1": "Ponta Madeira",
                    "3n": "Ponta Madeira",
                    "3s": "Ponta Madeira",
                    "4n": "Ponta Madeira",
                    "4s": "Ponta Madeira",
                    "2": "Tubarao",
                    "1n": "Tubarao",
                    "1s": "Tubarao",
                }
            )

            quantidade_pier = pd.DataFrame(
                g_c_pier[["TCA", "Porto", "Pier", "Quantity (t)"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .sum()
            ).reset_index()

            quantidade_porto = pd.DataFrame(
                g_c_porto[["TCA", "Porto", "Pier", "Quantity (t)"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .sum()
            ).reset_index()

            capacidade_pier_total = pd.DataFrame(
                g_c_pier[["TCA", "Porto", "Pier", "Dwt (K) total"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .sum()
            ).reset_index()

            # 04102023 - Teste aparentemente deu errado (aumento significativo nos valores de CAPACIDADE/Dwt)
            capacidade_pier_media = pd.DataFrame(
                g_c_pier[["TCA", "Porto", "Pier", "Dwt (K) medio"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .mean()
            ).reset_index()

            capacidade_porto_total = pd.DataFrame(
                g_c_porto[["TCA", "Porto", "Pier", "Dwt (K) total"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .sum()
            ).reset_index()

            # 04102023 - Teste Aparentemente deu errado (aumento significativo nos valores de CAPACIDADE/Dwt)
            capacidade_porto_media = pd.DataFrame(
                g_c_porto[["TCA", "Porto", "Pier", "Dwt (K) medio"]]
                .dropna()
                .groupby(["TCA", "Porto", "Pier"])
                .mean()
            ).reset_index()

            quantidade_porto_pier = pd.concat(
                [quantidade_pier, quantidade_porto], ignore_index=False, axis=0
            )

            capacidade_porto_pier_total = pd.concat(
                [capacidade_pier_total, capacidade_porto_total],
                ignore_index=False,
                axis=0,
            )

            capacidade_porto_pier_medio = pd.concat(
                [capacidade_pier_media, capacidade_porto_media],
                ignore_index=False,
                axis=0,
            )

            quantidade_capacidade_porto_pier = pd.merge(
                quantidade_porto_pier,
                capacidade_porto_pier_total,
                on=["TCA", "Porto", "Pier"],
                how="left",
            )

            quantidade_capacidade_porto_pier = pd.merge(
                quantidade_capacidade_porto_pier,
                capacidade_porto_pier_medio,
                on=["TCA", "Porto", "Pier"],
                how="left",
            )

            return quantidade_capacidade_porto_pier

        dataset_quantidade_capacidade = merge_qc(self.dataset_, quantidade_capacidade)

        dataset_quantidade_capacidade = dataset_quantidade_capacidade.reset_index()

        return dataset_quantidade_capacidade


class MergeDados:
    """A classe MergeDados faz a uniao de todas as bases de dados que criamos
    nas funcoes anteriores em uma só.

    Args:

        fila_g: output da classe ContagemNavios.

        fila_s: output da classe ContagemNavios.

        fila_gs: output da classe ContagemNavios.

        fila_pm: output da classe ContagemNavios.

        fila_pm1: output da classe ContagemNavios.

        fila_pm3n: output da classe ContagemNavios.

        fila_pm3s: output da classe ContagemNavios.

        fila_pm4n: output da classe ContagemNavios.

        fila_pm4s: output da classe ContagemNavios.

        fila_t: output da classe ContagemNavios.

        fila_t2: output da classe ContagemNavios.

        fila_t1n: output da classe ContagemNavios.

        fila_t1s: output da classe ContagemNavios.

        tca_g: output da classe ContagemNavios.

        tca_s: output da classe ContagemNavios.

        tca_gs: output da classe ContagemNavios.

        tca_pm: output da classe ContagemNavios.

        tca_pm1: output da classe ContagemNavios.

        tca_pm3n: output da classe ContagemNavios.

        tca_pm3s: output da classe ContagemNavios.

        tca_pm4n: output da classe ContagemNavios.

        tca_pm4s: output da classe ContagemNavios.

        tca_t: output da classe ContagemNavios.

        tca_t2: output da classe ContagemNavios.

        tca_t1n: output da classe ContagemNavios.

        tca_t1s: output da classe ContagemNavios.

        carregando_g: output da classe ContagemNavios.

        carregando_s: output da classe ContagemNavios.

        carregando_gs: output da classe ContagemNavios.

        carregando_pm: output da classe ContagemNavios.

        carregando_pm1: output da classe ContagemNavios.

        carregando_pm3n: output da classe ContagemNavios.

        carregando_pm3s: output da classe ContagemNavios.

        carregando_pm4n: output da classe ContagemNavios.

        carregando_pm4s: output da classe ContagemNavios.

        carregando_t: output da classe ContagemNavios.

        carregando_t2: output da classe ContagemNavios.

        carregando_t1n: output da classe ContagemNavios.

        carregando_t1s: output da classe ContagemNavios.

        desatracando_g: output da classe ContagemNavios.

        desatracando_s: output da classe ContagemNavios.

        desatracando_gs: output da classe ContagemNavios.

        desatracando_pm: output da classe ContagemNavios.

        desatracando_pm1: output da classe ContagemNavios.

        desatracando_pm3n: output da classe ContagemNavios.

        desatracando_pm3s: output da classe ContagemNavios.

        desatracando_pm4n: output da classe ContagemNavios.

        desatracando_pm4s: output da classe ContagemNavios.

        desatracando_t: output da classe ContagemNavios.

        desatracando_t2: output da classe ContagemNavios.

        desatracando_t1n: output da classe ContagemNavios.

        desatracando_t1s: output da classe ContagemNavios.

        chegaram_g: output da classe ContagemNavios.

        chegaram_s: output da classe ContagemNavios.

        chegaram_gs: output da classe ContagemNavios.

        chegaram_pm: output da classe ContagemNavios.

        chegaram_pm1: output da classe ContagemNavios.

        chegaram_pm3n: output da classe ContagemNavios.

        chegaram_pm3s: output da classe ContagemNavios.

        chegaram_pm4n: output da classe ContagemNavios.

        chegaram_pm4s: output da classe ContagemNavios.

        chegaram_t: output da classe ContagemNavios.

        chegaram_t2: output da classe ContagemNavios.

        chegaram_t1n: output da classe ContagemNavios.

        chegaram_t1s: output da classe ContagemNavios.

        dataset_fob: output da classe NaviosQuantPorcentagem.

        dataset_grupos: output da classe NaviosQuantPorcentagem.

        dataset_grupos2: output da classe NaviosQuantPorcentagem.

        dataset_quantidade_capacidade: output da classe QuantidadeCapacidade.

        dataset_total_embarcado: output da classe TotalEmbarcado.
    """

    def __init__(
        self,
        fila_g: Counter,
        fila_s: Counter,
        fila_gs: Counter,
        fila_pm: Counter,
        fila_pm1: Counter,
        fila_pm3n: Counter,
        fila_pm3s: Counter,
        fila_pm4n: Counter,
        fila_pm4s: Counter,
        fila_t: Counter,
        fila_t2: Counter,
        fila_t1n: Counter,
        fila_t1s: Counter,
        tca_g: Counter,
        tca_s: Counter,
        tca_gs: Counter,
        tca_pm: Counter,
        tca_pm1: Counter,
        tca_pm3n: Counter,
        tca_pm3s: Counter,
        tca_pm4n: Counter,
        tca_pm4s: Counter,
        tca_t: Counter,
        tca_t2: Counter,
        tca_t1n: Counter,
        tca_t1s: Counter,
        carregando_g: Counter,
        carregando_s: Counter,
        carregando_gs: Counter,
        carregando_pm: Counter,
        carregando_pm1: Counter,
        carregando_pm3n: Counter,
        carregando_pm3s: Counter,
        carregando_pm4n: Counter,
        carregando_pm4s: Counter,
        carregando_t: Counter,
        carregando_t2: Counter,
        carregando_t1n: Counter,
        carregando_t1s: Counter,
        desatracando_g: Counter,
        desatracando_s: Counter,
        desatracando_gs: Counter,
        desatracando_pm: Counter,
        desatracando_pm1: Counter,
        desatracando_pm3n: Counter,
        desatracando_pm3s: Counter,
        desatracando_pm4n: Counter,
        desatracando_pm4s: Counter,
        desatracando_t: Counter,
        desatracando_t2: Counter,
        desatracando_t1n: Counter,
        desatracando_t1s: Counter,
        chegaram_g: Counter,
        chegaram_s: Counter,
        chegaram_gs: Counter,
        chegaram_pm: Counter,
        chegaram_pm1: Counter,
        chegaram_pm3n: Counter,
        chegaram_pm3s: Counter,
        chegaram_pm4n: Counter,
        chegaram_pm4s: Counter,
        chegaram_t: Counter,
        chegaram_t2: Counter,
        chegaram_t1n: Counter,
        chegaram_t1s: Counter,
        dataset_fob: pd.DataFrame,
        dataset_grupos: pd.DataFrame,
        dataset_grupos2: pd.DataFrame,
        dataset_quantidade_capacidade: pd.DataFrame,
        dataset_total_embarcado: pd.DataFrame,
    ):

        self.fila_g = fila_g
        self.fila_s = fila_s
        self.fila_gs = fila_gs
        self.fila_pm = fila_pm
        self.fila_pm1 = fila_pm1
        self.fila_pm3n = fila_pm3n
        self.fila_pm3s = fila_pm3s
        self.fila_pm4n = fila_pm4n
        self.fila_pm4s = fila_pm4s
        self.fila_t = fila_t
        self.fila_t2 = fila_t2
        self.fila_t1n = fila_t1n
        self.fila_t1s = fila_t1s
        self.tca_g = tca_g
        self.tca_s = tca_s
        self.tca_gs = tca_gs
        self.tca_pm = tca_pm
        self.tca_pm1 = tca_pm1
        self.tca_pm3n = tca_pm3n
        self.tca_pm3s = tca_pm3s
        self.tca_pm4n = tca_pm4n
        self.tca_pm4s = tca_pm4s
        self.tca_t = tca_t
        self.tca_t2 = tca_t2
        self.tca_t1n = tca_t1n
        self.tca_t1s = tca_t1s
        self.carregando_g = carregando_g
        self.carregando_s = carregando_s
        self.carregando_gs = carregando_gs
        self.carregando_pm = carregando_pm
        self.carregando_pm1 = carregando_pm1
        self.carregando_pm3n = carregando_pm3n
        self.carregando_pm3s = carregando_pm3s
        self.carregando_pm4n = carregando_pm4n
        self.carregando_pm4s = carregando_pm4s
        self.carregando_t = carregando_t
        self.carregando_t2 = carregando_t2
        self.carregando_t1n = carregando_t1n
        self.carregando_t1s = carregando_t1s
        self.desatracando_g = desatracando_g
        self.desatracando_s = desatracando_s
        self.desatracando_gs = desatracando_gs
        self.desatracando_pm = desatracando_pm
        self.desatracando_pm1 = desatracando_pm1
        self.desatracando_pm3n = desatracando_pm3n
        self.desatracando_pm3s = desatracando_pm3s
        self.desatracando_pm4n = desatracando_pm4n
        self.desatracando_pm4s = desatracando_pm4s
        self.desatracando_t = desatracando_t
        self.desatracando_t2 = desatracando_t2
        self.desatracando_t1n = desatracando_t1n
        self.desatracando_t1s = desatracando_t1s
        self.chegaram_g = chegaram_g
        self.chegaram_s = chegaram_s
        self.chegaram_gs = chegaram_gs
        self.chegaram_pm = chegaram_pm
        self.chegaram_pm1 = chegaram_pm1
        self.chegaram_pm3n = chegaram_pm3n
        self.chegaram_pm3s = chegaram_pm3s
        self.chegaram_pm4n = chegaram_pm4n
        self.chegaram_pm4s = chegaram_pm4s
        self.chegaram_t = chegaram_t
        self.chegaram_t2 = chegaram_t2
        self.chegaram_t1n = chegaram_t1n
        self.chegaram_t1s = chegaram_t1s
        self.dataset_fob = dataset_fob
        self.dataset_grupos = dataset_grupos
        self.dataset_grupos2 = dataset_grupos2
        self.dataset_quantidade_capacidade = dataset_quantidade_capacidade
        self.dataset_total_embarcado = dataset_total_embarcado

    def unificando(self):

        # Unificando todas as bases de quantidade de navios na fila por dia.

        (
            fila_g,
            fila_s,
            fila_gs,
            fila_pm,
            fila_pm1,
            fila_pm3n,
            fila_pm3s,
            fila_pm4n,
            fila_pm4s,
            fila_t,
            fila_t2,
            fila_t1n,
            fila_t1s,
        ) = (
            pd.DataFrame(
                list(self.fila_g.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_s.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_gs.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_pm.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_pm1.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_pm3n.items()),
                columns=["Dia", "Numero de navios na fila"],
            ),
            pd.DataFrame(
                list(self.fila_pm3s.items()),
                columns=["Dia", "Numero de navios na fila"],
            ),
            pd.DataFrame(
                list(self.fila_pm4n.items()),
                columns=["Dia", "Numero de navios na fila"],
            ),
            pd.DataFrame(
                list(self.fila_pm4s.items()),
                columns=["Dia", "Numero de navios na fila"],
            ),
            pd.DataFrame(
                list(self.fila_t.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_t2.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_t1n.items()), columns=["Dia", "Numero de navios na fila"]
            ),
            pd.DataFrame(
                list(self.fila_t1s.items()), columns=["Dia", "Numero de navios na fila"]
            ),
        )

        # Unificando todos os dados de quantidade de navios por data TCA por dia.

        (
            tca_g,
            tca_s,
            tca_gs,
            tca_pm,
            tca_pm1,
            tca_pm3n,
            tca_pm3s,
            tca_pm4n,
            tca_pm4s,
            tca_t,
            tca_t2,
            tca_t1n,
            tca_t1s,
        ) = (
            pd.DataFrame(
                list(self.tca_g.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_s.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_gs.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm1.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm3n.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm3s.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm4n.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_pm4s.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_t.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_t2.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_t1n.items()), columns=["Dia", "Numero de navios TCA"],
            ),
            pd.DataFrame(
                list(self.tca_t1s.items()), columns=["Dia", "Numero de navios TCA"],
            ),
        )

        # Vamos unificar todas as bases de quantidade de navios carregando por dia.

        (
            carregando_g,
            carregando_s,
            carregando_gs,
            carregando_pm,
            carregando_pm1,
            carregando_pm3n,
            carregando_pm3s,
            carregando_pm4n,
            carregando_pm4s,
            carregando_t,
            carregando_t2,
            carregando_t1n,
            carregando_t1s,
        ) = (
            pd.DataFrame(
                list(self.carregando_g.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_s.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_gs.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm1.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm3n.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm3s.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm4n.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_pm4s.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_t.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_t2.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_t1n.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
            pd.DataFrame(
                list(self.carregando_t1s.items()),
                columns=["Dia", "Numero de navios carregando"],
            ),
        )

        # Vamos unificar todas as bases com quantidade de navios que destracaram por dia.

        (
            desatracando_g,
            desatracando_s,
            desatracando_gs,
            desatracando_pm,
            desatracando_pm1,
            desatracando_pm3n,
            desatracando_pm3s,
            desatracando_pm4n,
            desatracando_pm4s,
            desatracando_t,
            desatracando_t2,
            desatracando_t1n,
            desatracando_t1s,
        ) = (
            pd.DataFrame(
                list(self.desatracando_g.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_s.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_gs.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm1.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm3n.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm3s.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm4n.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_pm4s.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_t.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_t2.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_t1n.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
            pd.DataFrame(
                list(self.desatracando_t1s.items()),
                columns=["Dia", "Numero de navios desatracando"],
            ),
        )

        # Vamos unificar as bases com a quantidade de navios que chegaram por dia.

        (
            chegaram_g,
            chegaram_s,
            chegaram_gs,
            chegaram_pm,
            chegaram_pm1,
            chegaram_pm3n,
            chegaram_pm3s,
            chegaram_pm4n,
            chegaram_pm4s,
            chegaram_t,
            chegaram_t2,
            chegaram_t1n,
            chegaram_t1s,
        ) = (
            pd.DataFrame(
                list(self.chegaram_g.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_s.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_gs.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm1.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm3n.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm3s.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm4n.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_pm4s.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_t.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_t2.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_t1n.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
            pd.DataFrame(
                list(self.chegaram_t1s.items()),
                columns=["Dia", "Numero de navios que chegaram"],
            ),
        )

        # Vamos agora gerar bases por porto e pier com base nas demais bases que
        # acabamos de unificar: navios que estao na fila, navios que chegaram,
        # navios por TCA, navios carregando e navios que desatracaram.

        def uniao(fila, chegaram, tca, carregando, desatracando):

            dfs = [fila, chegaram, tca, carregando, desatracando]

            base = reduce(
                lambda left, right: pd.merge(left, right, on="Dia", how="outer"), dfs
            )

            return base

        (
            base_g,
            base_s,
            base_gs,
            base_pm,
            base_pm1,
            base_pm3n,
            base_pm3s,
            base_pm4n,
            base_pm4s,
            base_t,
            base_t2,
            base_t1n,
            base_t1s,
        ) = (
            uniao(fila_g, chegaram_g, tca_g, carregando_g, desatracando_g),
            uniao(fila_s, chegaram_s, tca_s, carregando_s, desatracando_s),
            uniao(fila_gs, chegaram_gs, tca_gs, carregando_gs, desatracando_gs),
            uniao(fila_pm, chegaram_pm, tca_pm, carregando_pm, desatracando_pm),
            uniao(fila_pm1, chegaram_pm1, tca_pm1, carregando_pm1, desatracando_pm1),
            uniao(
                fila_pm3n, chegaram_pm3n, tca_pm3n, carregando_pm3n, desatracando_pm3n,
            ),
            uniao(
                fila_pm3s, chegaram_pm3s, tca_pm3s, carregando_pm3s, desatracando_pm3s,
            ),
            uniao(
                fila_pm4n, chegaram_pm4n, tca_pm4n, carregando_pm4n, desatracando_pm4n,
            ),
            uniao(
                fila_pm4s, chegaram_pm4s, tca_pm4s, carregando_pm4s, desatracando_pm4s,
            ),
            uniao(fila_t, chegaram_t, tca_t, carregando_t, desatracando_t),
            uniao(fila_t2, chegaram_t2, tca_t2, carregando_t2, desatracando_t2),
            uniao(fila_t1n, chegaram_t1n, tca_t1n, carregando_t1n, desatracando_t1n),
            uniao(fila_t1s, chegaram_t1s, tca_t1s, carregando_t1s, desatracando_t1s),
        )

        # Vamos gerar as variaveis de porto e de pier nas bases que acabamos de criar

        for var in (
            (base_g, "Guaiba"),
            (base_s, "Sepetiba"),
            (base_gs, "Guaiba e Sepetiba"),
            (base_pm, "Ponta Madeira"),
            (base_pm1, "Ponta Madeira"),
            (base_pm3n, "Ponta Madeira"),
            (base_pm3s, "Ponta Madeira"),
            (base_pm4n, "Ponta Madeira"),
            (base_pm4s, "Ponta Madeira"),
            (base_t, "Tubarao"),
            (base_t2, "Tubarao"),
            (base_t1n, "Tubarao"),
            (base_t1s, "Tubarao"),
        ):
            var[0]["Porto"] = var[1]

        for var in (
            (base_g, "Total"),
            (base_s, "Total"),
            (base_gs, "Total"),
            (base_pm, "Total"),
            (base_pm1, "1"),
            (base_pm3n, "3n"),
            (base_pm3s, "3s"),
            (base_pm4n, "4n"),
            (base_pm4s, "4s"),
            (base_t, "Total"),
            (base_t2, "2"),
            (base_t1n, "1n"),
            (base_t1s, "1s"),
        ):
            var[0]["Pier"] = var[1]

        # Abaixo, completamos os dados missing por zero.

        for var in (
            base_g,
            base_s,
            base_gs,
            base_pm,
            base_pm1,
            base_pm3n,
            base_pm3s,
            base_pm4n,
            base_pm4s,
            base_t,
            base_t2,
            base_t1n,
            base_t1s,
        ):
            var.fillna(0)

        # Por fim, juntamos os dados dos piers de Ponta Madeira e Tubarao aos
        # dados dos portos aos seus respectivos piers.

        base_pm = pd.concat(
            [base_pm, base_pm1, base_pm3n, base_pm3s, base_pm4n, base_pm4s],
            ignore_index=False,
            axis=0,
        )

        base_t = pd.concat(
            [base_t, base_t2, base_t1n, base_t1s], ignore_index=False, axis=0
        )

        # Vamos renomear as variaveis de dia para unificar todas as bases

        self.dataset_fob.rename(columns={"ACE": "Dia"}, inplace=True)
        self.dataset_grupos.rename(columns={"ACE": "Dia"}, inplace=True)
        self.dataset_grupos2.rename(columns={"ACE": "Dia"}, inplace=True)
        self.dataset_quantidade_capacidade.rename(columns={"TCA": "Dia"}, inplace=True)

        # Versão anterior
        # self.dataset_fob.rename(columns={"ACE": "Dia"}, inplace=True)
        # self.dataset_grupos.rename(columns={"ACE": "Dia"}, inplace=True)
        # self.dataset_grupos2.rename(columns={"ACE": "Dia"}, inplace=True)
        # self.dataset_quantidade_capacidade.rename(columns={"ACE": "Dia"}, inplace=True)

        del self.dataset_quantidade_capacidade["index"]

        # Por fim, transformamos as variaveis de data em datetime

        self.dataset_fob["Dia"] = pd.to_datetime(
            self.dataset_fob["Dia"], dayfirst=True
        ).dt.normalize()
        self.dataset_grupos.Dia = pd.to_datetime(self.dataset_grupos.Dia, dayfirst=True)
        self.dataset_grupos["Dia"] = self.dataset_grupos["Dia"].dt.normalize()
        self.dataset_grupos2.Dia = pd.to_datetime(
            self.dataset_grupos2.Dia, dayfirst=True
        )
        self.dataset_grupos2["Dia"] = self.dataset_grupos2["Dia"].dt.normalize()
        self.dataset_quantidade_capacidade.Dia = pd.to_datetime(
            self.dataset_quantidade_capacidade.Dia, dayfirst=True
        )
        self.dataset_quantidade_capacidade["Dia"] = self.dataset_quantidade_capacidade[
            "Dia"
        ].dt.normalize()

        # Fazemos entao o merge de todas as bases de dados com quantidades e
        # porcentagens por tipos de navio alem da base de quantidade e capacidade.
        # Estamos fazendo essa uniao pelos portos Guaiba, Sepetiba, Guaiba + Sepetiba,
        # Ponta Madeira, Tubarao.

        dfs = [
            base_g,
            self.dataset_total_embarcado,
            self.dataset_fob,
            self.dataset_grupos,
            self.dataset_grupos2,
            self.dataset_quantidade_capacidade,
        ]

        base_g = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Dia", "Porto", "Pier"], how="left"
            ),
            dfs,
        )

        dfs = [
            base_s,
            self.dataset_total_embarcado,
            self.dataset_fob,
            self.dataset_grupos,
            self.dataset_grupos2,
            self.dataset_quantidade_capacidade,
        ]

        base_s = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Dia", "Porto", "Pier"], how="left"
            ),
            dfs,
        )

        dfs = [
            base_gs,
            self.dataset_total_embarcado,
            self.dataset_fob,
            self.dataset_grupos,
            self.dataset_grupos2,
            self.dataset_quantidade_capacidade,
        ]

        base_gs = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Dia", "Porto", "Pier"], how="left"
            ),
            dfs,
        )

        dfs = [
            base_pm,
            self.dataset_total_embarcado,
            self.dataset_fob,
            self.dataset_grupos,
            self.dataset_grupos2,
            self.dataset_quantidade_capacidade,
        ]

        base_pm = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Dia", "Porto", "Pier"], how="left"
            ),
            dfs,
        )

        dfs = [
            base_t,
            self.dataset_total_embarcado,
            self.dataset_fob,
            self.dataset_grupos,
            self.dataset_grupos2,
            self.dataset_quantidade_capacidade,
        ]

        base_t = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Dia", "Porto", "Pier"], how="left"
            ),
            dfs,
        )

        # Vamos substituir os valores 0 por missing

        base_g[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_g[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_s[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_s[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_gs[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_gs[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_pm[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_pm[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_t[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_t[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)

        # Para terminar, vamos unir as bases por portos em uma unica base de dados
        # chamada de base_diaria.

        base_diaria = pd.concat(
            [base_g, base_s, base_gs, base_pm, base_t], ignore_index=False, axis=0
        )

        base_diaria.rename(columns={"Dia": "Day", "Porto": "Port"}, inplace=True)

        base_diaria = base_diaria.fillna(0)

        # Vamos preencher os valores de volume embarcado que estao com zero pelo
        # ultimo valor nao negativo presente.

        base_diaria["Total Embarcado"].replace(0, np.nan, inplace=True)
        base_diaria["Total Embarcado"] = base_diaria.groupby(["Port", "Pier"])[
            ["Total Embarcado"]
        ].ffill()

        return base_diaria


class BaseUnica:
    """A classe BaseUnica utiliza outras classes do modulo classes_tratamento
    para gerar uma base de dados que contenha informacoes sobre a quantidade
    de navios que estao na fila por dia, a quantidade de navios que atracou por
    dia, a quantidade de navios que ficou apenas carregando, a quantidade que
    desatracou e a quantidade que chegou no porto. Para alem dessas informacoes
    sobre navios, temos tambem quantidade e porcentagem de cada tipo de navio
    por dia, dados sobre a quantidade embarcada e a capacidade
    de cada navio. Para maiores detalhes, veja as classes ContagemNavios,
    NaviosQuantPorcentagem, QuantidadeCapacidade e TotalEmbarcado, todas do
    modulo classes_tratamento.

    Args:

        dataset: Output da classe MergeDemurrageOperationalDesk

        operational_desk: Output da classe MergeDemurrageOperationalDesk

        indices1: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1 de Ponta Madeira.

        indices2: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 3n de Ponta Madeira.

        indices3: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 3s de Ponta Madeira.

        indices4: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 4n de Ponta Madeira.

        indices5: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 4s de Ponta Madeira.

        indices6: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao porto Sepetiba.

        indices7: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao porto Guaiba.

        indices8: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 2 de Tubarao.

        indices9: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1n de Tubarao.

        indices10: Dataframe com os indicadores DISPONIBILIDADE, UTILIZACAO, OEE, CAPACIDADE referente ao pier 1s de Tubarao.

    Returns:

        base_diaria: Base de dados que acrescenta novas variaveis a base dataset.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        dataset_: pd.DataFrame,
        operational_desk: pd.DataFrame,
        indices1: pd.DataFrame,
        indices2: pd.DataFrame,
        indices3: pd.DataFrame,
        indices4: pd.DataFrame,
        indices5: pd.DataFrame,
        indices6: pd.DataFrame,
        indices7: pd.DataFrame,
        indices8: pd.DataFrame,
        indices9: pd.DataFrame,
        indices10: pd.DataFrame,
    ):
        self.dataset = dataset
        self.dataset_ = dataset
        self.dataset_g = self.dataset[self.dataset["Porto"] == "Guaiba"].reset_index(
            drop=True
        )
        self.dataset_s = self.dataset[self.dataset["Porto"] == "Sepetiba"].reset_index(
            drop=True
        )
        self.dataset_gs = self.dataset[
            (self.dataset["Porto"] == "Guaiba") | (self.dataset["Porto"] == "Sepetiba")
        ].reset_index(drop=True)
        self.dataset_gs["Porto"] = "Guaiba e Sepetiba"
        self.dataset_pm = self.dataset[
            self.dataset["Porto"] == "Ponta Madeira"
        ].reset_index(drop=True)
        self.dataset_t = self.dataset[self.dataset["Porto"] == "Tubarao"].reset_index(
            drop=True
        )
        self.operational_desk = operational_desk
        self.indices1 = indices1
        self.indices2 = indices2
        self.indices3 = indices3
        self.indices4 = indices4
        self.indices5 = indices5
        self.indices6 = indices6
        self.indices7 = indices7
        self.indices8 = indices8
        self.indices9 = indices9
        self.indices10 = indices10

    def resultado(self):
        """Uniao das bases."""

        # Listas de navios na fila

        navios_na_fila = ContagemNavios(
            self.dataset_g,
            self.dataset_s,
            self.dataset_gs,
            self.dataset_pm,
            self.dataset_t,
        ).navios_esperando()
        (
            fila_g,
            fila_s,
            fila_gs,
            fila_pm,
            fila_pm1,
            fila_pm3n,
            fila_pm3s,
            fila_pm4n,
            fila_pm4s,
            fila_t,
            fila_t2,
            fila_t1n,
            fila_t1s,
        ) = (
            navios_na_fila[0],
            navios_na_fila[1],
            navios_na_fila[2],
            navios_na_fila[3],
            navios_na_fila[4],
            navios_na_fila[5],
            navios_na_fila[6],
            navios_na_fila[7],
            navios_na_fila[8],
            navios_na_fila[9],
            navios_na_fila[10],
            navios_na_fila[11],
            navios_na_fila[12],
        )

        # Listas de navios que atracaram

        navios_tca = ContagemNavios(
            self.dataset_g,
            self.dataset_s,
            self.dataset_gs,
            self.dataset_pm,
            self.dataset_t,
        ).navios_tca()
        (
            tca_g,
            tca_s,
            tca_gs,
            tca_pm,
            tca_pm1,
            tca_pm3n,
            tca_pm3s,
            tca_pm4n,
            tca_pm4s,
            tca_t,
            tca_t2,
            tca_t1n,
            tca_t1s,
        ) = (
            navios_tca[0],
            navios_tca[1],
            navios_tca[2],
            navios_tca[3],
            navios_tca[4],
            navios_tca[5],
            navios_tca[6],
            navios_tca[7],
            navios_tca[8],
            navios_tca[9],
            navios_tca[10],
            navios_tca[11],
            navios_tca[12],
        )

        # Listas de navios que estao carregando

        navios_carregando = ContagemNavios(
            self.dataset_g,
            self.dataset_s,
            self.dataset_gs,
            self.dataset_pm,
            self.dataset_t,
        ).navios_carregando()
        (
            carregando_g,
            carregando_s,
            carregando_gs,
            carregando_pm,
            carregando_pm1,
            carregando_pm3n,
            carregando_pm3s,
            carregando_pm4n,
            carregando_pm4s,
            carregando_t,
            carregando_t2,
            carregando_t1n,
            carregando_t1s,
        ) = (
            navios_carregando[0],
            navios_carregando[1],
            navios_carregando[2],
            navios_carregando[3],
            navios_carregando[4],
            navios_carregando[5],
            navios_carregando[6],
            navios_carregando[7],
            navios_carregando[8],
            navios_carregando[9],
            navios_carregando[10],
            navios_carregando[11],
            navios_carregando[12],
        )

        # Listas de navios que desatracaram

        navios_desatracaram = ContagemNavios(
            self.dataset_g,
            self.dataset_s,
            self.dataset_gs,
            self.dataset_pm,
            self.dataset_t,
        ).navios_desatracaram()
        (
            desatracando_g,
            desatracando_s,
            desatracando_gs,
            desatracando_pm,
            desatracando_pm1,
            desatracando_pm3n,
            desatracando_pm3s,
            desatracando_pm4n,
            desatracando_pm4s,
            desatracando_t,
            desatracando_t2,
            desatracando_t1n,
            desatracando_t1s,
        ) = (
            navios_desatracaram[0],
            navios_desatracaram[1],
            navios_desatracaram[2],
            navios_desatracaram[3],
            navios_desatracaram[4],
            navios_desatracaram[5],
            navios_desatracaram[6],
            navios_desatracaram[7],
            navios_desatracaram[8],
            navios_desatracaram[9],
            navios_desatracaram[10],
            navios_desatracaram[11],
            navios_desatracaram[12],
        )

        # Listas de navios que chegaram

        navios_chegaram = ContagemNavios(
            self.dataset_g,
            self.dataset_s,
            self.dataset_gs,
            self.dataset_pm,
            self.dataset_t,
        ).navios_chegaram()
        (
            chegaram_g,
            chegaram_s,
            chegaram_gs,
            chegaram_pm,
            chegaram_pm1,
            chegaram_pm3n,
            chegaram_pm3s,
            chegaram_pm4n,
            chegaram_pm4s,
            chegaram_t,
            chegaram_t2,
            chegaram_t1n,
            chegaram_t1s,
        ) = (
            navios_chegaram[0],
            navios_chegaram[1],
            navios_chegaram[2],
            navios_chegaram[3],
            navios_chegaram[4],
            navios_chegaram[5],
            navios_chegaram[6],
            navios_chegaram[7],
            navios_chegaram[8],
            navios_chegaram[9],
            navios_chegaram[10],
            navios_chegaram[11],
            navios_chegaram[12],
        )

        # Gera o dataframe com quantidade e porcentagem de navios por tipo

        tipos_de_navios = NaviosQuantPorcentagem(self.dataset).resultado()
        dataset_fob, dataset_grupos, dataset_grupos2 = (
            tipos_de_navios[1],
            tipos_de_navios[2],
            tipos_de_navios[3],
        )

        # 22-09-2023 procurando inconsistências
        # Gera variaveis com a quantidade embarcada e a capacidade dos navios.

        dataset_quantidade_capacidade = QuantidadeCapacidade(
            self.dataset_, self.operational_desk
        ).valores()

        # Faz o calculo do total embarcado com base na data de desatracacao

        dataset_total_embarcado = TotalEmbarcado(
            self.indices1,
            self.indices2,
            self.indices3,
            self.indices4,
            self.indices5,
            self.indices6,
            self.indices7,
            self.indices8,
            self.indices9,
            self.indices10,
        )

        # Abaixo, fazemos o merge de todas as bases de dados geradas ate agora em
        # uma unica base de dados chabada base diaria que contem dados dos seguintes
        # portos e piers: Guaba, Sepetiba, Guaiba+Sepetiba, Ponta Madeira, Tubarao,
        # 1, 3n, 3s, 4n, 4s, 2, 1n, 1s.

        base_diaria = MergeDados(
            fila_g,
            fila_s,
            fila_gs,
            fila_pm,
            fila_pm1,
            fila_pm3n,
            fila_pm3s,
            fila_pm4n,
            fila_pm4s,
            fila_t,
            fila_t2,
            fila_t1n,
            fila_t1s,
            tca_g,
            tca_s,
            tca_gs,
            tca_pm,
            tca_pm1,
            tca_pm3n,
            tca_pm3s,
            tca_pm4n,
            tca_pm4s,
            tca_t,
            tca_t2,
            tca_t1n,
            tca_t1s,
            carregando_g,
            carregando_s,
            carregando_gs,
            carregando_pm,
            carregando_pm1,
            carregando_pm3n,
            carregando_pm3s,
            carregando_pm4n,
            carregando_pm4s,
            carregando_t,
            carregando_t2,
            carregando_t1n,
            carregando_t1s,
            desatracando_g,
            desatracando_s,
            desatracando_gs,
            desatracando_pm,
            desatracando_pm1,
            desatracando_pm3n,
            desatracando_pm3s,
            desatracando_pm4n,
            desatracando_pm4s,
            desatracando_t,
            desatracando_t2,
            desatracando_t1n,
            desatracando_t1s,
            chegaram_g,
            chegaram_s,
            chegaram_gs,
            chegaram_pm,
            chegaram_pm1,
            chegaram_pm3n,
            chegaram_pm3s,
            chegaram_pm4n,
            chegaram_pm4s,
            chegaram_t,
            chegaram_t2,
            chegaram_t1n,
            chegaram_t1s,
            dataset_fob,
            dataset_grupos,
            dataset_grupos2,
            dataset_quantidade_capacidade,
            dataset_total_embarcado,
        ).unificando()

        # Agora que a base está pronta, vamos acrescentar o lag da varia vel de
        # navios na fila em 1 dia. A variável informa a quantidade de navios que
        # restou na fila na mudanca de uma semana para outra.

        base_diaria["Lag 1 mes numero de navios na fila"] = base_diaria.groupby(
            ["Port", "Pier"]
        )["Numero de navios na fila"].shift(30)
        base_diaria["Dia da semana"] = base_diaria["Day"].dt.dayofweek
        base_diaria["Lag 1 mes numero de navios na fila"] = np.where(
            base_diaria["Dia da semana"] != 0,
            0,
            base_diaria["Lag 1 mes numero de navios na fila"],
        )

        base_diaria["Lag 2 meses numero de navios na fila"] = base_diaria.groupby(
            ["Port", "Pier"]
        )["Numero de navios na fila"].shift(60)
        base_diaria["Dia da semana"] = base_diaria["Day"].dt.dayofweek
        base_diaria["Lag 2 meses numero de navios na fila"] = np.where(
            base_diaria["Dia da semana"] != 0,
            0,
            base_diaria["Lag 2 meses numero de navios na fila"],
        )

        base_diaria["Lag 3 meses numero de navios na fila"] = base_diaria.groupby(
            ["Port", "Pier"]
        )["Numero de navios na fila"].shift(90)
        base_diaria["Dia da semana"] = base_diaria["Day"].dt.dayofweek
        base_diaria["Lag 3 meses numero de navios na fila"] = np.where(
            base_diaria["Dia da semana"] != 0,
            0,
            base_diaria["Lag 3 meses numero de navios na fila"],
        )

        del base_diaria["Dia da semana"]

        return base_diaria


class Indicadores:
    """A classe Indicadores acrescenta a nossa base unificada os indicadores:
        'DISPONIBILIDADE','PRODUTIVIDADE','UTILIZACAO','OEE','CAPACIDADE',
        'TAXA_EFETIVA'.

    Args:

        base_diaria: Output da classe BaseUnica.

        data_inicial_real: Output da classe MergeDemurrageOperationalDesk.

        data_final_real: Output da classe MergeDemurrageOperationalDesk.

        indices1: Base de dados semanal com indicadores nível do pier 1.

        indices2: Base de dados semanal com indicadores nível do pier 3n.

        indices3: Base de dados semanal com indicadores nível do pier 3s.

        indices4: Base de dados semanal com indicadores nível do pier 4n.

        indices5: Base de dados semanal com indicadores nível do pier 4s.

        indices6: Base de dados semanal com indicadores nível do porto de Sepetiba.

        indices7: Base de dados semanal com indicadores nível do porto de Guaiba.

        indices8: Base de dados semanal com indicadores nível do pier 2.

        indices9: Base de dados semanal com indicadores nível do pier 1n.

        indices10: Base de dados semanal com indicadores nível do pier 1s.

    Returns:

        base_guaiba: Base de dados acrescida dos indicadores, apenas para
        Guaiba.

        base_sepetiba: Base de dados acrescida dos indicadores, apenas para
        Sepetiba.

        base_pontamadeira: Base de dados acrescida dos indicadores, apenas
        para Ponta Madeira.

        base_tubarao: Base de dados acrescida dos indicadores, apenas para
        Tubarao.

        base_guaibasepetiba: Base de dados acrescida dos indicadores,
        apenas para Guaiba e Sepetiba.

        embarcado_g: Base de dados do total embarcado, apenas para Guaiba.

        embarcado_s: Base de dados do total embarcado, apenas para Sepetiba.
    """

    def __init__(
        self,
        base_diaria: pd.DataFrame,
        data_inicial_real: str,
        data_final_real: str,
        indices1: pd.DataFrame,
        indices2: pd.DataFrame,
        indices3: pd.DataFrame,
        indices4: pd.DataFrame,
        indices5: pd.DataFrame,
        indices6: pd.DataFrame,
        indices7: pd.DataFrame,
        indices8: pd.DataFrame,
        indices9: pd.DataFrame,
        indices10: pd.DataFrame,
    ):

        self.base_diaria = base_diaria
        self.data_inicial_real = data_inicial_real
        self.data_final_real = data_final_real
        self.indices1 = indices1
        self.indices2 = indices2
        self.indices3 = indices3
        self.indices4 = indices4
        self.indices5 = indices5
        self.indices6 = indices6
        self.indices7 = indices7
        self.indices8 = indices8
        self.indices9 = indices9
        self.indices10 = indices10

    def bases(self):
        """Acrescimo dos indicadores."""

        # Vamos gerar uma base de dados com as datas de inicio a fim do periodo real.

        datas = pd.DataFrame(
            pd.date_range(start=self.data_inicial_real, end=self.data_final_real),
            columns=["Day"],
        )
        datas["Day"] = (pd.to_datetime(datas["Day"], dayfirst=True)).dt.normalize()

        # Vamos modificar o nome dos variaveis de data.

        indices1 = self.indices1.rename(columns={"DATA_MIN": "Day"})
        indices2 = self.indices2.rename(columns={"DATA_MIN": "Day"})
        indices3 = self.indices3.rename(columns={"DATA_MIN": "Day"})
        indices4 = self.indices4.rename(columns={"DATA_MIN": "Day"})
        indices5 = self.indices5.rename(columns={"DATA_MIN": "Day"})
        indices6 = self.indices6.rename(columns={"DATA_MIN": "Day"})
        indices7 = self.indices7.rename(columns={"DATA_MIN": "Day"})
        indices8 = self.indices8.rename(columns={"DATA_MIN": "Day"})
        indices9 = self.indices9.rename(columns={"DATA_MIN": "Day"})
        indices10 = self.indices10.rename(columns={"DATA_MIN": "Day"})

        for var in (
            indices1,
            indices2,
            indices3,
            indices4,
            indices5,
            indices6,
            indices7,
            indices8,
            indices9,
            indices10,
        ):
            var["Day"] = (pd.to_datetime(var["Day"], dayfirst=True)).dt.normalize()

        # Vamos fazer a juncao da variavel com as datas e as nossas variaveis
        # de indices.

        (base1, base2, base3, base4, base5, base6, base7, base8, base9, base10) = (
            pd.merge(datas, indices1, on="Day", how="left"),
            pd.merge(datas, indices2, on="Day", how="left"),
            pd.merge(datas, indices3, on="Day", how="left"),
            pd.merge(datas, indices4, on="Day", how="left"),
            pd.merge(datas, indices5, on="Day", how="left"),
            pd.merge(datas, indices6, on="Day", how="left"),
            pd.merge(datas, indices7, on="Day", how="left"),
            pd.merge(datas, indices8, on="Day", how="left"),
            pd.merge(datas, indices9, on="Day", how="left"),
            pd.merge(datas, indices10, on="Day", how="left"),
        )

        # Vamos filtrar apenas as variaveis que iremos utilizar para gerar os
        # indicadores que iremos acrescentar a nossa base de dados.

        variaveis = [
            "Day",
            "DISPONIBILIDADE",
            "PRODUTIVIDADE",
            "UTILIZACAO",
            "OEE",
            "CAPACIDADE",
            "TAXA_EFETIVA",
            "HORAS_DISP",
            "HORAS_OP",
        ]

        for base in [
            base1,
            base2,
            base3,
            base4,
            base5,
            base6,
            base7,
            base8,
            base9,
            base10,
        ]:
            base = base[variaveis]

        for base in [
            base1,
            base2,
            base3,
            base4,
            base5,
            base6,
            base7,
            base8,
            base9,
            base10,
        ]:
            base[
                [
                    "DISPONIBILIDADE",
                    "PRODUTIVIDADE",
                    "UTILIZACAO",
                    "OEE",
                    "CAPACIDADE",
                    "TAXA_EFETIVA",
                    "HORAS_DISP",
                    "HORAS_OP",
                ]
            ] = base[
                [
                    "DISPONIBILIDADE",
                    "PRODUTIVIDADE",
                    "UTILIZACAO",
                    "OEE",
                    "CAPACIDADE",
                    "TAXA_EFETIVA",
                    "HORAS_DISP",
                    "HORAS_OP",
                ]
            ].astype(
                float
            )

        # base1 = base1[variaveis]

        # base2 = base2[variaveis]

        # base3 = base3[variaveis]

        # base4 = base4[variaveis]

        # base5 = base5[variaveis]

        # base6 = base6[variaveis]

        # base7 = base7[variaveis]

        # base8 = base8[variaveis]

        # base9 = base9[variaveis]

        # base10 = base10[variaveis]

        # Vamos modificar os nomes das variaveis para poder distinguir no nome qual
        # o pier a qual pertence cada um desses dados.

        base1.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_1",
                "PRODUTIVIDADE": "PRODUTIVIDADE_1",
                "UTILIZACAO": "UTILIZACAO_1",
                "OEE": "OEE_1",
                "CAPACIDADE": "CAPACIDADE_1",
                "TAXA_EFETIVA": "TAXA_EFETIVA_1",
                "HORAS_DISP": "HORAS_DISP_1",
                "HORAS_OP": "HORAS_OP_1",
            },
            inplace=True,
        )

        base2.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_3n",
                "PRODUTIVIDADE": "PRODUTIVIDADE_3n",
                "UTILIZACAO": "UTILIZACAO_3n",
                "OEE": "OEE_3n",
                "CAPACIDADE": "CAPACIDADE_3n",
                "TAXA_EFETIVA": "TAXA_EFETIVA_3n",
                "HORAS_DISP": "HORAS_DISP_3n",
                "HORAS_OP": "HORAS_OP_3n",
            },
            inplace=True,
        )

        base3.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_3s",
                "PRODUTIVIDADE": "PRODUTIVIDADE_3s",
                "UTILIZACAO": "UTILIZACAO_3s",
                "OEE": "OEE_3s",
                "CAPACIDADE": "CAPACIDADE_3s",
                "TAXA_EFETIVA": "TAXA_EFETIVA_3s",
                "HORAS_DISP": "HORAS_DISP_3s",
                "HORAS_OP": "HORAS_OP_3s",
            },
            inplace=True,
        )

        base4.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_4n",
                "PRODUTIVIDADE": "PRODUTIVIDADE_4n",
                "UTILIZACAO": "UTILIZACAO_4n",
                "OEE": "OEE_4n",
                "CAPACIDADE": "CAPACIDADE_4n",
                "TAXA_EFETIVA": "TAXA_EFETIVA_4n",
                "HORAS_DISP": "HORAS_DISP_4n",
                "HORAS_OP": "HORAS_OP_4n",
            },
            inplace=True,
        )

        base5.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_4s",
                "PRODUTIVIDADE": "PRODUTIVIDADE_4s",
                "UTILIZACAO": "UTILIZACAO_4s",
                "OEE": "OEE_4s",
                "CAPACIDADE": "CAPACIDADE_4s",
                "TAXA_EFETIVA": "TAXA_EFETIVA_4s",
                "HORAS_DISP": "HORAS_DISP_4s",
                "HORAS_OP": "HORAS_OP_4s",
            },
            inplace=True,
        )

        base8.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_2",
                "PRODUTIVIDADE": "PRODUTIVIDADE_2",
                "UTILIZACAO": "UTILIZACAO_2",
                "OEE": "OEE_2",
                "CAPACIDADE": "CAPACIDADE_2",
                "TAXA_EFETIVA": "TAXA_EFETIVA_2",
                "HORAS_DISP": "HORAS_DISP_2",
                "HORAS_OP": "HORAS_OP_2",
            },
            inplace=True,
        )

        base9.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_1n",
                "PRODUTIVIDADE": "PRODUTIVIDADE_1n",
                "UTILIZACAO": "UTILIZACAO_1n",
                "OEE": "OEE_1n",
                "CAPACIDADE": "CAPACIDADE_1n",
                "TAXA_EFETIVA": "TAXA_EFETIVA_1n",
                "HORAS_DISP": "HORAS_DISP_1n",
                "HORAS_OP": "HORAS_OP_1n",
            },
            inplace=True,
        )

        base10.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_1s",
                "PRODUTIVIDADE": "PRODUTIVIDADE_1s",
                "UTILIZACAO": "UTILIZACAO_1s",
                "OEE": "OEE_1s",
                "CAPACIDADE": "CAPACIDADE_1s",
                "TAXA_EFETIVA": "TAXA_EFETIVA_1s",
                "HORAS_DISP": "HORAS_DISP_1s",
                "HORAS_OP": "HORAS_OP_1s",
            },
            inplace=True,
        )

        # Vamos unificar as variaveis dos piers em bases por porto.

        dfs = [base1, base2, base3, base4, base5]

        base_pontamadeira = reduce(
            lambda left, right: pd.merge(left, right, on=["Day"], how="outer"), dfs
        )

        dfs = [base8, base9, base10]

        base_tubarao = reduce(
            lambda left, right: pd.merge(left, right, on=["Day"], how="outer"), dfs
        )

        # Vamos agora calcular cada um desses indicadores para os portos com base
        # nesses dados por piers. Para cada variavel temos uma formula, como visto
        # abaixo.

        base_pontamadeira["DISPONIBILIDADE"] = (
            base_pontamadeira["DISPONIBILIDADE_1"]
            + base_pontamadeira["DISPONIBILIDADE_3n"]
            + base_pontamadeira["DISPONIBILIDADE_3s"]
            + base_pontamadeira["DISPONIBILIDADE_4n"]
            + base_pontamadeira["DISPONIBILIDADE_4s"]
        ) / 5

        base_pontamadeira["UTILIZACAO"] = (
            (base_pontamadeira["UTILIZACAO_1"] * base_pontamadeira["HORAS_DISP_1"])
            + (base_pontamadeira["UTILIZACAO_3n"] * base_pontamadeira["HORAS_DISP_3n"])
            + (base_pontamadeira["UTILIZACAO_3s"] * base_pontamadeira["HORAS_DISP_3s"])
            + (base_pontamadeira["UTILIZACAO_4n"] * base_pontamadeira["HORAS_DISP_4n"])
            + (base_pontamadeira["UTILIZACAO_4s"] * base_pontamadeira["HORAS_DISP_4s"])
        ) / (
            base_pontamadeira["HORAS_DISP_1"]
            + base_pontamadeira["HORAS_DISP_3n"]
            + base_pontamadeira["HORAS_DISP_3s"]
            + base_pontamadeira["HORAS_DISP_4n"]
            + base_pontamadeira["HORAS_DISP_4s"]
        )

        base_pontamadeira["PRODUTIVIDADE"] = (
            (base_pontamadeira["PRODUTIVIDADE_1"] * base_pontamadeira["HORAS_OP_1"])
            + (base_pontamadeira["PRODUTIVIDADE_3n"] * base_pontamadeira["HORAS_OP_3n"])
            + (base_pontamadeira["PRODUTIVIDADE_3s"] * base_pontamadeira["HORAS_OP_3s"])
            + (base_pontamadeira["PRODUTIVIDADE_4n"] * base_pontamadeira["HORAS_OP_4n"])
            + (base_pontamadeira["PRODUTIVIDADE_4s"] * base_pontamadeira["HORAS_OP_4s"])
        ) / (
            base_pontamadeira["HORAS_OP_1"]
            + base_pontamadeira["HORAS_OP_3n"]
            + base_pontamadeira["HORAS_OP_3s"]
            + base_pontamadeira["HORAS_OP_4n"]
            + base_pontamadeira["HORAS_OP_4s"]
        )

        base_pontamadeira["TAXA_EFETIVA"] = (
            (base_pontamadeira["TAXA_EFETIVA_1"] * base_pontamadeira["HORAS_OP_1"])
            + (base_pontamadeira["TAXA_EFETIVA_3n"] * base_pontamadeira["HORAS_OP_3n"])
            + (base_pontamadeira["TAXA_EFETIVA_3s"] * base_pontamadeira["HORAS_OP_3s"])
            + (base_pontamadeira["TAXA_EFETIVA_4n"] * base_pontamadeira["HORAS_OP_4n"])
            + (base_pontamadeira["TAXA_EFETIVA_4s"] * base_pontamadeira["HORAS_OP_4s"])
        ) / (
            base_pontamadeira["HORAS_OP_1"]
            + base_pontamadeira["HORAS_OP_3n"]
            + base_pontamadeira["HORAS_OP_3s"]
            + base_pontamadeira["HORAS_OP_4n"]
            + base_pontamadeira["HORAS_OP_4s"]
        )

        base_pontamadeira["CAPACIDADE"] = (
            base_pontamadeira["CAPACIDADE_1"]
            + base_pontamadeira["CAPACIDADE_3n"]
            + base_pontamadeira["CAPACIDADE_3s"]
            + base_pontamadeira["CAPACIDADE_4n"]
            + base_pontamadeira["CAPACIDADE_4s"]
        )

        base_pontamadeira["OEE"] = (
            base_pontamadeira["DISPONIBILIDADE"]
            * base_pontamadeira["UTILIZACAO"]
            * base_pontamadeira["PRODUTIVIDADE"]
        )

        base_tubarao["DISPONIBILIDADE"] = (
            base_tubarao["DISPONIBILIDADE_2"]
            + base_tubarao["DISPONIBILIDADE_1n"]
            + base_tubarao["DISPONIBILIDADE_1s"]
        ) / 3

        base_tubarao["UTILIZACAO"] = (
            (base_tubarao["UTILIZACAO_2"] * base_tubarao["HORAS_DISP_2"])
            + (base_tubarao["UTILIZACAO_1n"] * base_tubarao["HORAS_DISP_1n"])
            + (base_tubarao["UTILIZACAO_1s"] * base_tubarao["HORAS_DISP_1s"])
        ) / (
            base_tubarao["HORAS_DISP_2"]
            + base_tubarao["HORAS_DISP_1n"]
            + base_tubarao["HORAS_DISP_1s"]
        )

        base_tubarao["PRODUTIVIDADE"] = (
            (base_tubarao["PRODUTIVIDADE_2"] * base_tubarao["HORAS_OP_2"])
            + (base_tubarao["PRODUTIVIDADE_1n"] * base_tubarao["HORAS_OP_1n"])
            + (base_tubarao["PRODUTIVIDADE_1s"] * base_tubarao["HORAS_OP_1s"])
        ) / (
            base_tubarao["HORAS_OP_2"]
            + base_tubarao["HORAS_OP_1n"]
            + base_tubarao["HORAS_OP_1s"]
        )

        base_tubarao["TAXA_EFETIVA"] = (
            (base_tubarao["TAXA_EFETIVA_2"] * base_tubarao["HORAS_OP_2"])
            + (base_tubarao["TAXA_EFETIVA_1n"] * base_tubarao["HORAS_OP_1n"])
            + (base_tubarao["TAXA_EFETIVA_1s"] * base_tubarao["HORAS_OP_1s"])
        ) / (
            base_tubarao["HORAS_OP_2"]
            + base_tubarao["HORAS_OP_1n"]
            + base_tubarao["HORAS_OP_1s"]
        )

        base_tubarao["CAPACIDADE"] = (
            base_tubarao["CAPACIDADE_2"]
            + base_tubarao["CAPACIDADE_1n"]
            + base_tubarao["CAPACIDADE_1s"]
        )

        base_tubarao["OEE"] = (
            base_tubarao["DISPONIBILIDADE"]
            * base_tubarao["UTILIZACAO"]
            * base_tubarao["PRODUTIVIDADE"]
        )

        base7.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_G",
                "PRODUTIVIDADE": "PRODUTIVIDADE_G",
                "UTILIZACAO": "UTILIZACAO_G",
                "OEE": "OEE_G",
                "CAPACIDADE": "CAPACIDADE_G",
                "TAXA_EFETIVA": "TAXA_EFETIVA_G",
                "HORAS_DISP": "HORAS_DISP_G",
                "HORAS_OP": "HORAS_OP_G",
            },
            inplace=True,
        )

        base6.rename(
            columns={
                "DISPONIBILIDADE": "DISPONIBILIDADE_S",
                "PRODUTIVIDADE": "PRODUTIVIDADE_S",
                "UTILIZACAO": "UTILIZACAO_S",
                "OEE": "OEE_S",
                "CAPACIDADE": "CAPACIDADE_S",
                "TAXA_EFETIVA": "TAXA_EFETIVA_S",
                "HORAS_DISP": "HORAS_DISP_S",
                "HORAS_OP": "HORAS_OP_S",
            },
            inplace=True,
        )

        base_guaibasepetiba = pd.merge(base7, base6, how="outer", on="Day")

        base_guaibasepetiba["DISPONIBILIDADE"] = (
            base_guaibasepetiba["DISPONIBILIDADE_G"]
            + base_guaibasepetiba["DISPONIBILIDADE_S"]
        ) / 2

        base_guaibasepetiba["UTILIZACAO"] = (
            (base_guaibasepetiba["UTILIZACAO_G"] * base_guaibasepetiba["HORAS_DISP_G"])
            + (
                base_guaibasepetiba["UTILIZACAO_S"]
                * base_guaibasepetiba["HORAS_DISP_S"]
            )
        ) / (base_guaibasepetiba["HORAS_DISP_G"] + base_guaibasepetiba["HORAS_DISP_S"])

        base_guaibasepetiba["PRODUTIVIDADE"] = (
            (base_guaibasepetiba["PRODUTIVIDADE_G"] * base_guaibasepetiba["HORAS_OP_G"])
            + (
                base_guaibasepetiba["PRODUTIVIDADE_S"]
                * base_guaibasepetiba["HORAS_OP_S"]
            )
        ) / (base_guaibasepetiba["HORAS_OP_G"] + base_guaibasepetiba["HORAS_OP_S"])

        base_guaibasepetiba["TAXA_EFETIVA"] = (
            (base_guaibasepetiba["TAXA_EFETIVA_G"] * base_guaibasepetiba["HORAS_OP_G"])
            + (
                base_guaibasepetiba["TAXA_EFETIVA_S"]
                * base_guaibasepetiba["HORAS_OP_S"]
            )
        ) / (base_guaibasepetiba["HORAS_OP_G"] + base_guaibasepetiba["HORAS_OP_S"])

        base_guaibasepetiba["CAPACIDADE"] = (
            base_guaibasepetiba["CAPACIDADE_G"] + base_guaibasepetiba["CAPACIDADE_S"]
        )

        base_guaibasepetiba["OEE"] = (
            base_guaibasepetiba["DISPONIBILIDADE"]
            * base_guaibasepetiba["UTILIZACAO"]
            * base_guaibasepetiba["PRODUTIVIDADE"]
        )

        # Calculados os indicadores para cada um dos portos com base nos piers, vamos
        # modificar os nomes das variaveis que teinham os nomes dos piers para que
        # possamos ter essas informacoes tanto para os portos quanto para piers.

        base1.rename(
            columns={
                "DISPONIBILIDADE_1": "DISPONIBILIDADE",
                "PRODUTIVIDADE_1": "PRODUTIVIDADE",
                "UTILIZACAO_1": "UTILIZACAO",
                "OEE_1": "OEE",
                "CAPACIDADE_1": "CAPACIDADE",
                "TAXA_EFETIVA_1": "TAXA_EFETIVA",
                "HORAS_DISP_1": "HORAS_DISP",
                "HORAS_OP_1": "HORAS_OP",
            },
            inplace=True,
        )

        base2.rename(
            columns={
                "DISPONIBILIDADE_3n": "DISPONIBILIDADE",
                "PRODUTIVIDADE_3n": "PRODUTIVIDADE",
                "UTILIZACAO_3n": "UTILIZACAO",
                "OEE_3n": "OEE",
                "CAPACIDADE_3n": "CAPACIDADE",
                "TAXA_EFETIVA_3n": "TAXA_EFETIVA",
                "HORAS_DISP_3n": "HORAS_DISP",
                "HORAS_OP_3n": "HORAS_OP",
            },
            inplace=True,
        )

        base3.rename(
            columns={
                "DISPONIBILIDADE_3s": "DISPONIBILIDADE",
                "PRODUTIVIDADE_3s": "PRODUTIVIDADE",
                "UTILIZACAO_3s": "UTILIZACAO",
                "OEE_3s": "OEE",
                "CAPACIDADE_3s": "CAPACIDADE",
                "TAXA_EFETIVA_3s": "TAXA_EFETIVA",
                "HORAS_DISP_3s": "HORAS_DISP",
                "HORAS_OP_3s": "HORAS_OP",
            },
            inplace=True,
        )

        base4.rename(
            columns={
                "DISPONIBILIDADE_4n": "DISPONIBILIDADE",
                "PRODUTIVIDADE_4n": "PRODUTIVIDADE",
                "UTILIZACAO_4n": "UTILIZACAO",
                "OEE_4n": "OEE",
                "CAPACIDADE_4n": "CAPACIDADE",
                "TAXA_EFETIVA_4n": "TAXA_EFETIVA",
                "HORAS_DISP_4n": "HORAS_DISP",
                "HORAS_OP_4n": "HORAS_OP",
            },
            inplace=True,
        )

        base5.rename(
            columns={
                "DISPONIBILIDADE_4s": "DISPONIBILIDADE",
                "PRODUTIVIDADE_4s": "PRODUTIVIDADE",
                "UTILIZACAO_4s": "UTILIZACAO",
                "OEE_4s": "OEE",
                "CAPACIDADE_4s": "CAPACIDADE",
                "TAXA_EFETIVA_4s": "TAXA_EFETIVA",
                "HORAS_DISP_4s": "HORAS_DISP",
                "HORAS_OP_4s": "HORAS_OP",
            },
            inplace=True,
        )

        base6.rename(
            columns={
                "DISPONIBILIDADE_S": "DISPONIBILIDADE",
                "PRODUTIVIDADE_S": "PRODUTIVIDADE",
                "UTILIZACAO_S": "UTILIZACAO",
                "OEE_S": "OEE",
                "CAPACIDADE_S": "CAPACIDADE",
                "TAXA_EFETIVA_S": "TAXA_EFETIVA",
                "HORAS_DISP_S": "HORAS_DISP",
                "HORAS_OP_S": "HORAS_OP",
            },
            inplace=True,
        )

        base7.rename(
            columns={
                "DISPONIBILIDADE_G": "DISPONIBILIDADE",
                "PRODUTIVIDADE_G": "PRODUTIVIDADE",
                "UTILIZACAO_G": "UTILIZACAO",
                "OEE_G": "OEE",
                "CAPACIDADE_G": "CAPACIDADE",
                "TAXA_EFETIVA_G": "TAXA_EFETIVA",
                "HORAS_DISP_G": "HORAS_DISP",
                "HORAS_OP_G": "HORAS_OP",
            },
            inplace=True,
        )

        base8.rename(
            columns={
                "DISPONIBILIDADE_2": "DISPONIBILIDADE",
                "PRODUTIVIDADE_2": "PRODUTIVIDADE",
                "UTILIZACAO_2": "UTILIZACAO",
                "OEE_2": "OEE",
                "CAPACIDADE_2": "CAPACIDADE",
                "TAXA_EFETIVA_2": "TAXA_EFETIVA",
                "HORAS_DISP_2": "HORAS_DISP",
                "HORAS_OP_2": "HORAS_OP",
            },
            inplace=True,
        )

        base9.rename(
            columns={
                "DISPONIBILIDADE_1n": "DISPONIBILIDADE",
                "PRODUTIVIDADE_1n": "PRODUTIVIDADE",
                "UTILIZACAO_1n": "UTILIZACAO",
                "OEE_1n": "OEE",
                "CAPACIDADE_1n": "CAPACIDADE",
                "TAXA_EFETIVA_1n": "TAXA_EFETIVA",
                "HORAS_DISP_1n": "HORAS_DISP",
                "HORAS_OP_1n": "HORAS_OP",
            },
            inplace=True,
        )

        base10.rename(
            columns={
                "DISPONIBILIDADE_1s": "DISPONIBILIDADE",
                "PRODUTIVIDADE_1s": "PRODUTIVIDADE",
                "UTILIZACAO_1s": "UTILIZACAO",
                "OEE_1s": "OEE",
                "CAPACIDADE_1s": "CAPACIDADE",
                "TAXA_EFETIVA_1s": "TAXA_EFETIVA",
                "HORAS_DISP_1s": "HORAS_DISP",
                "HORAS_OP_1s": "HORAS_OP",
            },
            inplace=True,
        )

        # Feitas as modificacoes, vamos filtrar as variaveis de interesse

        variaveis = [
            "Day",
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "CAPACIDADE",
            "TAXA_EFETIVA",
        ]

        base1 = base1[variaveis]
        base2 = base2[variaveis]
        base3 = base3[variaveis]
        base4 = base4[variaveis]
        base5 = base5[variaveis]
        base6 = base6[variaveis]
        base7 = base7[variaveis]
        base8 = base8[variaveis]
        base9 = base9[variaveis]
        base10 = base10[variaveis]
        base_pontamadeira = base_pontamadeira[variaveis]
        base_tubarao = base_tubarao[variaveis]
        base_guaibasepetiba = base_guaibasepetiba[variaveis]

        # Criamos abaixo as variaveis de porto e pier para as novas variaveis criadas

        for var in (
            (base1, "Ponta Madeira"),
            (base2, "Ponta Madeira"),
            (base3, "Ponta Madeira"),
            (base4, "Ponta Madeira"),
            (base5, "Ponta Madeira"),
            (base6, "Sepetiba"),
            (base7, "Guaiba"),
            (base8, "Tubarao"),
            (base9, "Tubarao"),
            (base10, "Tubarao"),
            (base_pontamadeira, "Ponta Madeira"),
            (base_tubarao, "Tubarao"),
            (base_guaibasepetiba, "Guaiba e Sepetiba"),
        ):
            var[0]["Port"] = var[1]

        for var in (
            (base1, "1"),
            (base2, "3n"),
            (base3, "3s"),
            (base4, "4n"),
            (base5, "4s"),
            (base6, "Total"),
            (base7, "Total"),
            (base8, "2"),
            (base9, "1n"),
            (base10, "1s"),
            (base_pontamadeira, "Total"),
            (base_tubarao, "Total"),
            (base_guaibasepetiba, "Total"),
        ):
            var[0]["Pier"] = var[1]

        # Criamos abaixo as bases de dados por porto.

        base_pontamadeira = pd.concat(
            [base1, base2, base3, base4, base5, base_pontamadeira],
            ignore_index=False,
            axis=0,
        )

        base_sepetiba = base6

        base_guaiba = base7

        base_tubarao = pd.concat(
            [base8, base9, base10, base_tubarao], ignore_index=False, axis=0
        )

        # Agora acrscentamos as novas variaveis a base diaria que haviamos criado
        # anteriormente. Tambem geramos as bases de dados com os totais embarcados
        # exclusivamente para os portos de Guaiba e Sepetiba. Estes dados serao
        # utilizados posteriormente.

        base_diaria_guaiba = self.base_diaria[self.base_diaria["Port"] == "Guaiba"]
        base_guaiba = pd.merge(
            base_diaria_guaiba, base_guaiba, on=["Day", "Port", "Pier"], how="left"
        )
        base_diaria_sepetiba = self.base_diaria[self.base_diaria["Port"] == "Sepetiba"]
        base_sepetiba = pd.merge(
            base_diaria_sepetiba, base_sepetiba, on=["Day", "Port", "Pier"], how="left"
        )
        base_diaria_pontamadeira = self.base_diaria[
            self.base_diaria["Port"] == "Ponta Madeira"
        ]
        base_pontamadeira = pd.merge(
            base_diaria_pontamadeira,
            base_pontamadeira,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_diaria_tubarao = self.base_diaria[self.base_diaria["Port"] == "Tubarao"]
        base_tubarao = pd.merge(
            base_diaria_tubarao, base_tubarao, on=["Day", "Port", "Pier"], how="left"
        )
        base_diaria_guaibasepetiba = self.base_diaria[
            self.base_diaria["Port"] == "Guaiba e Sepetiba"
        ]
        base_guaibasepetiba = pd.merge(
            base_diaria_guaibasepetiba,
            base_guaibasepetiba,
            on=["Day", "Port", "Pier"],
            how="left",
        )

        embarcado_g = base_diaria_guaiba[["Day", "Total Embarcado"]]
        embarcado_g.rename(
            columns={"Total Embarcado": "Total Embarcado G"}, inplace=True
        )
        embarcado_s = base_diaria_sepetiba[["Day", "Total Embarcado"]]
        embarcado_s.rename(
            columns={"Total Embarcado": "Total Embarcado S"}, inplace=True
        )

        return [
            base_guaiba,
            base_sepetiba,
            base_pontamadeira,
            base_tubarao,
            base_guaibasepetiba,
            embarcado_g,
            embarcado_s,
        ]


class VariaveisDependentes:
    """A classe VariaveisDependentes cria as variáveis custo de demurrage e
    estadia em horas que nos dá essas informacoes com base nas datas de
    desatracacao.

    Args:

        dataset_: Output da classe MergeDemurrageOperationalDesk.

        base_guaiba: Output da classe ElementosQuimicos.

        base_sepetiba: Output da classe ElementosQuimicos.

        base_guaibasepetiba: Output da classe ElementosQuimicos.

        base_pontamadeira: Output da classe ElementosQuimicos.

        base_tubarao: Output da classe ElementosQuimicos.

    Returns:

        base_guaiba: Base com os dados do porto Guaiba acrescida dos dados
        de elementos químicos.

        base_sepetiba: Base com os dados do porto Sepetiba acrescida dos
        dados de elementos químicos.

        base_guaibasepetiba: Base com os dados dos portos Guaiba e Sepetiba
        acrescida dos dados de elementos químicos.

        base_pontamadeira: Base com os dados do porto Ponta Madeira acrescida
        dos dados de elementos químicos.

        base_tubarao: Base com os dados do porto Tubarao acrescida dos dados
        de elementos químicos.
    """

    def __init__(
        self,
        dataset_: pd.DataFrame,
        base_guaiba: pd.DataFrame,
        base_sepetiba: pd.DataFrame,
        base_guaibasepetiba: pd.DataFrame,
        base_pontamadeira: pd.DataFrame,
        base_tubarao: pd.DataFrame,
    ):
        self.dataset_ = dataset_
        self.base_guaiba = base_guaiba
        self.base_sepetiba = base_sepetiba
        self.base_guaibasepetiba = base_guaibasepetiba
        self.base_pontamadeira = base_pontamadeira
        self.base_tubarao = base_tubarao

    def resultados(self):

        # Vamos separar as bases de dados entre uma base de dados apenas com os
        # dados de porto e outro apenas com os dados de pier.

        dataset = self.dataset_[
            (self.dataset_["Porto"] == "Guaiba")
            | (self.dataset_["Porto"] == "Sepetiba")
            | (self.dataset_["Porto"] == "Ponta Madeira")
            | (self.dataset_["Porto"] == "Tubarao")
        ]

        dataset.rename(columns={"PIER": "Pier"}, inplace=True)
        dataset_pier = dataset[
            (dataset["Pier"] == "1")
            | (dataset["Pier"] == "3n")
            | (dataset["Pier"] == "3s")
            | (dataset["Pier"] == "4n")
            | (dataset["Pier"] == "4s")
            | (dataset["Pier"] == "2")
            | (dataset["Pier"] == "1n")
            | (dataset["Pier"] == "1s")
        ]

        # Vamos recriar as variaveis de porto

        dataset_pier["Porto"] = dataset_pier["Pier"].map(
            {
                "1": "Ponta Madeira",
                "3n": "Ponta Madeira",
                "3s": "Ponta Madeira",
                "4n": "Ponta Madeira",
                "4s": "Ponta Madeira",
                "2": "Tubarao",
                "1n": "Tubarao",
                "1s": "Tubarao",
            }
        )

        dataset["Porto"] = dataset["Pier"].map(
            {
                "1": "Ponta Madeira",
                "3n": "Ponta Madeira",
                "3s": "Ponta Madeira",
                "4n": "Ponta Madeira",
                "4s": "Ponta Madeira",
                "2": "Tubarao",
                "1n": "Tubarao",
                "1s": "Tubarao",
                "Guaiba": "Guaiba",
                "Sepetiba": "Sepetiba",
            }
        )

        dataset["Pier"] = "Total"

        dataset_gs = dataset[
            (dataset["Porto"] == "Guaiba") | (dataset["Porto"] == "Sepetiba")
        ]

        dataset_gs["Porto"] = "Guaiba e Sepetiba"

        dataset = pd.concat(
            [
                dataset,
                pd.concat([dataset_gs, dataset_pier], ignore_index=False, axis=0),
            ],
            ignore_index=False,
            axis=0,
        )

        # Agora vamos criar a variavel de demurrage, fazendo a soma dos valores
        # de multa por portos e por piers com base na data de desatracacao.

        variavel_demurrage = (
            dataset.groupby(["TCA", "Porto", "Pier"])["Valor de Multa ou Prêmio"]
            .agg("sum")
            .to_frame()
        )

        variavel_demurrage = variavel_demurrage.reset_index()

        variavel_demurrage.rename(
            columns={
                "Valor de Multa ou Prêmio": "Multas por dia",
                "Porto": "Port",
                "TCA": "Day",
            },
            inplace=True,
        )

        variavel_demurrage["Day"] = (
            pd.to_datetime(variavel_demurrage["Day"], dayfirst=True)
        ).dt.normalize()

        # Abaixo, acrescentamos a variavel de custo de demurrage as nossas bases

        base_guaiba = pd.merge(
            self.base_guaiba, variavel_demurrage, on=["Day", "Port", "Pier"], how="left"
        )
        base_sepetiba = pd.merge(
            self.base_sepetiba,
            variavel_demurrage,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_guaibasepetiba = pd.merge(
            self.base_guaibasepetiba,
            variavel_demurrage,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_pontamadeira = pd.merge(
            self.base_pontamadeira,
            variavel_demurrage,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_tubarao = pd.merge(
            self.base_tubarao,
            variavel_demurrage,
            on=["Day", "Port", "Pier"],
            how="left",
        )

        # Agora criamos nossa outra variavel dependente, tempo de estadia, que é
        # calculada com base na média do tempo de estadia de cada navio com base
        # na data de desatracacao do mesmo.

        estadia_em_horas = (
            dataset.groupby(["TCA", "Porto", "Pier"])["Estadia em Horas"].agg("mean")
        ).to_frame()

        estadia_em_horas = estadia_em_horas.reset_index()
        estadia_em_horas.rename(columns={"TCA": "Day", "Porto": "Port"}, inplace=True)

        # Vamos acrescentar também uma variável de soma de horas de estádia

        soma_estadia_em_horas = (
            dataset.groupby(["TCA", "Porto", "Pier"])["Estadia em Horas"].agg("sum")
        ).to_frame()

        soma_estadia_em_horas = soma_estadia_em_horas.reset_index()
        soma_estadia_em_horas.rename(
            columns={
                "TCA": "Day",
                "Porto": "Port",
                "Estadia em Horas": "Soma Estadia em Horas",
            },
            inplace=True,
        )

        # Vamos acrescentar essa nova variavel as nossas bases de dados criadas anteriormente

        base_guaiba = pd.merge(
            base_guaiba, estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )
        base_guaiba = pd.merge(
            base_guaiba, soma_estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )
        base_sepetiba = pd.merge(
            base_sepetiba, estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )
        base_sepetiba = pd.merge(
            base_sepetiba, soma_estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )
        base_guaibasepetiba = pd.merge(
            base_guaibasepetiba,
            estadia_em_horas,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_guaibasepetiba = pd.merge(
            base_guaibasepetiba,
            soma_estadia_em_horas,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_pontamadeira = pd.merge(
            base_pontamadeira, estadia_em_horas, on=["Day", "Port", "Pier"], how="left",
        )
        base_pontamadeira = pd.merge(
            base_pontamadeira,
            soma_estadia_em_horas,
            on=["Day", "Port", "Pier"],
            how="left",
        )
        base_tubarao = pd.merge(
            base_tubarao, estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )

        base_tubarao = pd.merge(
            base_tubarao, soma_estadia_em_horas, on=["Day", "Port", "Pier"], how="left"
        )

        base_diaria = pd.concat(
            [
                base_guaiba,
                base_sepetiba,
                base_guaibasepetiba,
                base_pontamadeira,
                base_tubarao,
            ],
            ignore_index=False,
            axis=0,
        )

        coluna_port = base_diaria.pop("Port")
        coluna_pier = base_diaria.pop("Pier")
        base_diaria.insert(1, "Pier", coluna_pier)
        base_diaria.insert(1, "Port", coluna_port)

        base_diaria["Day"] = (
            pd.to_datetime(base_diaria["Day"], dayfirst=True)
        ).dt.normalize()

        base_diaria["Semana"] = base_diaria["Day"].dt.week

        base_diaria["Ano"] = base_diaria["Day"].dt.year

        # Vamos criar a variavel de numero de navios que desatracaram por semana pois
        # ela será necessária quando formos fazer a semanalizacao. O calculo do
        # tempo de estadia semanal sera feito aplicando um peso que da mais peso
        # aos dias referentes as semanas nas quais desatracaram muitos navios. Criada
        # essa variavel peso, a semanalizasao do tempo de estadia e feita com a
        # soma simples da mesma ao longo dos dias da semana.

        navios_chegaram_por_semana = (
            base_diaria.groupby(["Port", "Pier", "Semana", "Ano"])[
                "Numero de navios desatracando"
            ]
            .agg("sum")
            .to_frame()
        )
        navios_chegaram_por_semana = navios_chegaram_por_semana.reset_index()
        navios_chegaram_por_semana.rename(
            columns={"Numero de navios desatracando": "Navios desatracaram SEMANA"},
            inplace=True,
        )

        base_diaria = pd.merge(
            base_diaria,
            navios_chegaram_por_semana,
            on=["Port", "Pier", "Semana", "Ano"],
            how="left",
        )
        del base_diaria["Ano"]
        del base_diaria["Semana"]

        base_diaria["Peso"] = (
            base_diaria["Estadia em Horas"]
            * base_diaria["Numero de navios desatracando"]
        ) / (base_diaria["Navios desatracaram SEMANA"])

        base_guaiba = base_guaiba.drop_duplicates()
        base_sepetiba = base_sepetiba.drop_duplicates()
        base_guaibasepetiba = base_guaibasepetiba.drop_duplicates()
        base_pontamadeira = base_pontamadeira.drop_duplicates()
        base_tubarao = base_tubarao.drop_duplicates()
        base_diaria = base_diaria.drop_duplicates()

        return [
            base_guaiba,
            base_sepetiba,
            base_guaibasepetiba,
            base_pontamadeira,
            base_tubarao,
            base_diaria,
        ]


class Semanalizacao:
    """Na classe Semanalizacao transforma nossa base, que está a nível diário, em
    uma base a nível semanal.

    Args:

        base_diaria: Output da classe PrevisoesCOICMA.

    Returns:

        base_semanal: Base semanalizada a partir da base diária.

        base_diaria: Base diária.
    """

    def __init__(self, base_diaria: pd.DataFrame):
        self.base_diaria = base_diaria

    def base(self):
        """Semanalizacao a partir da base diaria."""

        # Vamos separar as bases por porto e por pier para, em seguida aplicar o
        # algoritmo de semanalizacao.

        self.base_diaria["Day"] = (
            pd.to_datetime(self.base_diaria["Day"], dayfirst=True)
        ).dt.normalize()
        self.base_diaria = self.base_diaria.set_index("Day")

        base_diaria_ = self.base_diaria[self.base_diaria.columns.difference(["Peso"])]
        self.base_diaria = self.base_diaria[
            self.base_diaria.columns.difference(["Estadia em Horas"])
        ]

        self.base_diaria["Port_Pier"] = (
            self.base_diaria["Port"] + self.base_diaria["Pier"]
        )

        base_guaiba = self.base_diaria[
            self.base_diaria["Port_Pier"] == "GuaibaTotal"
        ].sort_index()
        base_sepetiba = self.base_diaria[
            self.base_diaria["Port_Pier"] == "SepetibaTotal"
        ].sort_index()
        base_guaibasepetiba = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Guaiba e SepetibaTotal"
        ].sort_index()
        base_pontamadeira = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta MadeiraTotal"
        ].sort_index()
        base_1 = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta Madeira1"
        ].sort_index()
        base_3n = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta Madeira3n"
        ].sort_index()
        base_3s = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta Madeira3s"
        ].sort_index()
        base_4n = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta Madeira4n"
        ].sort_index()
        base_4s = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Ponta Madeira4s"
        ].sort_index()
        base_tubarao = self.base_diaria[
            self.base_diaria["Port_Pier"] == "TubaraoTotal"
        ].sort_index()
        base_2 = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Tubarao2"
        ].sort_index()
        base_1n = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Tubarao1n"
        ].sort_index()
        base_1s = self.base_diaria[
            self.base_diaria["Port_Pier"] == "Tubarao1s"
        ].sort_index()

        # Abaixo temos o dicionario com cada uma das variaveis que iremos semanalizar
        # e o metodo pelo qual vamos fazer essa semanalizacao. Em sua maioria, utilizamos
        # a soma, mas podemos fazer por medias, medianas ou outra estatistica relevante.

        logic = {
            "Numero de navios na fila": "sum",
            "Numero de navios que chegaram": "sum",
            "Numero de navios TCA": "sum",
            "Numero de navios carregando": "sum",
            "Numero de navios desatracando": "sum",
            "Lag 1 mes numero de navios na fila": "sum",
            "Lag 2 meses numero de navios na fila": "sum",
            "Lag 3 meses numero de navios na fila": "sum",
            "Total Embarcado": "sum",
            "Numero de navios FOB": "sum",
            "Numero de navios CFR": "sum",
            "Porcentagem FOB": "sum",
            "Porcentagem CFR": "sum",
            "Numero de navios PANAMAX": "sum",
            "Porcentagem PANAMAX": "sum",
            "Numero de navios CAPE": "sum",
            "Porcentagem CAPE": "sum",
            "Numero de navios VLOC": "sum",
            "Porcentagem VLOC": "sum",
            "Numero de navios NEWCASTLE": "sum",
            "Porcentagem NEWCASTLE": "sum",
            "Numero de navios VALEMAX": "sum",
            "Porcentagem VALEMAX": "sum",
            "Numero de navios BABYCAPE": "sum",
            "Porcentagem BABYCAPE": "sum",
            "Numero de navios SPOT/FOB": "sum",
            "Porcentagem SPOT/FOB": "sum",
            "Numero de navios Frota Dedicada/SPOT/FOB": "sum",
            "Porcentagem Frota Dedicada/SPOT/FOB": "sum",
            "Numero de navios Frota Dedicada/FOB": "sum",
            "Porcentagem Frota Dedicada/FOB": "sum",
            "Numero de navios Frota Dedicada": "sum",
            "Porcentagem Frota Dedicada": "sum",
            "Quantity (t)": "sum",
            "Dwt (K) total": "sum",
            "Dwt (K) medio": "mean",
            "DISPONIBILIDADE": "max",
            "UTILIZACAO": "max",
            "OEE": "max",
            "CAPACIDADE": "sum",
            "TAXA_EFETIVA": "max",
            "Multas por dia": "sum",
            "Peso": "sum",
            "Soma Estadia em Horas": "sum",
        }

        # Em sequencia, fazemos a semanalizacao dos dados

        base_guaibaw = base_guaiba.resample("W").apply(logic)
        base_sepetibaw = base_sepetiba.resample("W").apply(logic)
        base_guaibasepetibaw = base_guaibasepetiba.resample("W").apply(logic)
        base_pontamadeiraw = base_pontamadeira.resample("W").apply(logic)
        base_1w = base_1.resample("W").apply(logic)
        base_3nw = base_3n.resample("W").apply(logic)
        base_3sw = base_3s.resample("W").apply(logic)
        base_4nw = base_4n.resample("W").apply(logic)
        base_4sw = base_4s.resample("W").apply(logic)
        base_tubaraow = base_tubarao.resample("W").apply(logic)
        base_2w = base_2.resample("W").apply(logic)
        base_1nw = base_1n.resample("W").apply(logic)
        base_1sw = base_1s.resample("W").apply(logic)

        for var in (
            base_guaibaw,
            base_sepetibaw,
            base_guaibasepetibaw,
            base_pontamadeiraw,
            base_1w,
            base_3nw,
            base_3sw,
            base_4nw,
            base_4sw,
            base_tubaraow,
            base_2w,
            base_1nw,
            base_1sw,
        ):
            var.index = var.index - pd.tseries.frequencies.to_offset("6D")

        # Feita a semanalizacao, recriamos as variaveis porto e pier

        for var in (
            (base_guaibaw, "Guaiba"),
            (base_sepetibaw, "Sepetiba"),
            (base_guaibasepetibaw, "Guaiba e Sepetiba"),
            (base_pontamadeiraw, "Ponta Madeira"),
            (base_1w, "Ponta Madeira"),
            (base_3nw, "Ponta Madeira"),
            (base_3sw, "Ponta Madeira"),
            (base_4nw, "Ponta Madeira"),
            (base_4sw, "Ponta Madeira"),
            (base_tubaraow, "Tubarao"),
            (base_2w, "Tubarao"),
            (base_1nw, "Tubarao"),
            (base_1sw, "Tubarao"),
        ):
            var[0]["Port"] = var[1]

        for var in (
            (base_guaibaw, "Total"),
            (base_sepetibaw, "Total"),
            (base_guaibasepetibaw, "Total"),
            (base_pontamadeiraw, "Total"),
            (base_1w, "1"),
            (base_3nw, "3n"),
            (base_3sw, "3s"),
            (base_4nw, "4n"),
            (base_4sw, "4s"),
            (base_tubaraow, "Total"),
            (base_2w, "2"),
            (base_1nw, "1n"),
            (base_1sw, "1s"),
        ):
            var[0]["Pier"] = var[1]

        # Por fim, vamos fazer o merge de todas as bases, criando uma base semanal

        base_semanal = pd.concat(
            [
                base_guaibaw,
                base_sepetibaw,
                base_guaibasepetibaw,
                base_pontamadeiraw,
                base_1w,
                base_3nw,
                base_3sw,
                base_4nw,
                base_4sw,
                base_tubaraow,
                base_2w,
                base_1nw,
                base_1sw,
            ],
            axis=0,
        )

        # Como, com a semanalizacao, os valores das variaveis que calculam as porcentagens
        # de navios por tipo, passam a nao fazer mais sentido, vamos refazer esses
        # calculos para a base semanal. Lembrendo que eles continuam funcionando
        # para a base diaria.

        base_semanal["Porcentagem FOB"] = (
            base_semanal["Numero de navios FOB"]
            / (
                base_semanal["Numero de navios FOB"]
                + base_semanal["Numero de navios CFR"]
            )
        ).fillna(0)

        base_semanal["Porcentagem CFR"] = (
            base_semanal["Numero de navios CFR"]
            / (
                base_semanal["Numero de navios FOB"]
                + base_semanal["Numero de navios CFR"]
            )
        ).fillna(0)

        base_semanal["Porcentagem SPOT/FOB"] = (
            base_semanal["Numero de navios SPOT/FOB"]
            / (
                base_semanal["Numero de navios SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/FOB"]
                + base_semanal["Numero de navios Frota Dedicada"]
            )
        ).fillna(0)

        base_semanal["Porcentagem Frota Dedicada/SPOT/FOB"] = (
            base_semanal["Numero de navios Frota Dedicada/SPOT/FOB"]
            / (
                base_semanal["Numero de navios SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/FOB"]
                + base_semanal["Numero de navios Frota Dedicada"]
            )
        ).fillna(0)

        base_semanal["Porcentagem Frota Dedicada/FOB"] = (
            base_semanal["Numero de navios Frota Dedicada/FOB"]
            / (
                base_semanal["Numero de navios SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/FOB"]
                + base_semanal["Numero de navios Frota Dedicada"]
            )
        ).fillna(0)

        base_semanal["Porcentagem Frota Dedicada"] = (
            base_semanal["Numero de navios Frota Dedicada"]
            / (
                base_semanal["Numero de navios SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/SPOT/FOB"]
                + base_semanal["Numero de navios Frota Dedicada/FOB"]
                + base_semanal["Numero de navios Frota Dedicada"]
            )
        ).fillna(0)

        base_semanal["Porcentagem PANAMAX"] = (
            base_semanal["Numero de navios PANAMAX"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        base_semanal["Porcentagem CAPE"] = (
            base_semanal["Numero de navios CAPE"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        base_semanal["Porcentagem VLOC"] = (
            base_semanal["Numero de navios VLOC"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        base_semanal["Porcentagem NEWCASTLE"] = (
            base_semanal["Numero de navios NEWCASTLE"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        base_semanal["Porcentagem VALEMAX"] = (
            base_semanal["Numero de navios VALEMAX"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        base_semanal["Porcentagem BABYCAPE"] = (
            base_semanal["Numero de navios BABYCAPE"]
            / (
                base_semanal["Numero de navios PANAMAX"]
                + base_semanal["Numero de navios CAPE"]
                + base_semanal["Numero de navios VLOC"]
                + base_semanal["Numero de navios NEWCASTLE"]
                + base_semanal["Numero de navios VALEMAX"]
                + base_semanal["Numero de navios BABYCAPE"]
            )
        ).fillna(0)

        # Vamos renomear a variavel Peso, que foi insumo para a variavel tempo de
        # estadia.

        base_semanal.rename(columns={"Peso": "Estadia em Horas"}, inplace=True)
        self.base_diaria = base_diaria_

        # Abaixo substituimos os valores 0 por valores vazios nas variavels Quantity
        # e Dwt, nas bases diaria e semanal. Tambem criamos a variavel Quantity/Dwt

        base_semanal[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_semanal[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_semanal[["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]] = base_semanal[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(0, np.nan)
        base_semanal["Quantity / Dwt"] = (
            base_semanal["Quantity (t)"] / base_semanal["Dwt (K) total"]
        )

        self.base_diaria[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ] = self.base_diaria[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(
            0, np.nan
        )
        self.base_diaria[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ] = self.base_diaria[
            ["Quantity (t)", "Dwt (K) total", "Dwt (K) medio"]
        ].replace(
            0, np.nan
        )
        self.base_diaria["Quantity / Dwt"] = (
            self.base_diaria["Quantity (t)"] / self.base_diaria["Dwt (K) total"]
        )

        coluna_port = base_semanal.pop("Port")
        coluna_pier = base_semanal.pop("Pier")
        base_semanal.insert(0, "Port", coluna_port)
        base_semanal.insert(0, "Pier", coluna_pier)

        return [base_semanal, self.base_diaria]


class Projecoes:
    """A classe Projecoes acrscenta previsoes de todas as variáveis com base
    em modelos prophet. A funcao acrescenta 1 ano e meio de previsoes
    ao último dia disponível dos valores reais.

    Args:

        base_diaria: Output da classe Semanalizacao.

        base_semanal: Output da classe Semanalizacao.

        data_final_real: Output da classe MergeDemurrageOperationalDesk.

    Returns:

        base_diaria: Base diária acrescida das previsoes.

        base_semanal: Base semanal acrescida das previsoes.
    """

    def __init__(
        self,
        base_diaria: pd.DataFrame,
        base_semanal: pd.DataFrame,
        data_final_real: str,
        data_inicial_real: str,
        data_inicio_projecoes: str,
        data_final_projecoes: str,
    ):
        self.base_diaria = base_diaria
        self.base_semanal = base_semanal
        self.data_final_real = data_final_real
        self.data_inicial_real = data_inicial_real
        self.data_inicio_projecoes = data_inicio_projecoes
        self.data_final_projecoes = data_final_projecoes

    def result(self):
        """Vamos fazer a previsao."""

        datas_semanais = pd.DataFrame(
            {
                "Day": pd.date_range(
                    start=self.data_inicial_real,
                    end=self.data_final_projecoes,
                    freq="W-MON",
                )
            }
        )

        datas_semanais_GS = datas_semanais.copy()
        datas_semanais_GS["Port"] = "Guaiba e Sepetiba"
        datas_semanais_GS["Pier"] = "Total"
        datas_semanais_PM = datas_semanais.copy()
        datas_semanais_PM["Port"] = "Ponta Madeira"
        datas_semanais_PM["Pier"] = "Total"
        datas_semanais_T = datas_semanais.copy()
        datas_semanais_T["Port"] = "Tubarao"
        datas_semanais_T["Pier"] = "Total"

        datas_semanais = pd.concat(
            [datas_semanais_GS, datas_semanais_PM, datas_semanais_T], axis=0
        )

        self.base_semanal = pd.merge(
            datas_semanais,
            self.base_semanal.reset_index(),
            on=["Day", "Port", "Pier"],
            how="outer",
        )

        self.base_semanal = self.base_semanal.set_index("Day")

        datas_diarias = pd.DataFrame(
            {
                "Day": pd.date_range(
                    start=self.data_inicial_real, end=self.data_final_projecoes
                )
            }
        )

        datas_diarias_GS = datas_diarias.copy()
        datas_diarias_GS["Port"] = "Guaiba e Sepetiba"
        datas_diarias_GS["Pier"] = "Total"
        datas_diarias_PM = datas_diarias.copy()
        datas_diarias_PM["Port"] = "Ponta Madeira"
        datas_diarias_PM["Pier"] = "Total"
        datas_diarias_T = datas_diarias.copy()
        datas_diarias_T["Port"] = "Tubarao"
        datas_diarias_T["Pier"] = "Total"

        datas_diarias = pd.concat(
            [datas_diarias_GS, datas_diarias_PM, datas_diarias_T], axis=0
        )

        self.base_diaria = pd.merge(
            datas_diarias,
            self.base_diaria.reset_index(),
            on=["Day", "Port", "Pier"],
            how="outer",
        )

        self.base_diaria = self.base_diaria.set_index("Day")

        self.base_semanal = previsao_prophet(
            self.base_semanal,
            self.data_inicial_real,
            self.data_inicio_projecoes,
            self.data_final_projecoes,
            "W",
        )
        self.base_diaria = previsao_prophet(
            self.base_diaria,
            self.data_inicial_real,
            self.data_inicio_projecoes,
            self.data_final_projecoes,
            "D",
        )

        self.base_diaria["Real/Previsto"] = 0
        self.base_diaria["Real/Previsto"] = np.where(
            self.base_diaria.index > pd.to_datetime(self.data_final_real),
            "Previsto",
            "Real",
        )

        return [self.base_diaria, self.base_semanal]


class NovosNomes:
    """A classe NovosNomes modifica os nomes das variáveis para os nomes que vamos
    utilizar na base de dados final.

    Args:

        base_diaria: Output da PremissaSimulado.

        base_semanal: Output da PremissaSimulado.

    Returns:

        base_diaria: base diária.

        base_semanal: base semanal.
    """

    def __init__(self, base_diaria: pd.DataFrame, base_semanal: pd.DataFrame):
        self.base_diaria = base_diaria
        self.base_semanal = base_semanal

    def base(self):
        """Modificamos os nomes e criamos algumas variaveis adcionais."""

        for var in self.base_diaria, self.base_semanal:
            var["Estadia em Horas"] = np.where(
                var["Estadia em Horas"] == 0, np.nan, var["Estadia em Horas"]
            )
            var["CAPACIDADE/Dwt"] = var["CAPACIDADE"] / var["Dwt (K) total"]
            var["Mudanca Politica"] = np.where(var.index.year >= 2023, 1, 0)
            var["Lag 1 CAPACIDADE"] = var.groupby(["Port", "Pier"])["CAPACIDADE"].shift(
                4
            )
            var["Lag 2 CAPACIDADE"] = var.groupby(["Port", "Pier"])["CAPACIDADE"].shift(
                8
            )
            var["Lag 3 CAPACIDADE"] = var.groupby(["Port", "Pier"])["CAPACIDADE"].shift(
                12
            )
            var["Lag 1 OEE"] = var.groupby(["Port", "Pier"])["OEE"].shift(4)
            var["Lag 1 DISPONIBILIDADE"] = var.groupby(["Port", "Pier"])[
                "DISPONIBILIDADE"
            ].shift(4)

        self.base_diaria[
            "Lag 1 mes numero de navios na fila"
        ] = self.base_diaria.groupby(["Port", "Pier"])[
            "Numero de navios na fila"
        ].shift(
            30
        )
        self.base_diaria[
            "Lag 2 meses numero de navios na fila"
        ] = self.base_diaria.groupby(["Port", "Pier"])[
            "Numero de navios na fila"
        ].shift(
            60
        )
        self.base_diaria[
            "Lag 3 meses numero de navios na fila"
        ] = self.base_diaria.groupby(["Port", "Pier"])[
            "Numero de navios na fila"
        ].shift(
            90
        )
        self.base_diaria["Quantity (t) copy"] = self.base_diaria["Quantity (t)"]

        self.base_semanal.reset_index(inplace=True)
        self.base_semanal["Day"] = pd.to_datetime(self.base_semanal["Day"])
        self.base_semanal["Período"] = np.where(
            (self.base_semanal["Port"] == "Ponta da Madeira")
            & (self.base_semanal["Day"].dt.month <= 6),
            "Chuvoso",
            np.where(
                (self.base_semanal["Port"] == "Ponta da Madeira")
                & (self.base_semanal["Day"].dt.month > 6),
                "Seco",
                np.where(
                    (self.base_semanal["Day"].dt.month <= 4)
                    | (self.base_semanal["Day"].dt.month >= 11),
                    "Chuvoso",
                    "Seco",
                ),
            ),
        )
        self.base_semanal = self.base_semanal.set_index("Day")

        # "Quantity (t)": "Quantity_t",

        # Linha removida: "Total Embarcado": "Volume_Embarcado",
        self.base_diaria.rename(
            columns={
                "Numero de navios na fila": "Navios na fila",
                "Numero de navios que chegaram": "Navios que chegaram",
                "Numero de navios TCA": "Navios TCA",
                "Numero de navios carregando": "Navios carregando",
                "Numero de navios desatracando": "Navios desatracando",
                "Lag 1 mes numero de navios na fila": "Lag 1 mes numero de navios na fila",
                "Lag 2 meses numero de navios na fila": "Lag 2 meses numero de navios na fila",
                "Lag 3 meses numero de navios na fila": "Lag 3 meses numero de navios na fila",
                "Lag 1 CAPACIDADE": "Lag 1 mes Capacidade",
                "Lag 2 CAPACIDADE": "Lag 2 meses Capacidade",
                "Lag 3 CAPACIDADE": "Lag 3 meses Capacidade",
                "Lag 1 OEE": "Lag 1 OEE",
                "Lag 1 DISPONIBILIDADE": "Lag 1 DISPONIBILIDADE",
                "Numero de navios FOB": "Navios FOB",
                "Numero de navios CFR": "Navios CFR",
                "Porcentagem FOB": "Pcg-FOB",
                "Porcentagem CFR": "Pcg-CFR",
                "Numero de navios PANAMAX": "Navios PANAMAX",
                "Porcentagem PANAMAX": "Pcg-PANAMAX",
                "Numero de navios CAPE": "Navios CAPE",
                "Porcentagem CAPE": "Pcg-CAPE",
                "Numero de navios VLOC": "Navios VLOC",
                "Porcentagem VLOC": "Pcg-VLOC",
                "Numero de navios NEWCASTLE": "Navios NEWCASTLE",
                "Porcentagem NEWCASTLE": "Pcg-NEWCASTLE",
                "Numero de navios VALEMAX": "Navios VALEMAX",
                "Porcentagem VALEMAX": "Pcg-VALEMAX",
                "Numero de navios BABYCAPE": "Navios BABYCAPE",
                "Porcentagem BABYCAPE": "Pcg-BABYCAPE",
                "Numero de navios SPOT/FOB": "Navios SPOT/FOB",
                "Porcentagem SPOT/FOB": "Pcg-SPOT/FOB",
                "Numero de navios Frota Dedicada/SPOT/FOB": "Navios Frota Dedicada/SPOT/FOB",
                "Porcentagem Frota Dedicada/SPOT/FOB": "Pcg-Frota Dedicada/SPOT/FOB",
                "Numero de navios Frota Dedicada/FOB": "Navios Frota Dedicada/FOB",
                "Porcentagem Frota Dedicada/FOB": "Pcg-Frota Dedicada/FOB",
                "Numero de navios Frota Dedicada": "Navios Frota Dedicada",
                "Porcentagem Frota Dedicada": "Pcg-Frota Dedicada",
                "Quantity (t)": "Volume_Embarcado",
                "Quantity (t) copy": "Quantity_t",
                "Dwt (K) total": "Dwt_K_total",
                "Dwt (K) medio": "Dwt_K_medio",
                "Quantity / Dwt": "Qtde/Dwt",
                "DISPONIBILIDADE": "DISPONIBILIDADE",
                "UTILIZACAO": "UTILIZACAO",
                "OEE": "OEE",
                "CAPACIDADE": "CAPACIDADE",
                "CAPACIDADE/Dwt": "CAPACIDADE/Dwt",
                "TAXA_EFETIVA": "TAXA_EFETIVA",
                "Multas por dia": "Multa_Demurrage",
                "Estadia em Horas": "Estadia_media_navios_hs",
                "Soma Estadia em Horas": "Soma Estadia em Horas",
            },
            inplace=True,
        )

        self.base_semanal["Quantity (t) copy"] = self.base_semanal["Quantity (t)"]
        # Linha Removida: "Total Embarcado": "Volume_Embarcado",
        self.base_semanal.rename(
            columns={
                "Lag numero de navios na fila": "Lag 1 mes numero de navios na fila",
                "Lag 1 CAPACIDADE": "Lag 1 mes Capacidade",
                "Lag 2 CAPACIDADE": "Lag 2 meses Capacidade",
                "Lag 3 CAPACIDADE": "Lag 3 meses Capacidade",
                "Lag 1 OEE": "Lag 1 OEE",
                "Lag 1 DISPONIBILIDADE": "Lag 1 DISPONIBILIDADE",
                "Numero de navios FOB": "Navios FOB",
                "Numero de navios CFR": "Navios CFR",
                "Porcentagem FOB": "Pcg-FOB",
                "Porcentagem CFR": "Pcg-CFR",
                "Numero de navios PANAMAX": "Navios PANAMAX",
                "Porcentagem PANAMAX": "Pcg-PANAMAX",
                "Numero de navios CAPE": "Navios CAPE",
                "Porcentagem CAPE": "Pcg-CAPE",
                "Numero de navios VLOC": "Navios VLOC",
                "Porcentagem VLOC": "Pcg-VLOC",
                "Numero de navios NEWCASTLE": "Navios NEWCASTLE",
                "Porcentagem NEWCASTLE": "Pcg-NEWCASTLE",
                "Numero de navios VALEMAX": "Navios VALEMAX",
                "Porcentagem VALEMAX": "Pcg-VALEMAX",
                "Numero de navios BABYCAPE": "Navios BABYCAPE",
                "Porcentagem BABYCAPE": "Pcg-BABYCAPE",
                "Numero de navios SPOT/FOB": "Navios SPOT/FOB",
                "Porcentagem SPOT/FOB": "Pcg-SPOT/FOB",
                "Numero de navios Frota Dedicada/SPOT/FOB": "Navios Frota Dedicada/SPOT/FOB",
                "Porcentagem Frota Dedicada/SPOT/FOB": "Pcg-Frota Dedicada/SPOT/FOB",
                "Numero de navios Frota Dedicada/FOB": "Navios Frota Dedicada/FOB",
                "Porcentagem Frota Dedicada/FOB": "Pcg-Frota Dedicada/FOB",
                "Numero de navios Frota Dedicada": "Navios Frota Dedicada",
                "Porcentagem Frota Dedicada": "Pcg-Frota Dedicada",
                "Quantity (t)": "Volume_Embarcado",
                "Quantity (t) copy": "Quantity_t",
                "Dwt (K) total": "Dwt_K_total",
                "Dwt (K) medio": "Dwt_K_medio",
                "Quantity / Dwt": "Qtde/Dwt",
                "DISPONIBILIDADE": "DISPONIBILIDADE",
                "UTILIZACAO": "UTILIZACAO",
                "OEE": "OEE",
                "CAPACIDADE": "CAPACIDADE",
                "CAPACIDADE/Dwt": "CAPACIDADE/Dwt",
                "TAXA_EFETIVA": "TAXA_EFETIVA",
                "Multas por dia": "Multa_Demurrage",
                "Estadia em Horas": "Estadia_media_navios_hs",
                "Soma Estadia em Horas": "Soma Estadia em Horas",
            },
            inplace=True,
        )

        return [self.base_diaria, self.base_semanal]
