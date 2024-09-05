import os
import glob
import shutil
import calendar
from datetime import date
import shap as s
import numpy as np
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def shap_values_fun(mdl, train_x):
    """Essa funcao faz o cálculo dos valores de shap.
    """

    explainer = s.TreeExplainer(mdl)
    shap_values = explainer.shap_values(train_x)
    expected_value = explainer.expected_value
    return shap_values, expected_value


class Clusterizacao:
    """A classe Clusterizacao gera como output todas as bases de dados utilizadas
    como inputs do Power Bi.

    Args:

        base_semanal: Base com os dados semanais.

        base_diaria: Base com os dados diários.

        df_demurrage_pm: Previsao da variável custo de demurrage para o porto de
        Ponta Madeira.

        df_estadia_pm: Previsao da variável tempo de estadia para o porto de
        Ponta Madeira.

        df_volume_pm: Previsao da variável volume para o porto de Ponta Madeira.

        df_demurrage_gs: Previsao da variável custo de demurrage para o porto de
        Guaiba e Sepetiba.

        df_estadia_gs: Previsao da variável tempo de estadia para o porto de
        Guaiba e Sepetiba.

        df_volume_gs: Previsao da variável volume para o porto de Guaiba e Sepetiba.

        df_demurrage_tb: Previsao da variável custo de demurrage para o porto de
        Tubarao.

        df_estadia_tb: Previsao da variável tempo de estadia para o porto de
        Tubarao.

        df_volume_tb: Previsao da variável volume para o porto de Tubarao.

        tuning_: Valor 0 ou 1, sendo 1 para executar o tuning dos modelos random
        forest e 0 caso nao fizermos o tuning, importando os hiperparametros do
        último tuning.

        DATA_FIM_TESTE: Data final para os dados que compreendem a base de teste.

        DATA_INICIO_TESTE: Data início para os dados que compreendem a base de teste.

        path: Caminhos para a pasta onde sao salvas as bases inputs do power bi.

        hp_rf_cluster: Huperparametros correspondentes ao último tuning feito
        para os modelos de random forest.

        historico: Histórico de previsoes das variáveis Custo de Demurrage, Tempo
        de Estadia e Volume para todos os portos.

    Returns:

        base1: Base a nível mensal.

        base2: Base a nível diário.

        base3: Base a nível mensal.

        base4: Base mensal com valores máximo e mínimo para cada mes do ano e
        cada variável.

        base5: Base semanal agrupada para que possamos plicar filtros no power
        bi.

        base8: Base com valores de SHAP.

        base9: Base com valores de corte para definir influencias positivas e
        negativas que cada variável x tem sobre as variáveis y.

        hp_rf_cluster: Valores de hiperparametros correspondentes as previsoes
        de demurrage, estadia e volume com a melhor acurácia.

        historico: Histórico de previsoes de Demurrage, Estadia e Volume.

    """
    def __init__(
        self,
        base_semanal,
        base_diaria,
        df_demurrage_pm,
        df_estadia_pm,
        df_volume_pm,
        df_demurrage_gs,
        df_estadia_gs,
        df_volume_gs,
        df_demurrage_tb,
        df_estadia_tb,
        df_volume_tb,
        tuning_,
        DATA_FIM_TESTE,
        DATA_INICIO_TESTE,
        path,
        hp_rf_cluster,
        historico,
    ):
        self.base_semanal = base_semanal
        self.base_diaria = base_diaria
        self.df_demurrage_pm = df_demurrage_pm
        self.df_estadia_pm = df_estadia_pm
        self.df_volume_pm = df_volume_pm
        self.df_demurrage_gs = df_demurrage_gs
        self.df_estadia_gs = df_estadia_gs
        self.df_volume_gs = df_volume_gs
        self.df_demurrage_tb = df_demurrage_tb
        self.df_estadia_tb = df_estadia_tb
        self.df_volume_tb = df_volume_tb
        self.tuning_ = tuning_
        self.DATA_FIM_TESTE = DATA_FIM_TESTE
        self.DATA_INICIO_TESTE = DATA_INICIO_TESTE
        self.path = path
        self.hp_rf_cluster = hp_rf_cluster
        self.historico = historico

    def resultado(self):

        #remocao_dados_reais_extras = "2023-10-01"
        del self.base_semanal["Período"]

        # bases

        mes_final_treino = int(dt.strptime(self.DATA_FIM_TESTE, "%Y-%m-%d").month)
        ano_final_treino = int(dt.strptime(self.DATA_FIM_TESTE, "%Y-%m-%d").day)

        # Agrupamento por porto
        df_pm = pd.concat(
            [self.df_demurrage_pm, self.df_estadia_pm, self.df_volume_pm], axis=1
        )
        df_gs = pd.concat(
            [self.df_demurrage_gs, self.df_estadia_gs, self.df_volume_gs], axis=1
        )
        df_tb = pd.concat(
            [self.df_demurrage_tb, self.df_estadia_tb, self.df_volume_tb], axis=1
        )

        # Tratamento
        df_pm = df_pm.drop(
            columns=[
                "prophet_t",
                "prophet_v",
                "prophet_d",
                "Multa_Demurrage",
                "Estadia_media_navios_hs",
                "Volume_Embarcado",
            ]
        )
        df_gs = df_gs.drop(
            columns=[
                "prophet_t",
                "prophet_v",
                "prophet_d",
                "Multa_Demurrage",
                "Estadia_media_navios_hs",
                "Volume_Embarcado",
            ]
        )
        df_tb = df_tb.drop(
            columns=[
                "prophet_t",
                "prophet_v",
                "prophet_d",
                "Multa_Demurrage",
                "Estadia_media_navios_hs",
                "Volume_Embarcado",
            ]
        )

        # Data final dos dados Reais

        df_pm["Porto"] = "Ponta da Madeira"
        df_gs["Porto"] = "Guaíba e Sepetiba"
        df_tb["Porto"] = "Tubarão"

        df_pm["Variavel"] = "Previsto"
        df_gs["Variavel"] = "Previsto"
        df_tb["Variavel"] = "Previsto"
        df = pd.concat([df_pm, df_gs, df_tb], axis=0)

        df.columns.values[1] = "Multa_Demurrage"
        df.columns.values[3] = "Estadia_media_navios_hs"
        df.columns.values[5] = "Volume_Embarcado"

        #df_prev = df.loc[:, ~df.columns.duplicated()]
        df = df.loc[:, ~df.columns.duplicated()]

        df_semanal = self.base_semanal

        df_semanal = df_semanal.rename({"Port": "Porto"}, axis="columns")
        df_semanal = df_semanal.replace(
            {
                "Ponta Madeira": "Ponta da Madeira",
                "Tubarao": "Tubarão",
                "Guaiba": "Guaíba",
                "Guaiba e Sepetiba": "Guaíba e Sepetiba",
            }
        )

        df_semanal1 = df_semanal[
            (df_semanal["Day"] <= pd.to_datetime(self.DATA_FIM_TESTE))
            & (df_semanal["Pier"] == "Total")
        ]
        df_semanal2 = df_semanal[
            (df_semanal["Day"] > pd.to_datetime(self.DATA_FIM_TESTE))
            & (df_semanal["Pier"] == "Total")
        ]
        df_semanal = df_semanal[(df_semanal["Pier"] == "Total")]
        df_semanal1 = df_semanal1.iloc[:, :6]
        df_semanal2 = df_semanal2.iloc[:, :6]

        df_semanal1["Variavel"] = "Real"
        df_semanal2["Variavel"] = "Médias"

        df_final = pd.concat([df_semanal1, df, df_semanal2], axis=0)

        df_final = df_final.drop(columns=["Pier"])
        # df_semanal['Porto'].unique()

        df_semanal = df_semanal.drop(
            columns=[
                "Pier",
                "Multa_Demurrage",
                "Estadia_media_navios_hs",
                "Volume_Embarcado",
            ]
        )

        df = pd.merge(df_final, df_semanal, how="inner", on=["Day", "Porto"])

        df_periodo = df.copy()

        df["Periodo"] = "Seco e Chuvoso"

        # Criar a função para determinar o período com base no mês e no porto
        def determinar_periodo(row):
            if row["Porto"] == "Ponta da Madeira":
                if 1 <= row["Day"].month <= 6:
                    return "Chuvoso"
                else:
                    return "Seco"
            else:
                if (
                    1 <= row["Day"].month <= 4
                    or row["Day"].month == 11
                    or row["Day"].month == 12
                ):
                    return "Chuvoso"
                else:
                    return "Seco"

        # Aplicar a função em cada linha do DataFrame para criar a coluna "Periodo"
        df_periodo["Periodo"] = df_periodo.apply(determinar_periodo, axis=1)

        df1 = pd.concat([df, df_periodo])

        """
        Mensalização
        """

        df["ds"] = pd.to_datetime(df["Day"])  # Transformando para o formato date

        # Criação das variáveis que serão utilizadas para calcular as mensalizações
        df["semana_numero"] = (df["ds"].dt.isocalendar().week).astype(int)
        df["ano"] = (df["ds"].dt.isocalendar().year).astype(int)
        df["mes_inicio"] = df["ds"].dt.month

        # Função que calcula o último dia da semana
        def last_day_of_weekt(dt):
            return dt + relativedelta(days=6)

        # Criação das variáveis que serão utilizadas para calcular as mensalizações
        df["semana_fim"] = df.apply(
            lambda x: last_day_of_weekt(x["ds"]), axis=1
        )  # Aplicação da função que calcula o último dia da semana
        df["semana_fim"] = pd.to_datetime(df["semana_fim"])
        df["semana_fim"] = df["semana_fim"].dt.normalize()
        df["mes_fim"] = df["semana_fim"].dt.month
        df["day_inicio"] = df["ds"].dt.day
        df["day_fim"] = df["semana_fim"].dt.day
        df["index"] = df.index

        df["veri"] = df.apply(lambda x: (x["mes_inicio"] - x["mes_fim"]), axis=1)

        # Atribuição dos pesos
        df1_1 = df[
            df["veri"] == 0
        ]  # Caso a semana não possua datas de meses diferentes
        df1_1["pesos"] = 7  # Peso 7 pois os 7 dias da semana estão no mesmo mês
        df1_1["mes"] = df1_1["ds"].dt.month
        df1_1["ano_peso"] = df1_1["ds"].dt.year
        df2 = df[df["veri"] != 0]  # Caso a semana possua data de meses diferentes
        df2["pesos"] = df2[
            "semana_fim"
        ].dt.day  # Quantidade de dias da semana do mês posterior
        df2["mes"] = df2["semana_fim"].dt.month
        df2["ano_peso"] = df2["semana_fim"].dt.year
        df3 = df[df["veri"] != 0]  # Caso a semana possua data de meses diferentes
        df3["pesos"] = df3.apply(
            lambda x: (7 - x["day_fim"]), axis=1
        )  # Quantidade de dias da semana do mês anterior
        df3["mes"] = df3["ds"].dt.month
        df3["ano_peso"] = df3["ds"].dt.year

        # Junção dos dataframes criados
        frames = [df1_1, df2, df3]
        df4 = pd.concat(frames)
        df_final = df4.sort_values("index", axis=0, ascending=True)

        # Cálculo dos valores mensalizados
        df_final["mes_ano"] = (
            df_final["mes"].map(str) + "-" + df_final["ano_peso"].map(str)
        )  # Junção do mês e do ano, para posteriormente fazer o agrupamento mensal

        # Código que pega o último dia do mês
        ultimo_dia_treino = calendar.monthrange(ano_final_treino, mes_final_treino)[1]

        # último dia do treino do modelo
        data_limite = pd.to_datetime(
            f"{ano_final_treino}-{mes_final_treino}-{ultimo_dia_treino}", yearfirst=True
        )

        # Função que calcula o último dia do mês quando recebe os dados de ano e mes
        def ultimo_dia_do_mes(ano, mes):
            ultimo_dia = calendar.monthrange(ano, mes)[1]
            return pd.to_datetime(f"{ano}-{mes}-{ultimo_dia}", yearfirst=True)

        # Aplicando a função acima para todas as linhas do df_final
        df_final["ultimo_dia_mes"] = df_final.apply(
            lambda x: ultimo_dia_do_mes(x["ano_peso"], x["mes"]), axis=1
        )

        # Remove as linhas Reais que não deveriam existir
        df_final = df_final[
            ~(
                (df_final["Variavel"] == "Real")
                & (df_final["ultimo_dia_mes"] > data_limite)
            )
        ]

        for i in ["Multa_Demurrage", "Volume_Embarcado"]:
            df_final[i] = (df_final[i] * df_final["pesos"]) / 7

        df_final["Estadia_media_navios_hs"] = (
            df_final["Estadia_media_navios_hs"] * df_final["pesos"]
        )

        # Tranformação da variável mes_ano para o formato data
        df_final["mes_ano"] = pd.to_datetime(df_final["mes_ano"])
        df_final["mes_ano"] = df_final["mes_ano"].dt.normalize()

        # Junção dos valores por mês
        df_final1 = df_final.groupby(
            ["mes_ano", "Porto", "Variavel", "Periodo"], as_index=False, sort=False
        ).agg(
            {
                **{
                    col: lambda x: x.mean()
                    for col in df_final.columns
                    if col
                    not in [
                        "mes_ano",
                        "Porto",
                        "Variavel",
                        "Periodo",
                        "pesos",
                        "Multa_Demurrage",
                        "Volume_Embarcado",
                        "Estadia_media_navios_hs",
                        "Demurrage Unitária",
                        "Dwt_K_medio",
                        "Dwt_K_total",
                    ]
                    and pd.api.types.is_numeric_dtype(df_final[col])
                },
                "Multa_Demurrage": "sum",
                "Volume_Embarcado": "sum",
                "pesos": "sum",
                "Estadia_media_navios_hs": "sum",
                "Dwt_K_medio": "mean",
                "Dwt_K_total": "sum",
            }
        )

        df_final1["Estadia_media_navios_hs"] = (
            df_final1["Estadia_media_navios_hs"] / df_final1["pesos"]
        )

        #df_final1 = df_final1.drop(
        #    df_final1[
        #        (df_final1["mes_ano"] == remocao_dados_reais_extras)
        #        & (df_final1["Variavel"] == "Real")
        #    ].index
        #)

        df_periodo_mensal = df_final1.copy()

        # Criar a função para determinar o período com base no mês e no porto
        def determinar_periodo(row):
            if row["Porto"] == "Ponta da Madeira":
                if 1 <= row["mes_ano"].month <= 6:
                    return "Chuvoso"
                else:
                    return "Seco"
            else:
                if (
                    1 <= row["mes_ano"].month <= 4
                    or row["mes_ano"].month == 11
                    or row["mes_ano"].month == 12
                ):
                    return "Chuvoso"
                else:
                    return "Seco"

        # Aplicar a função em cada linha do DataFrame para criar a coluna "Periodo"
        df_periodo_mensal["Periodo"] = df_periodo_mensal.apply(
            determinar_periodo, axis=1
        )

        df_mensal = pd.concat([df_final1, df_periodo_mensal])

        df_mensal["Tipo"] = "Semanal"

        df_diario = self.base_diaria

        df_diario["mes_ano"] = df_diario["Day"].dt.strftime("%Y-%m")
        df_diario = df_diario.rename({"Port": "Porto"}, axis="columns")
        df_diario = df_diario[df_diario["Pier"] == "Total"]
        df_diario = df_diario.replace(
            {
                "Ponta Madeira": "Ponta da Madeira",
                "Tubarao": "Tubarão",
                "Guaiba": "Guaíba",
                "Guaiba e Sepetiba": "Guaíba e Sepetiba",
            }
        )

        df_diario_mensal = df_diario.groupby(["mes_ano", "Porto"], as_index=False).agg(
            {
                **{
                    col: lambda x: x.mean()
                    for col in df_diario.columns
                    if col
                    not in ["mes_ano", "Porto", "Multa_Demurrage", "Volume_Embarcado"]
                    and pd.api.types.is_numeric_dtype(df_diario[col])
                },
                "Multa_Demurrage": "sum",
                "Volume_Embarcado": "sum",
                "Soma Estadia em Horas": "sum",
                "Navios TCA": "sum",
            }
        )

        df_diario_mensal["Multa_Demurrage"] = abs(df_diario_mensal["Multa_Demurrage"])

        df_diario_mensal["Variavel"] = np.where(
            df_diario_mensal["mes_ano"]
            < self.DATA_FIM_TESTE[0:4] + "-" + self.DATA_FIM_TESTE[5:7],
            "Real",
            "Médias",
        )

        df_diario_mensal["Tipo"] = "Diário"

        df_diario_mensal["mes_ano"] = pd.to_datetime(df_diario_mensal["mes_ano"])

        df_diario_mensal1 = df_diario_mensal.copy()
        df_diario_mensal["Periodo"] = "Seco e Chuvoso"

        # Aplicar a função em cada linha do DataFrame para criar a coluna "Periodo"
        df_diario_mensal1["Periodo"] = df_diario_mensal1.apply(
            determinar_periodo, axis=1
        )

        df_diario_mensal = pd.concat([df_diario_mensal, df_diario_mensal1])

        df_mensal = pd.concat([df_mensal, df_diario_mensal])

        df_mensal = df_mensal.rename(
            {
                "Multa_Demurrage": "Demurrage",
                "Estadia_media_navios_hs": "Estadia",
                "Volume_Embarcado": "Volume",
                "Lag 1 mes numero de navios na fila": "Navios em fila - mês anterior",
                "Lag 2 meses numero de navios na fila": "Navios em fila - 2 meses anteriores",
                "Lag 3 meses numero de navios na fila": "Navios em fila - 3 meses anteriores",
                "Lag 1 mes Capacidade": "Capacidade - mês anterior",
                "Lag 2 meses Capacidade": "Capacidade - 2 meses anteriores",
                "Lag 3 meses Capacidade": "Capacidade - 3 meses anteriores",
                "Lag 1 OEE": "OEE - mês anterior",
                "Lag 1 DISPONIBILIDADE": "Disponibilidade - mês anterior",
            },
            axis="columns",
        )

        df_mensal["Demurrage"] = abs(df_mensal["Demurrage"])

        df_mensal["Demurrage Unitária"] = df_mensal["Demurrage"] / df_mensal["Volume"]

        df_mensal["Calcular_MAPE"] = np.where(
            (
                df_mensal["mes_ano"]
                >= pd.to_datetime(self.DATA_FIM_TESTE) + relativedelta(months=-6)
            )
            & (df_mensal["mes_ano"] <= self.DATA_FIM_TESTE)
            & (df_mensal["Tipo"] == "Semanal"),
            1,
            0,
        )

        # df_mensal.to_excel(f"C:/Users/KG858HY/OneDrive - EY/Desktop/Scripts/Output/base_previsao_mensal_{date.today()}.xlsx")

        self.historico = self.historico[
            [
                "mes_ano",
                "Porto",
                "Variavel",
                "Periodo",
                "Demurrage",
                "Volume",
                "Estadia",
                "Tipo",
            ]
        ]
        self.historico.rename(
            columns={
                "Demurrage": "Demurrage_hist",
                "Volume": "Volume_hist",
                "Estadia": "Estadia_hist",
            },
            inplace=True,
        )

        df_mensal = pd.merge(
            df_mensal,
            self.historico,
            how="outer",
            on=["mes_ano", "Porto", "Variavel", "Periodo", "Tipo"],
        )

        df_mensal["Demurrage"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Demurrage_hist"],
            df_mensal["Demurrage"],
        )
        df_mensal["Volume"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Volume_hist"],
            df_mensal["Volume"],
        )
        df_mensal["Estadia"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Estadia_hist"],
            df_mensal["Estadia"],
        )

        df_mensal["Demurrage_hist"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Demurrage"],
            df_mensal["Demurrage_hist"],
        )
        df_mensal["Volume_hist"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Volume"],
            df_mensal["Volume_hist"],
        )
        df_mensal["Estadia_hist"] = np.where(
            (df_mensal["mes_ano"] < self.DATA_INICIO_TESTE)
            & (df_mensal["Variavel"] == "Previsto")
            & (df_mensal["Tipo"] == "Semanal"),
            df_mensal["Estadia"],
            df_mensal["Estadia_hist"],
        )

        df_mensal["Demurrage Unitária"] = df_mensal["Demurrage"] / df_mensal["Volume"]

        self.historico = df_mensal[
            [
                "mes_ano",
                "Porto",
                "Variavel",
                "Periodo",
                "Demurrage_hist",
                "Volume_hist",
                "Estadia_hist",
                "Tipo",
            ]
        ]
        self.historico.rename(
            columns={
                "Demurrage_hist": "Demurrage",
                "Volume_hist": "Volume",
                "Estadia_hist": "Estadia",
            },
            inplace=True,
        )
        self.historico = self.historico.dropna()
        self.historico = self.historico[self.historico["Variavel"] == "Previsto"]

        del df_mensal["Demurrage_hist"]
        del df_mensal["Volume_hist"]
        del df_mensal["Estadia_hist"]

        df_mensal = df_mensal.dropna(subset=["Demurrage", "Volume", "Estadia"])

        base1 = df_mensal

        """
        Base Diária
        """

        df_diaria = self.base_diaria
        df_diaria_1 = df_diaria.copy()

        df_diaria["Periodo"] = "Seco e Chuvoso"

        df_diaria

        # Criar a função para determinar o período com base no mês e no porto
        def determinar_periodo(row):
            if row["Port"] == "Ponta Madeira":
                if 1 <= row["Day"].month <= 6:
                    return "Chuvoso"
                else:
                    return "Seco"
            else:
                if (
                    1 <= row["Day"].month <= 4
                    or row["Day"].month == 11
                    or row["Day"].month == 12
                ):
                    return "Chuvoso"
                else:
                    return "Seco"

        # Aplicar a função em cada linha do DataFrame para criar a coluna "Periodo"
        df_diaria_1["Periodo"] = df_diaria_1.apply(determinar_periodo, axis=1)

        df_day = pd.concat([df_diaria, df_diaria_1])

        df_day = df_day.rename(
            {
                "Port": "Porto",
                "Multa_Demurrage": "Demurrage",
                "Estadia_media_navios_hs": "Estadia",
                "Volume_Embarcado": "Volume",
                "Lag 1 mes numero de navios na fila": "Navios em fila - mês anterior",
                "Lag 2 meses numero de navios na fila": "Navios em fila - 2 meses anteriores",
                "Lag 3 meses numero de navios na fila": "Navios em fila - 3 meses anteriores",
                "Lag 1 mes Capacidade": "Capacidade - mês anterior",
                "Lag 2 meses Capacidade": "Capacidade - 2 meses anteriores",
                "Lag 3 meses Capacidade": "Capacidade - 3 meses anteriores",
                "Lag 1 OEE": "OEE - mês anterior",
                "Lag 1 DISPONIBILIDADE": "Disponibilidade - mês anterior",
            },
            axis="columns",
        )

        df_day = df_day.replace(
            {
                "Ponta Madeira": "Ponta da Madeira",
                "Tubarao": "Tubarão",
                "Guaiba": "Guaíba",
                "Guaiba e Sepetiba": "Guaíba e Sepetiba",
            }
        )

        df_day["Demurrage Unitária"] = df_day["Demurrage"] / df_day["Volume"]

        # df_day.to_excel(f"C:/Users/KG858HY/OneDrive - EY/Desktop/Scripts/Output/self.base_diaria_{date.today()}.xlsx")
        base2 = df_day

        """
        Base Variáveis

        """

        df_variaveis = df_mensal.copy()

        df_melted_mensal = pd.melt(
            df_variaveis,
            id_vars=[
                "mes_ano",
                "Porto",
                "Variavel",
                "Periodo",
                "Tipo",
                "Calcular_MAPE",
            ],
            value_vars=["Demurrage", "Volume", "Estadia", "Demurrage Unitária"],
            var_name="Análise",
            value_name="Valor",
        )

        base3 = df_melted_mensal

        """
        Base Variáveis min x max
        """
        df_min_max = df_mensal.copy()
        for i in [
            "Navios em fila - mês anterior",
            "Navios em fila - 2 meses anteriores",
            "Navios em fila - 3 meses anteriores",
            "Capacidade - mês anterior",
            "Capacidade - 2 meses anteriores",
            "Capacidade - 3 meses anteriores",
            "OEE - mês anterior",
            "Disponibilidade - mês anterior",
            "Navios FOB",
            "Navios CFR",
            "Pcg-FOB",
            "Pcg-CFR",
            "Navios PANAMAX",
            "Pcg-PANAMAX",
            "Navios CAPE",
            "Pcg-CAPE",
            "Navios VLOC",
            "Pcg-VLOC",
            "Navios NEWCASTLE",
            "Pcg-NEWCASTLE",
            "Navios VALEMAX",
            "Pcg-VALEMAX",
            "Navios BABYCAPE",
            "Pcg-BABYCAPE",
            "Navios SPOT/FOB",
            "Pcg-SPOT/FOB",
            "Navios Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Navios Frota Dedicada/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Navios Frota Dedicada",
            "Pcg-Frota Dedicada",
            "Quantity_t",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Qtde/Dwt",
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "CAPACIDADE",
            "CAPACIDADE/Dwt",
            "TAXA_EFETIVA",
            "Demurrage",
            "Volume",
            "Estadia",
            "Demurrage Unitária",
        ]:
            df_min_max["minimo" + i] = df_min_max[i].min()
            df_min_max["máximo" + i] = df_min_max[i].max()

        list_var = [
            "mes_ano",
            "Porto",
            "Variavel",
            "Periodo",
            "Tipo",
            "Navios em fila - mês anterior",
            "Navios em fila - 2 meses anteriores",
            "Navios em fila - 3 meses anteriores",
            "Capacidade - mês anterior",
            "Capacidade - 2 meses anteriores",
            "Capacidade - 3 meses anteriores",
            "OEE - mês anterior",
            "Disponibilidade - mês anterior",
            "Navios FOB",
            "Navios CFR",
            "Pcg-FOB",
            "Pcg-CFR",
            "Navios PANAMAX",
            "Pcg-PANAMAX",
            "Navios CAPE",
            "Pcg-CAPE",
            "Navios VLOC",
            "Pcg-VLOC",
            "Navios NEWCASTLE",
            "Pcg-NEWCASTLE",
            "Navios VALEMAX",
            "Pcg-VALEMAX",
            "Navios BABYCAPE",
            "Pcg-BABYCAPE",
            "Navios SPOT/FOB",
            "Pcg-SPOT/FOB",
            "Navios Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Navios Frota Dedicada/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Navios Frota Dedicada",
            "Pcg-Frota Dedicada",
            "Quantity_t",
            "Dwt_K_medio",
            "Dwt_K_total",
            "Qtde/Dwt",
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "CAPACIDADE",
            "CAPACIDADE/Dwt",
            "TAXA_EFETIVA",
            "Mudanca Politica",
            "Demurrage",
            "Volume",
            "Estadia",
            "Demurrage Unitária",
        ]

        list_max = [
            "máximoNavios em fila - mês anterior",
            "máximoNavios em fila - 2 meses anteriores",
            "máximoNavios em fila - 3 meses anteriores",
            "máximoCapacidade - mês anterior",
            "máximoCapacidade - 2 meses anteriores",
            "máximoCapacidade - 3 meses anteriores",
            "máximoOEE - mês anterior",
            "máximoDisponibilidade - mês anterior",
            "máximoNavios FOB",
            "máximoNavios CFR",
            "máximoPcg-FOB",
            "máximoPcg-CFR",
            "máximoNavios PANAMAX",
            "máximoPcg-PANAMAX",
            "máximoNavios CAPE",
            "máximoPcg-CAPE",
            "máximoNavios VLOC",
            "máximoPcg-VLOC",
            "máximoNavios NEWCASTLE",
            "máximoPcg-NEWCASTLE",
            "máximoNavios VALEMAX",
            "máximoPcg-VALEMAX",
            "máximoNavios BABYCAPE",
            "máximoPcg-BABYCAPE",
            "máximoNavios SPOT/FOB",
            "máximoPcg-SPOT/FOB",
            "máximoNavios Frota Dedicada/SPOT/FOB",
            "máximoPcg-Frota Dedicada/SPOT/FOB",
            "máximoNavios Frota Dedicada/FOB",
            "máximoPcg-Frota Dedicada/FOB",
            "máximoNavios Frota Dedicada",
            "máximoPcg-Frota Dedicada",
            "máximoQuantity_t",
            "máximoDwt_K_medio",
            "máximoDwt_K_total",
            "máximoQtde/Dwt",
            "máximoDISPONIBILIDADE",
            "máximoUTILIZACAO",
            "máximoOEE",
            "máximoCAPACIDADE",
            "máximoCAPACIDADE/Dwt",
            "máximoTAXA_EFETIVA",
            "máximoDemurrage",
            "máximoVolume",
            "máximoEstadia",
            "máximoDemurrage Unitária",
        ]

        list_min = [
            "minimoNavios em fila - mês anterior",
            "minimoNavios em fila - 2 meses anteriores",
            "minimoNavios em fila - 3 meses anteriores",
            "minimoCapacidade - mês anterior",
            "minimoCapacidade - 2 meses anteriores",
            "minimoCapacidade - 3 meses anteriores",
            "minimoOEE - mês anterior",
            "minimoDisponibilidade - mês anterior",
            "minimoNavios FOB",
            "minimoNavios CFR",
            "minimoPcg-FOB",
            "minimoPcg-CFR",
            "minimoNavios PANAMAX",
            "minimoPcg-PANAMAX",
            "minimoNavios CAPE",
            "minimoPcg-CAPE",
            "minimoNavios VLOC",
            "minimoPcg-VLOC",
            "minimoNavios NEWCASTLE",
            "minimoPcg-NEWCASTLE",
            "minimoNavios VALEMAX",
            "minimoPcg-VALEMAX",
            "minimoNavios BABYCAPE",
            "minimoPcg-BABYCAPE",
            "minimoNavios SPOT/FOB",
            "minimoPcg-SPOT/FOB",
            "minimoNavios Frota Dedicada/SPOT/FOB",
            "minimoPcg-Frota Dedicada/SPOT/FOB",
            "minimoNavios Frota Dedicada/FOB",
            "minimoPcg-Frota Dedicada/FOB",
            "minimoNavios Frota Dedicada",
            "minimoPcg-Frota Dedicada",
            "minimoQuantity_t",
            "minimoDwt_K_medio",
            "minimoDwt_K_total",
            "minimoQtde/Dwt",
            "minimoDISPONIBILIDADE",
            "minimoUTILIZACAO",
            "minimoOEE",
            "minimoCAPACIDADE",
            "minimoCAPACIDADE/Dwt",
            "minimoTAXA_EFETIVA",
            "minimoDemurrage",
            "minimoVolume",
            "minimoEstadia",
            "minimoDemurrage Unitária",
        ]

        df_melted_mensal_max = pd.melt(
            df_min_max,
            id_vars=list_var,
            value_vars=list_max,
            var_name="Maximo",
            value_name="Valor maximo",
        )
        df_melted_mensal_min = pd.melt(
            df_min_max,
            id_vars=list_var,
            value_vars=list_min,
            var_name="Minimo",
            value_name="Valor minimo",
        )
        df_melted_mensal_min = df_melted_mensal_min[["Minimo", "Valor minimo"]]

        df_min_max = pd.concat([df_melted_mensal_max, df_melted_mensal_min], axis=1)

        # df_min_max.to_excel(f"C:/Users/KG858HY/OneDrive - EY/Desktop/Scripts/Output/base_previsao_mensal_min_max_{date.today()}.xlsx")
        base4 = df_min_max

        """
        Correlação
        """

        # Base Semanal
        df = self.base_semanal
        df = df.rename(
            {
                "Multa_Demurrage": "Demurrage",
                "Estadia_media_navios_hs": "Estadia",
                "Volume_Embarcado": "Volume",
                "Lag 1 mes numero de navios na fila": "Navios em fila - mês anterior",
                "Lag 2 meses numero de navios na fila": "Navios em fila - 2 meses anteriores",
                "Lag 3 meses numero de navios na fila": "Navios em fila - 3 meses anteriores",
                "Lag 1 mes Capacidade": "Capacidade - mês anterior",
                "Lag 2 meses Capacidade": "Capacidade - 2 meses anteriores",
                "Lag 3 meses Capacidade": "Capacidade - 3 meses anteriores",
                "Lag 1 OEE": "OEE - mês anterior",
                "Lag 1 DISPONIBILIDADE": "Disponibilidade - mês anterior",
            },
            axis="columns",
        )

        df = df[(df["Day"] > "2020")]
        df
        df["Demurrage"] = abs(df["Demurrage"])
        df["Demurrage Unitária"] = df["Demurrage"] / df["Volume"]
        df = df.fillna(0)
        df1 = df.copy()

        df["Periodo"] = "Seco e Chuvoso"

        # Criar a função para determinar o período com base no mês e no porto
        def determinar_periodo(row):
            if row["Port"] == "Ponta Madeira":
                if 1 <= row["Day"].month <= 6:
                    return "Chuvoso"
                else:
                    return "Seco"
            else:
                if (
                    1 <= row["Day"].month <= 4
                    or row["Day"].month == 11
                    or row["Day"].month == 12
                ):
                    return "Chuvoso"
                else:
                    return "Seco"

        # Aplicar a função em cada linha do DataFrame para criar a coluna "Periodo"
        df1["Periodo"] = df1.apply(determinar_periodo, axis=1)
        df = pd.concat([df, df1])
        df = df[df["Pier"] == "Total"]

        ponta_madeira = df.loc[df["Port"] == "Ponta Madeira"]
        tubarao = df.loc[df["Port"] == "Tubarao"]
        guaiba = df.loc[df["Port"] == "Guaiba"]
        sepetiba = df.loc[df["Port"] == "Sepetiba"]
        GS = df.loc[df["Port"] == "Guaiba e Sepetiba"]

        for i in [ponta_madeira, tubarao, guaiba, sepetiba, GS]:
            i = i.drop(["Day", "Port", "Pier"], axis=1, inplace=True)
        ponta_madeira
        corr_pm = ponta_madeira.corr()
        corr_tb = tubarao.corr()
        corr_gb = guaiba.corr()
        corr_sx = sepetiba.corr()
        corr_gs = GS.corr()

        sepetiba = sepetiba.dropna(axis=0)

        corr_pm["Porto"] = "Ponta da Madeira"
        corr_tb["Porto"] = "Tubarão"
        corr_gb["Porto"] = "Guaíba"
        corr_sx["Porto"] = "Sepetiba"
        corr_gs["Porto"] = "Guaíba e Sepetiba"
        df = pd.concat([corr_pm, corr_tb, corr_gb, corr_sx, corr_gs], axis=0)
        df = df.reset_index()

        df_melted = pd.melt(
            df,
            id_vars=["index", "Porto"],
            value_vars=["Demurrage", "Volume", "Estadia", "Demurrage Unitária"],
            var_name="Variavel",
            value_name="Correlacao",
        )
        df_melted
        # df_melted.to_excel(f"C:/Users/KG858HY/OneDrive - EY/Desktop/Scripts/Output/correlacao_bi_{date.today()}.xlsx")
        base5 = df_melted

        ##########################################################################################
        """
        Cálculo Shap Values

        input: base semanal atualizada

        output: bases de shap values por porto e variável resposta

        """
        if not os.path.exists(self.path + "/shap + clusters " + str(date.today())):
            os.makedirs(self.path + "/shap + clusters " + str(date.today()))
        files = os.listdir(self.path + "/shap + clusters " + str(date.today()) + "/")
        for file in files:
            os.remove(self.path + "/shap + clusters " + str(date.today()) + "/" + file)

        for i in ["Ponta Madeira", "Guaiba e Sepetiba", "Tubarao"]:
            for j in ["Multa_Demurrage", "Volume_Embarcado"]:
                for l in ["Seco", "Chuvoso", "Seco e Chuvoso"]:
                    porto = i
                    pier = "Total"

                    DATA_INICIO_TREINO = "2019-07-01"

                    df_train = self.base_semanal

                    df_train = df_train[
                        (df_train["Day"] >= DATA_INICIO_TREINO)
                        & (df_train["Day"] <= self.DATA_INICIO_TESTE)
                    ]
                    df_train.info()
                    df_train = df_train.fillna(0)
                    df_train

                    dep_var = j

                    dep_var1 = j

                    seco_ou_chuvoso = l

                    df_train1 = df_train.copy()

                    df_train["Periodo"] = "Seco e Chuvoso"

                    df_train1["Periodo"] = np.where(
                        df_train1["Port"] == "Ponta Madeira",
                        np.where(
                            df_train1["Day"].dt.month.isin([1, 2, 3, 4, 5, 6]),
                            "Chuvoso",
                            "Seco",
                        ),
                        np.where(
                            (df_train1["Day"].dt.month >= 11)
                            | (df_train1["Day"].dt.month <= 4),
                            "Chuvoso",
                            "Seco",
                        ),
                    )

                    df_train = pd.concat([df_train, df_train1])

                    df_train = df_train[
                        (df_train["Port"] == porto) & (df_train["Pier"] == pier)
                    ]
                    df_train = df_train[(df_train["Periodo"] == seco_ou_chuvoso)]

                    df_train = df_train.drop(columns=["Day", "Port", "Pier", "Periodo"])
                    train_x = df_train[
                        df_train.columns.difference(
                            ["Multa_Demurrage", "Volume_Embarcado"]
                        )
                    ]

                    df_train.columns

                    if self.tuning_ == 1:

                        n_estimators = [
                            int(x)
                            for x in np.linspace(
                                # Numero de arvores random forest
                                start=200,
                                stop=2000,
                                num=10,
                            )
                        ]
                        # Numero de features para considerar em cada split
                        max_features = ["auto", "sqrt"]
                        # Numero maximo de folhas
                        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                        max_depth.append(None)
                        # Numero minimo de samples para fazer o split de um node
                        min_samples_split = [2, 5, 10]
                        # Numero minimo de samples requereida para fazer o split da folha
                        min_samples_leaf = [1, 2, 4]
                        # Metodo de selecao de amostras para treinar cada arvore
                        bootstrap = [True, False]
                        # Criando o random grid
                        random_grid = {
                            "n_estimators": n_estimators,
                            "max_features": max_features,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "bootstrap": bootstrap,
                        }

                        # Vamos utilizar o random grid para buscar pelos melhores hyperparameters

                        r_f = RandomForestRegressor()

                        # Random search of parameters, using 3 fold cross validation,
                        # search across 100 different combinations, and use all available cores

                        r_f_random = RandomizedSearchCV(
                            estimator=r_f,
                            param_distributions=random_grid,
                            n_iter=100,
                            cv=3,
                            verbose=2,
                            random_state=42,
                            n_jobs=-1,
                        )

                        # Fit the random search model

                        r_f_random.fit(
                            train_x, df_train[dep_var]
                        )  # Estorou erro (module '_winapi' has no attribute 'SYNCHRONIZE')

                        # From these results, we should be able to narrow the range of values
                        # for each hyperparameter.

                        # Create the parameter grid train_xd on the results of random
                        # search

                        if r_f_random.best_params_["max_depth"] is None:
                            param_grid = {
                                "bootstrap": [r_f_random.best_params_["bootstrap"]],
                                "max_depth": [r_f_random.best_params_["max_depth"]],
                                "max_features": [
                                    r_f_random.best_params_["max_features"]
                                ],
                                "min_samples_leaf": [1, 2, 4],
                                "min_samples_split": [2, 5, 10],
                                "n_estimators": [
                                    r_f_random.best_params_["n_estimators"]
                                ]
                                + [r_f_random.best_params_["n_estimators"] + 100]
                                + [r_f_random.best_params_["n_estimators"] + 200],
                            }
                        else:
                            param_grid = {
                                "bootstrap": [r_f_random.best_params_["bootstrap"]],
                                "max_depth": [r_f_random.best_params_["max_depth"]]
                                + [r_f_random.best_params_["max_depth"] + 10]
                                + [r_f_random.best_params_["max_depth"] + 20],
                                "max_features": [
                                    r_f_random.best_params_["max_features"]
                                ],
                                "min_samples_leaf": [1, 2, 4],
                                "min_samples_split": [2, 5, 10],
                                "n_estimators": [
                                    r_f_random.best_params_["n_estimators"]
                                ]
                                + [r_f_random.best_params_["n_estimators"] + 100]
                                + [r_f_random.best_params_["n_estimators"] + 200],
                            }

                        # Create a train_xd model

                        r_f = RandomForestRegressor()

                        # Instantiate the grid search model

                        grid_search = GridSearchCV(
                            estimator=r_f,
                            param_grid=param_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=2,
                        )

                        # Fit the grid search to the data

                        grid_search.fit(train_x, df_train[dep_var])

                        r_f = RandomForestRegressor(
                            n_estimators=grid_search.best_params_["n_estimators"],
                            min_samples_split=grid_search.best_params_[
                                "min_samples_split"
                            ],
                            min_samples_leaf=grid_search.best_params_[
                                "min_samples_leaf"
                            ],
                            max_features=grid_search.best_params_["max_features"],
                            max_depth=grid_search.best_params_["max_depth"],
                            bootstrap=grid_search.best_params_["bootstrap"],
                            random_state=42,
                        )

                        self.hp_rf_cluster[i][j][l][
                            "n_estimators"
                        ] = grid_search.best_params_["n_estimators"]
                        self.hp_rf_cluster[i][j][l][
                            "min_samples_split"
                        ] = grid_search.best_params_["min_samples_split"]
                        self.hp_rf_cluster[i][j][l][
                            "min_samples_leaf"
                        ] = grid_search.best_params_["min_samples_leaf"]
                        self.hp_rf_cluster[i][j][l][
                            "max_features"
                        ] = grid_search.best_params_["max_features"]
                        self.hp_rf_cluster[i][j][l][
                            "max_depth"
                        ] = grid_search.best_params_["max_depth"]
                        self.hp_rf_cluster[i][j][l][
                            "bootstrap"
                        ] = grid_search.best_params_["bootstrap"]
                        self.hp_rf_cluster[i][j][l]["n_estimators"] = 42

                    elif self.tuning_ == 0:

                        if (
                            i == "Ponta Madeira"
                            and j == "Multa_Demurrage"
                            and l == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )
                        elif (
                            i == "Ponta Madeira"
                            and j == "Multa_Demurrage"
                            and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Ponta Madeira"
                            and j == "Multa_Demurrage"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Ponta Madeira"
                            and j == "Volume_Embarcado"
                            and l == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Ponta Madeira"
                            and j == "Volume_Embarcado"
                            and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Ponta Madeira"
                            and j == "Volume_Embarcado"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Multa_Demurrage"
                            and l == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Multa_Demurrage"
                            and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Multa_Demurrage"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Volume_Embarcado"
                            and l == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Volume_Embarcado"
                            and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Guaiba e Sepetiba"
                            and j == "Volume_Embarcado"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif i == "Tubarao" and j == "Multa_Demurrage" and l == "Seco":
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Tubarao" and j == "Multa_Demurrage" and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Tubarao"
                            and j == "Multa_Demurrage"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif i == "Tubarao" and j == "Volume_Embarcado" and l == "Seco":
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Tubarao"
                            and j == "Volume_Embarcado"
                            and l == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            i == "Tubarao"
                            and j == "Volume_Embarcado"
                            and l == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[i][j][l][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[i][j][l][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[i][j][l][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[i][j][l]["max_depth"],
                                bootstrap=self.hp_rf_cluster[i][j][l]["bootstrap"],
                                random_state=self.hp_rf_cluster[i][j][l][
                                    "n_estimators"
                                ],
                            )

                    r_f.fit(train_x, df_train[dep_var])
                    mdl = r_f
                    shap_values, expected_values = shap_values_fun(mdl, train_x)

                    df_shap_values = pd.DataFrame(shap_values)

                    renames = {}

                    # for columns in range(len(train_x.columns)):
                    #      renames.append({columns : train_x.columns[columns]})

                    for column in range(len(train_x.columns)):
                        renames.update({column: train_x.columns[column] + " SHAP"})

                    df_shap_values = df_shap_values.rename(columns=renames)

                    variaveis_shape = [
                        "Estadia_media_navios_hs SHAP",
                        "Lag 1 mes numero de navios na fila SHAP",
                        "Lag 2 meses numero de navios na fila SHAP",
                        "Lag 3 meses numero de navios na fila SHAP",
                        "Lag 1 mes Capacidade SHAP",
                        "Lag 2 meses Capacidade SHAP",
                        "Lag 3 meses Capacidade SHAP",
                        "Lag 1 OEE SHAP",
                        "Lag 1 DISPONIBILIDADE SHAP",
                        "Navios FOB SHAP",
                        "Navios CFR SHAP",
                        "Pcg-FOB SHAP",
                        "Pcg-CFR SHAP",
                        "Navios PANAMAX SHAP",
                        "Pcg-PANAMAX SHAP",
                        "Navios CAPE SHAP",
                        "Pcg-CAPE SHAP",
                        "Navios VLOC SHAP",
                        "Pcg-VLOC SHAP",
                        "Navios NEWCASTLE SHAP",
                        "Pcg-NEWCASTLE SHAP",
                        "Navios VALEMAX SHAP",
                        "Pcg-VALEMAX SHAP",
                        "Navios BABYCAPE SHAP",
                        "Pcg-BABYCAPE SHAP",
                        "Navios SPOT/FOB SHAP",
                        "Pcg-SPOT/FOB SHAP",
                        "Navios Frota Dedicada/SPOT/FOB SHAP",
                        "Pcg-Frota Dedicada/SPOT/FOB SHAP",
                        "Navios Frota Dedicada/FOB SHAP",
                        "Pcg-Frota Dedicada/FOB SHAP",
                        "Navios Frota Dedicada SHAP",
                        "Pcg-Frota Dedicada SHAP",
                        "Quantity_t SHAP",
                        "Dwt_K_total SHAP",
                        "Dwt_K_medio SHAP",
                        "Qtde/Dwt SHAP",
                        "DISPONIBILIDADE SHAP",
                        "UTILIZACAO SHAP",
                        "OEE SHAP",
                        "CAPACIDADE SHAP",
                        "CAPACIDADE/Dwt SHAP",
                        "TAXA_EFETIVA SHAP",
                    ]

                    df_shap_values = df_shap_values[variaveis_shape]

                    variaveis_train = [
                        "Estadia_media_navios_hs",
                        "Lag 1 mes numero de navios na fila",
                        "Lag 2 meses numero de navios na fila",
                        "Lag 3 meses numero de navios na fila",
                        "Lag 1 mes Capacidade",
                        "Lag 2 meses Capacidade",
                        "Lag 3 meses Capacidade",
                        "Lag 1 OEE",
                        "Lag 1 DISPONIBILIDADE",
                        "Navios FOB",
                        "Navios CFR",
                        "Pcg-FOB",
                        "Pcg-CFR",
                        "Navios PANAMAX",
                        "Pcg-PANAMAX",
                        "Navios CAPE",
                        "Pcg-CAPE",
                        "Navios VLOC",
                        "Pcg-VLOC",
                        "Navios NEWCASTLE",
                        "Pcg-NEWCASTLE",
                        "Navios VALEMAX",
                        "Pcg-VALEMAX",
                        "Navios BABYCAPE",
                        "Pcg-BABYCAPE",
                        "Navios SPOT/FOB",
                        "Pcg-SPOT/FOB",
                        "Navios Frota Dedicada/SPOT/FOB",
                        "Pcg-Frota Dedicada/SPOT/FOB",
                        "Navios Frota Dedicada/FOB",
                        "Pcg-Frota Dedicada/FOB",
                        "Navios Frota Dedicada",
                        "Pcg-Frota Dedicada",
                        "Quantity_t",
                        "Dwt_K_total",
                        "Dwt_K_medio",
                        "Qtde/Dwt",
                        "DISPONIBILIDADE",
                        "UTILIZACAO",
                        "OEE",
                        "CAPACIDADE",
                        "CAPACIDADE/Dwt",
                        "TAXA_EFETIVA",
                    ]

                    df_train = df_train[variaveis_train]

                    X = pd.concat(
                        [train_x.reset_index(), df_shap_values], axis=1
                    ).reset_index()
                    X.to_excel(
                        self.path
                        + "/shap + clusters "
                        + str(date.today())
                        + "/base_shap_"
                        + dep_var1
                        + "_"
                        + porto
                        + "_"
                        + seco_ou_chuvoso
                        + ".xlsx",
                        index=False,
                    )

        for a in ["Ponta Madeira", "Guaiba e Sepetiba", "Tubarao"]:
            for b in ["Estadia_media_navios_hs"]:
                for c in ["Seco", "Chuvoso", "Seco e Chuvoso"]:
                    porto = a
                    pier = "Total"

                    DATA_INICIO_TREINO = "2019-07-01"

                    df_train = self.base_semanal

                    df_train = df_train[
                        (df_train["Day"] >= DATA_INICIO_TREINO)
                        & (df_train["Day"] <= self.DATA_INICIO_TESTE)
                    ]
                    df_train.info()
                    df_train = df_train.fillna(0)

                    dep_var = b

                    dep_var1 = b

                    seco_ou_chuvoso = c

                    df_train1 = df_train.copy()

                    df_train["Periodo"] = "Seco e Chuvoso"

                    df_train1["Periodo"] = np.where(
                        df_train1["Port"] == "Ponta Madeira",
                        np.where(
                            df_train1["Day"].dt.month.isin([1, 2, 3, 4, 5, 6]),
                            "Chuvoso",
                            "Seco",
                        ),
                        np.where(
                            (df_train1["Day"].dt.month >= 11)
                            | (df_train1["Day"].dt.month <= 4),
                            "Chuvoso",
                            "Seco",
                        ),
                    )

                    df_train = pd.concat([df_train, df_train1])

                    df_train = df_train[
                        (df_train["Port"] == porto) & (df_train["Pier"] == pier)
                    ]
                    df_train = df_train[(df_train["Periodo"] == seco_ou_chuvoso)]

                    df_train = df_train.drop(columns=["Day", "Port", "Pier", "Periodo"])
                    train_x = df_train[
                        df_train.columns.difference(
                            [
                                "Multa_Demurrage",
                                "Volume_Embarcado",
                                "Estadia_media_navios_hs",
                            ]
                        )
                    ]

                    if self.tuning_ == 1:

                        n_estimators = [
                            int(x)
                            for x in np.linspace(
                                # Numero de arvores random forest
                                start=200,
                                stop=2000,
                                num=10,
                            )
                        ]
                        # Numero de features para considerar em cada split
                        max_features = ["auto", "sqrt"]
                        # Numero maximo de folhas
                        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                        max_depth.append(None)
                        # Numero minimo de samples para fazer o split de um node
                        min_samples_split = [2, 5, 10]
                        # Numero minimo de samples requereida para fazer o split da folha
                        min_samples_leaf = [1, 2, 4]
                        # Metodo de selecao de amostras para treinar cada arvore
                        bootstrap = [True, False]
                        # Criando o random grid
                        random_grid = {
                            "n_estimators": n_estimators,
                            "max_features": max_features,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "bootstrap": bootstrap,
                        }

                        # Vamos utilizar o random grid para buscar pelos melhores hyperparameters

                        r_f = RandomForestRegressor()

                        # Random search of parameters, using 3 fold cross validation,
                        # search across 100 different combinations, and use all available cores

                        r_f_random = RandomizedSearchCV(
                            estimator=r_f,
                            param_distributions=random_grid,
                            n_iter=100,
                            cv=3,
                            verbose=2,
                            random_state=42,
                            n_jobs=-1,
                        )

                        # Fit the random search model

                        r_f_random.fit(
                            train_x, df_train[dep_var]
                        )  # Estorou erro (module '_winapi' has no attribute 'SYNCHRONIZE')

                        # From these results, we should be able to narrow the range of values
                        # for each hyperparameter.

                        # Create the parameter grid train_xd on the results of random
                        # search

                        if r_f_random.best_params_["max_depth"] is None:
                            param_grid = {
                                "bootstrap": [r_f_random.best_params_["bootstrap"]],
                                "max_depth": [r_f_random.best_params_["max_depth"]],
                                "max_features": [
                                    r_f_random.best_params_["max_features"]
                                ],
                                "min_samples_leaf": [1, 2, 4],
                                "min_samples_split": [2, 5, 10],
                                "n_estimators": [
                                    r_f_random.best_params_["n_estimators"]
                                ]
                                + [r_f_random.best_params_["n_estimators"] + 100]
                                + [r_f_random.best_params_["n_estimators"] + 200],
                            }
                        else:
                            param_grid = {
                                "bootstrap": [r_f_random.best_params_["bootstrap"]],
                                "max_depth": [r_f_random.best_params_["max_depth"]]
                                + [r_f_random.best_params_["max_depth"] + 10]
                                + [r_f_random.best_params_["max_depth"] + 20],
                                "max_features": [
                                    r_f_random.best_params_["max_features"]
                                ],
                                "min_samples_leaf": [1, 2, 4],
                                "min_samples_split": [2, 5, 10],
                                "n_estimators": [
                                    r_f_random.best_params_["n_estimators"]
                                ]
                                + [r_f_random.best_params_["n_estimators"] + 100]
                                + [r_f_random.best_params_["n_estimators"] + 200],
                            }

                        # Create a train_xd model

                        r_f = RandomForestRegressor()

                        # Instantiate the grid search model

                        grid_search = GridSearchCV(
                            estimator=r_f,
                            param_grid=param_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=2,
                        )

                        # Fit the grid search to the data

                        grid_search.fit(train_x, df_train[dep_var])

                        r_f = RandomForestRegressor(
                            n_estimators=grid_search.best_params_["n_estimators"],
                            min_samples_split=grid_search.best_params_[
                                "min_samples_split"
                            ],
                            min_samples_leaf=grid_search.best_params_[
                                "min_samples_leaf"
                            ],
                            max_features=grid_search.best_params_["max_features"],
                            max_depth=grid_search.best_params_["max_depth"],
                            bootstrap=grid_search.best_params_["bootstrap"],
                            random_state=42,
                        )

                        self.hp_rf_cluster[a][b][c][
                            "n_estimators"
                        ] = grid_search.best_params_["n_estimators"]
                        self.hp_rf_cluster[a][b][c][
                            "min_samples_split"
                        ] = grid_search.best_params_["min_samples_split"]
                        self.hp_rf_cluster[a][b][c][
                            "min_samples_leaf"
                        ] = grid_search.best_params_["min_samples_leaf"]
                        self.hp_rf_cluster[a][b][c][
                            "max_features"
                        ] = grid_search.best_params_["max_features"]
                        self.hp_rf_cluster[a][b][c][
                            "max_depth"
                        ] = grid_search.best_params_["max_depth"]
                        self.hp_rf_cluster[a][b][c][
                            "bootstrap"
                        ] = grid_search.best_params_["bootstrap"]
                        self.hp_rf_cluster[a][b][c]["n_estimators"] = 42

                    elif self.tuning_ == 0:

                        if (
                            a == "Ponta Madeira"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Ponta Madeira"
                            and b == "Estadia_media_navios_hs"
                            and c == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Ponta Madeira"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Guaiba e Sepetiba"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Guaiba e Sepetiba"
                            and b == "Estadia_media_navios_hs"
                            and c == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Guaiba e Sepetiba"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Tubarao"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Tubarao"
                            and b == "Estadia_media_navios_hs"
                            and c == "Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                        elif (
                            a == "Tubarao"
                            and b == "Estadia_media_navios_hs"
                            and c == "Seco e Chuvoso"
                        ):
                            r_f = RandomForestRegressor(
                                n_estimators=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                                min_samples_split=self.hp_rf_cluster[a][b][c][
                                    "min_samples_split"
                                ],
                                min_samples_leaf=self.hp_rf_cluster[a][b][c][
                                    "min_samples_leaf"
                                ],
                                max_features=self.hp_rf_cluster[a][b][c][
                                    "max_features"
                                ],
                                max_depth=self.hp_rf_cluster[a][b][c]["max_depth"],
                                bootstrap=self.hp_rf_cluster[a][b][c]["bootstrap"],
                                random_state=self.hp_rf_cluster[a][b][c][
                                    "n_estimators"
                                ],
                            )

                    r_f.fit(train_x, df_train[dep_var])
                    mdl = r_f
                    shap_values, expected_values = shap_values_fun(mdl, train_x)

                    df_shap_values = pd.DataFrame(shap_values)

                    renames = {}

                    for column in range(len(train_x.columns)):
                        renames.update({column: train_x.columns[column] + " SHAP"})

                    df_shap_values = df_shap_values.rename(columns=renames)

                    variaveis_shape = [
                        "Lag 1 mes numero de navios na fila SHAP",
                        "Lag 2 meses numero de navios na fila SHAP",
                        "Lag 3 meses numero de navios na fila SHAP",
                        "Lag 1 mes Capacidade SHAP",
                        "Lag 2 meses Capacidade SHAP",
                        "Lag 3 meses Capacidade SHAP",
                        "Lag 1 OEE SHAP",
                        "Lag 1 DISPONIBILIDADE SHAP",
                        "Navios FOB SHAP",
                        "Navios CFR SHAP",
                        "Pcg-FOB SHAP",
                        "Pcg-CFR SHAP",
                        "Navios PANAMAX SHAP",
                        "Pcg-PANAMAX SHAP",
                        "Navios CAPE SHAP",
                        "Pcg-CAPE SHAP",
                        "Navios VLOC SHAP",
                        "Pcg-VLOC SHAP",
                        "Navios NEWCASTLE SHAP",
                        "Pcg-NEWCASTLE SHAP",
                        "Navios VALEMAX SHAP",
                        "Pcg-VALEMAX SHAP",
                        "Navios BABYCAPE SHAP",
                        "Pcg-BABYCAPE SHAP",
                        "Navios SPOT/FOB SHAP",
                        "Pcg-SPOT/FOB SHAP",
                        "Navios Frota Dedicada/SPOT/FOB SHAP",
                        "Pcg-Frota Dedicada/SPOT/FOB SHAP",
                        "Navios Frota Dedicada/FOB SHAP",
                        "Pcg-Frota Dedicada/FOB SHAP",
                        "Navios Frota Dedicada SHAP",
                        "Pcg-Frota Dedicada SHAP",
                        "Quantity_t SHAP",
                        "Dwt_K_total SHAP",
                        "Dwt_K_medio SHAP",
                        "Qtde/Dwt SHAP",
                        "DISPONIBILIDADE SHAP",
                        "UTILIZACAO SHAP",
                        "OEE SHAP",
                        "CAPACIDADE SHAP",
                        "CAPACIDADE/Dwt SHAP",
                        "TAXA_EFETIVA SHAP",
                    ]

                    df_shap_values = df_shap_values[variaveis_shape]

                    variaveis_train = [
                        "Lag 1 mes numero de navios na fila",
                        "Lag 2 meses numero de navios na fila",
                        "Lag 3 meses numero de navios na fila",
                        "Lag 1 mes Capacidade",
                        "Lag 2 meses Capacidade",
                        "Lag 3 meses Capacidade",
                        "Lag 1 OEE",
                        "Lag 1 DISPONIBILIDADE",
                        "Navios FOB",
                        "Navios CFR",
                        "Pcg-FOB",
                        "Pcg-CFR",
                        "Navios PANAMAX",
                        "Pcg-PANAMAX",
                        "Navios CAPE",
                        "Pcg-CAPE",
                        "Navios VLOC",
                        "Pcg-VLOC",
                        "Navios NEWCASTLE",
                        "Pcg-NEWCASTLE",
                        "Navios VALEMAX",
                        "Pcg-VALEMAX",
                        "Navios BABYCAPE",
                        "Pcg-BABYCAPE",
                        "Navios SPOT/FOB",
                        "Pcg-SPOT/FOB",
                        "Navios Frota Dedicada/SPOT/FOB",
                        "Pcg-Frota Dedicada/SPOT/FOB",
                        "Navios Frota Dedicada/FOB",
                        "Pcg-Frota Dedicada/FOB",
                        "Navios Frota Dedicada",
                        "Pcg-Frota Dedicada",
                        "Quantity_t",
                        "Dwt_K_total",
                        "Dwt_K_medio",
                        "Qtde/Dwt",
                        "DISPONIBILIDADE",
                        "UTILIZACAO",
                        "OEE",
                        "CAPACIDADE",
                        "CAPACIDADE/Dwt",
                        "TAXA_EFETIVA",
                    ]

                    df_train = df_train[variaveis_train]

                    Y = pd.concat(
                        [train_x.reset_index(), df_shap_values], axis=1
                    ).reset_index()
                    Y.to_excel(
                        self.path
                        + "/shap + clusters "
                        + str(date.today())
                        + "/base_shap_"
                        + dep_var1
                        + "_"
                        + porto
                        + "_"
                        + seco_ou_chuvoso
                        + ".xlsx",
                        index=False,
                    )

        """
        Shap Values

        """

        padrao_arquivo = self.path + "/shap + clusters " + str(date.today()) + "/*.xlsx"

        dataframes = []

        for arquivo in glob.glob(padrao_arquivo):
            # Extrai as informações do nome do arquivo
            nome_arquivo = arquivo.split("/")[-1]
            nome_variavel = nome_arquivo.split("_")[2]
            nome_porto = nome_arquivo.split("_")[-2]
            nome_periodo = nome_arquivo.split("_")[-1].split(".")[0]

            # Importa o arquivo XLSX como dataframe usando o pandas
            df = pd.read_excel(arquivo)
            df["Variavel"] = nome_variavel
            df["Porto"] = nome_porto
            df["Periodo"] = nome_periodo

            # Adiciona o dataframe à lista
            dataframes.append(df)

        df_final = pd.concat(dataframes, axis=0, ignore_index=True)

        df_final = df_final.rename(
            {
                "Lag 1 mes numero de navios na fila": "Navios em fila - mês anterior",
                "Lag 2 meses numero de navios na fila": "Navios em fila - 2 meses anteriores",
                "Lag 3 meses numero de navios na fila": "Navios em fila - 3 meses anteriores",
                "Lag 1 mes Capacidade": "Capacidade - mês anterior",
                "Lag 2 meses Capacidade": "Capacidade - 2 meses anteriores",
                "Lag 3 meses Capacidade": "Capacidade - 3 meses anteriores",
                "Lag 1 OEE": "OEE - mês anterior",
                "Lag 1 DISPONIBILIDADE": "Disponibilidade - mês anterior",
            },
            axis="columns",
        )
        df_final = df_final.rename(
            {
                "Lag 1 mes numero de navios na fila SHAP": "Navios em fila - mês anterior SHAP",
                "Lag 2 meses numero de navios na fila SHAP": "Navios em fila - 2 meses anteriores SHAP",
                "Lag 3 meses numero de navios na fila SHAP": "Navios em fila - 3 meses anteriores SHAP",
                "Lag 1 mes Capacidade SHAP": "Capacidade - mês anterior SHAP",
                "Lag 2 meses Capacidade SHAP": "Capacidade - 2 meses anteriores SHAP",
                "Lag 3 meses Capacidade SHAP": "Capacidade - 3 meses anteriores SHAP",
                "Lag 1 OEE SHAP": "OEE - mês anterior SHAP",
                "Lag 1 DISPONIBILIDADE SHAP": "Disponibilidade - mês anterior SHAP",
            },
            axis="columns",
        )
        df_final = df_final.replace(
            {
                "Ponta Madeira": "Ponta da Madeira",
                "Tubarao": "Tubarão",
                "Guaiba": "Guaíba",
                "Guaiba e Sepetiba": "Guaíba e Sepetiba",
            }
        )

        # df_final.to_excel(f+"/shap_values_{date.today()}.xlsx")
        base8 = df_final

        """
        Clusters

        """

        padrao_arquivo = self.path + "/shap + clusters " + str(date.today()) + "/*.xlsx"

        dataframes1 = []

        for arquivo in glob.glob(padrao_arquivo):
            # Extrai as informações do nome do arquivo
            nome_arquivo = arquivo.split("/")[-1]
            nome_variavel = nome_arquivo.split("_")[2]
            nome_porto = nome_arquivo.split("_")[-2]
            nome_periodo = nome_arquivo.split("_")[-1].split(".")[0]

            # Importa o arquivo XLSX como dataframe usando o pandas
            df = pd.read_excel(arquivo)
            df["Variavel"] = nome_variavel
            df["Porto"] = nome_porto
            df["Periodo"] = nome_periodo

            # Adiciona o dataframe à lista
            dataframes1.append(df)

        df_final_clusters = pd.concat(dataframes1, axis=0, ignore_index=True)

        df_final_clusters = df_final_clusters.replace({"Multa": "Demurrage"})

        df_final_clusters["DISPONIBILIDADE SHAP"] = np.where(
            (df_final_clusters["Variavel"] == "Demurrage")
            & (df_final_clusters["Porto"] == "Guaiba e Sepetiba"),
            df_final_clusters["DISPONIBILIDADE SHAP"] * -1,
            df_final_clusters["DISPONIBILIDADE SHAP"] * 1,
        )

        # df_final_clusters['Dwt_K'] = df_final_clusters['Dwt_K']*4

        df_final_clusters["Estadia_media_navios_hs"] = (
            df_final_clusters["Estadia_media_navios_hs"] / 24
        )

        df = df_final_clusters.copy()

        list_shap = [
            "Estadia_media_navios_hs SHAP",
            "Lag 1 mes numero de navios na fila SHAP",
            "Lag 2 meses numero de navios na fila SHAP",
            "Lag 3 meses numero de navios na fila SHAP",
            "Lag 1 mes Capacidade SHAP",
            "Lag 2 meses Capacidade SHAP",
            "Lag 3 meses Capacidade SHAP",
            "Lag 1 OEE SHAP",
            "Lag 1 DISPONIBILIDADE SHAP",
            "Navios FOB SHAP",
            "Navios CFR SHAP",
            "Pcg-FOB SHAP",
            "Pcg-CFR SHAP",
            "Navios PANAMAX SHAP",
            "Pcg-PANAMAX SHAP",
            "Navios CAPE SHAP",
            "Pcg-CAPE SHAP",
            "Navios VLOC SHAP",
            "Pcg-VLOC SHAP",
            "Navios NEWCASTLE SHAP",
            "Pcg-NEWCASTLE SHAP",
            "Navios VALEMAX SHAP",
            "Pcg-VALEMAX SHAP",
            "Navios BABYCAPE SHAP",
            "Pcg-BABYCAPE SHAP",
            "Navios SPOT/FOB SHAP",
            "Pcg-SPOT/FOB SHAP",
            "Navios Frota Dedicada/SPOT/FOB SHAP",
            "Pcg-Frota Dedicada/SPOT/FOB SHAP",
            "Navios Frota Dedicada/FOB SHAP",
            "Pcg-Frota Dedicada/FOB SHAP",
            "Navios Frota Dedicada SHAP",
            "Pcg-Frota Dedicada SHAP",
            "Quantity_t SHAP",
            "Dwt_K_total SHAP",
            "Dwt_K_medio SHAP",
            "Qtde/Dwt SHAP",
            "DISPONIBILIDADE SHAP",
            "UTILIZACAO SHAP",
            "OEE SHAP",
            "CAPACIDADE SHAP",
            "CAPACIDADE/Dwt SHAP",
            "TAXA_EFETIVA SHAP",
        ]

        df = df.drop(columns=["level_0", "index"])

        df = df.reset_index()

        df_negativo = {}

        # Itere sobre as colunas de interesse
        for i in list_shap:
            i2 = i.replace(" SHAP", "")
            df1 = df.copy()
            df_negativo[i] = df1[df1[i] < 0][[i2, "Variavel", "Porto", "Periodo"]]

        df_positivo = {}

        # Itere sobre as colunas de interesse
        for i in list_shap:
            i2 = i.replace(" SHAP", "")
            df1 = df.copy()
            df_positivo[i] = df1[df1[i] > 0][[i2, "Variavel", "Porto", "Periodo"]]

        dataframes_agrupados = []

        # Itere sobre os dataframes no dicionário
        for chave, df in df_negativo.items():
            # Realize o groupby e obtenha o resultado
            resultado_groupby = df.groupby(
                ["Variavel", "Porto", "Periodo"], as_index=False
            ).median()

            # Adicione o dataframe resultante à lista
            dataframes_agrupados.append(resultado_groupby)

        # Concatene os dataframes da lista em um único dataframe
        df_concatenado_neg = pd.concat(dataframes_agrupados)

        dataframes_agrupados2 = []

        # Itere sobre os dataframes no dicionário
        for chave, df in df_positivo.items():
            # Realize o groupby e obtenha o resultado
            resultado_groupby2 = df.groupby(
                ["Variavel", "Porto", "Periodo"], as_index=False
            ).median()

            # Adicione o dataframe resultante à lista
            dataframes_agrupados2.append(resultado_groupby2)

        # Concatene os dataframes da lista em um único dataframe
        df_concatenado_posi = pd.concat(dataframes_agrupados2)

        list_x = [
            "Estadia_media_navios_hs",
            "CAPACIDADE",
            "CAPACIDADE/Dwt",
            "DISPONIBILIDADE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Lag 1 DISPONIBILIDADE",
            "Lag 1 OEE",
            "Lag 1 mes Capacidade",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses Capacidade",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses Capacidade",
            "Lag 3 meses numero de navios na fila",
            "OEE",
            "Qtde/Dwt",
            "Quantity_t",
            "TAXA_EFETIVA",
            "UTILIZACAO",
        ]

        list_x1 = [
            "Navios BABYCAPE",
            "Navios CAPE",
            "Navios CFR",
            "Navios FOB",
            "Navios Frota Dedicada",
            "Navios Frota Dedicada/FOB",
            "Navios Frota Dedicada/SPOT/FOB",
            "Navios NEWCASTLE",
            "Navios PANAMAX",
            "Navios SPOT/FOB",
            "Navios VALEMAX",
            "Navios VLOC",
            "Pcg-BABYCAPE",
            "Pcg-CAPE",
            "Pcg-CFR",
            "Pcg-FOB",
            "Pcg-Frota Dedicada",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-NEWCASTLE",
            "Pcg-PANAMAX",
            "Pcg-SPOT/FOB",
            "Pcg-VALEMAX",
            "Pcg-VLOC",
        ]

        df_x_posi = pd.melt(
            df_concatenado_posi,
            id_vars=["Porto", "Variavel", "Periodo"],
            value_vars=list_x,
            var_name="Análise",
            value_name="X",
        )

        df_x_neg = pd.melt(
            df_concatenado_neg,
            id_vars=["Porto", "Variavel", "Periodo"],
            value_vars=list_x,
            var_name="Análise",
            value_name="X",
        )

        df_x_posi["Impacto"] = np.where(
            df_x_posi["Variavel"] == "Volume", "Positivo", "Negativo"
        )

        df_x_neg["Impacto"] = np.where(
            df_x_neg["Variavel"] == "Volume", "Negativo", "Positivo"
        )

        df_x_posi_disp = df_x_posi[df_x_posi["Análise"] == "DISPONIBILIDADE"][
            "X"
        ].copy()

        df_x_neg_disp = df_x_neg[df_x_posi["Análise"] == "DISPONIBILIDADE"]["X"].copy()

        df_x_posi["X"] = np.where(
            df_x_posi["Variavel"] == "Volume", df_x_posi["X"] * 1, df_x_posi["X"] * 0.6
        )

        df_x_neg["X"] = np.where(
            df_x_neg["Variavel"] == "Volume", df_x_neg["X"] * 0.6, df_x_neg["X"] * 1
        )

        # alteração no cluster de DISPONIBILIDADE para considerar 85% do cluster inferior

        df_x_neg.loc[
            (df_x_neg["Análise"] == "DISPONIBILIDADE")
            & (df_x_neg["Variavel"] == "Volume"),
            "X",
        ] = (
            df_x_neg_disp * 0.85
        )

        df_x_posi.loc[
            (df_x_posi["Análise"] == "DISPONIBILIDADE")
            & ~(df_x_neg["Variavel"] == "Volume"),
            "X",
        ] = (
            df_x_posi_disp * 0.85
        )

        df_x_posi1 = pd.melt(
            df_concatenado_posi,
            id_vars=["Porto", "Variavel", "Periodo"],
            value_vars=list_x1,
            var_name="Análise",
            value_name="X",
        )

        df_x_neg1 = pd.melt(
            df_concatenado_neg,
            id_vars=["Porto", "Variavel", "Periodo"],
            value_vars=list_x1,
            var_name="Análise",
            value_name="X",
        )

        df_x_posi1["Impacto"] = np.where(
            df_x_posi1["Variavel"] == "Volume", "Positivo", "Negativo"
        )

        df_x_neg1["Impacto"] = np.where(
            df_x_neg1["Variavel"] == "Volume", "Negativo", "Positivo"
        )

        df_x_posi1["X"] = np.where(
            df_x_posi1["Variavel"] == "Volume",
            df_x_posi1["X"] + 0.3,
            df_x_posi1["X"] + 0.02,
        )

        df_x_neg1["X"] = np.where(
            df_x_neg1["Variavel"] == "Volume",
            df_x_neg1["X"] + 0.02,
            df_x_neg1["X"] + 0.3,
        )

        df_final_ok = pd.concat([df_x_posi, df_x_neg], axis=0)

        df_final_ok1 = pd.concat([df_x_posi1, df_x_neg1], axis=0)

        df_final_ok = pd.concat([df_final_ok, df_final_ok1], axis=0)

        df_final_ok = df_final_ok.dropna()

        df_final_ok = df_final_ok.replace(
            {
                "Lag 1 mes numero de navios na fila": "Navios em fila - mês anterior",
                "Lag 2 meses numero de navios na fila": "Navios em fila - 2 meses anteriores",
                "Lag 3 meses numero de navios na fila": "Navios em fila - 3 meses anteriores",
                "Lag 1 mes Capacidade": "Capacidade - mês anterior",
                "Lag 2 meses Capacidade": "Capacidade - 2 meses anteriores",
                "Lag 3 meses Capacidade": "Capacidade - 3 meses anteriores",
                "Lag 1 OEE": "OEE - mês anterior",
                "Lag 1 DISPONIBILIDADE": "Disponibilidade - mês anterior",
            }
        )
        df_final_ok = df_final_ok.replace(
            {
                "Ponta Madeira": "Ponta da Madeira",
                "Tubarao": "Tubarão",
                "Guaiba": "Guaíba",
                "Guaiba e Sepetiba": "Guaíba e Sepetiba",
            }
        )

        # df_final_ok.to_excel(f+"/cluster_{date.today()}.xlsx")

        base9 = df_final_ok

        if not os.path.exists(self.path + "/shap + clusters"):
            os.makedirs(self.path + "/shap + clusters")

        files = os.listdir(self.path + "/shap + clusters" + "/")
        for file in files:
            os.remove(self.path + "/shap + clusters" + "/" + file)

        files = os.listdir(self.path + "/shap + clusters " + str(date.today()))

        for fname in files:
            shutil.copy2(
                os.path.join(
                    self.path + "/shap + clusters " + str(date.today()), fname
                ),
                self.path + "/shap + clusters",
            )

        if not os.path.exists(self.path + "/Output"):
            os.makedirs(self.path + "/Output")

        files = os.listdir(self.path + "/Output" + "/")
        for file in files:
            os.remove(self.path + "/Output" + "/" + file)

        if not os.path.exists(self.path + "/Output " + str(date.today())):
            os.makedirs(self.path + "/Output " + str(date.today()))

        return [
            base1,
            base2,
            base3,
            base4,
            base5,
            base8,
            base9,
            self.hp_rf_cluster,
            self.historico,
        ]
