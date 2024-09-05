"""
Classes utilizadas para as estimacoes no SHAP
"""

import random
import pickle
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, ParameterGrid

def tuning_prophet(
    data_base: pd.DataFrame,
    variable: str,
    data_inicio_teste: str,
    data_fim_teste: str,
):
    """Nessa funcao fazemos o tunning do modelo prophet. Vamos testar
    valores dentre intervalos distintos para cada hiperparametro e
    selecionar a que nos der o melhor valor para o mape.
    """
    # Estabelecemos essa data como data inicial para o treino do modelo
    # pois os dados anteriores a essa data se mostram viesados.
    data_inicio_treino = "2019-07-01"
    prophet_data = data_base[[variable]]
    prophet_data["Day"] = prophet_data.index
    prophet_data.rename(columns={variable: "y", "Day": "ds"}, inplace=True)
    prophet_data = prophet_data[["y", "ds"]]
    # Abaixo, selecionamos os intervalos de valores a serem testados
    # para cada hiperparametro.

    params_grid = {
        "seasonality_mode": ("multiplicative", "additive"),
        "changepoint_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
        "holidays_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
        "n_changepoints": [100, 150, 200],
    }

    grid = ParameterGrid(params_grid)
    cnt = 0
    for p_value in grid:
        cnt = cnt + 1
        # Abaixo, estimamos um modelos para cada uma das diferentes
        # combinacoes de valores para os hiperparametros e salvamos.
        # Como resultado da funcao, vamos retornar apenas os valores
        # de hiperparametros que possibilitaram o menor mape.

        model_parameters = pd.DataFrame(columns=["MAPE", "Parameters"])
        for p_value in grid:
            print(p_value)
            random.seed(0)
            train_model = Prophet(
                changepoint_prior_scale=p_value["changepoint_prior_scale"],
                holidays_prior_scale=p_value["holidays_prior_scale"],
                n_changepoints=p_value["n_changepoints"],
                seasonality_mode=p_value["seasonality_mode"],
                weekly_seasonality=True,
                daily_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95,
            )
            train_model.fit(
                prophet_data[
                    (prophet_data.index >= data_inicio_treino)
                    & (prophet_data.index < data_inicio_teste)
                ]
            )
            train_forecast = pd.DataFrame(
                data_base.index[
                    (data_base.index >= data_inicio_teste)
                    & (data_base.index <= data_fim_teste)
                ]
            )
            train_forecast.rename(
                columns={train_forecast.columns[0]: "ds"}, inplace=True
            )
            forecast_d = train_model.predict(train_forecast)
            pred_d = forecast_d[["ds", "yhat"]]
            mape = mean_absolute_percentage_error(
                data_base[
                    (data_base.index >= data_inicio_teste)
                    & (data_base.index <= data_fim_teste)
                ][[variable]],
                pred_d[
                    (pred_d["ds"] >= data_inicio_teste)
                    & (pred_d["ds"] <= data_fim_teste)
                ][["yhat"]],
            )
            print("Mean Absolute Percentage Error(MAPE)-------------", mape)
            model_parameters = model_parameters.append(
                {"MAPE": mape, "Parameters": p_value}, ignore_index=True
            )

            # Associados todos os valores de mape a combinacao de
            # hiperparametros que o gerou, vamos selecionar apenas os
            # hiperparametros associados ao menor valor de mape.

        parameters = model_parameters.sort_values(by=["MAPE"])
        parameters = parameters.reset_index(drop=True)
        parameters.head()

        return parameters["Parameters"][0]


class TratamentoDados:
    """A classe TratamentoDados prepara a base de dados que estamos utilizando
    para que possa ser insumo para o modelo Random Forest. Basicamente, os
    passos executados por essa classe sao: definicao de variáveis dependentes
    e independentes utilizadas nos tes modelos considerados; filtro dos dados
    que estao entre as datas de inicio do treinamento dos dados e de fim do
    período de teste; criacao de variáveis de controle.

    Args:

        df: Data frame completo, com todas as variáveis disponíveis para
        utilizacao no modelo.

        PORTO: Porto que iremos considerar no modelo.

        DATA_INICIO_TESTE: Data a partir da qual inicia nossa base de teste
        dos modelos.

        DATA_FIM_TESTE: Data que delimita o fim da base de teste dos
        modelos.

    Returns:

        df: Dataframe após tratamento.

        var_x: Lista com o nome das variáveis independentes consideradas no
        modelo previsto.

        var_controle: Lista com o nome das variáveis de controle
        consideradas  modelo previsto.
    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        porto: str,
        data_inicio_teste: str,
        data_fim_teste: str,
    ):
        self.df_data = df_data
        self.porto = porto
        self.data_inicio_teste = data_inicio_teste
        self.data_fim_teste = data_fim_teste

    def resultado(self):
        """
        Definição das variaveis independentes que iremos utilizar

        """
        variaveis_demurrage = [
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "CAPACIDADE/Dwt",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
            "Lag 1 mes Capacidade",
            "Lag 2 meses Capacidade",
            "Lag 3 meses Capacidade",
            "Lag 1 OEE",
            "Lag 1 DISPONIBILIDADE",
        ]

        variaveis_estadia = [
            "DISPONIBILIDADE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
            "Lag 1 mes Capacidade",
            "Lag 2 meses Capacidade",
            "Lag 3 meses Capacidade",
        ]

        variaveis_volume = [
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
        ]
        data_inicio_treino = "2019-07-01"

        pier = "Total"

        var_x = list(
            set(
                variaveis_demurrage
                + variaveis_estadia
                + variaveis_volume
            )
        )

        var_controle = [
            "Ano",
            "Mes",
            "Semana",
            "Trimestre",
            "Quinzena",
            "Periodo chuvoso",
        ]

        # Definidas as variáveis de inderesse, vamos fazer o filtro na base de
        # dados considerando as variáveis independentes, dependentes e porto
        # e pier que queremos analisar.

        self.df_data = self.df_data[
            (self.df_data["Port"] == self.porto) & (self.df_data["Pier"] == pier)
        ]
        self.df_data = self.df_data[
            [
                "Day",
                "Multa_Demurrage",
                "Estadia_media_navios_hs",
                "Volume_Embarcado",
                "Port",
            ]
            + var_x
        ]

        # Abaixo modificamos o sinal da variável custo de demurrage.

        self.df_data[["Multa_Demurrage"]] = self.df_data[["Multa_Demurrage"]] * (-1)

        # Como nao pode haver missing na base de dados utilizada para treinar
        # o modelo, vamos deletar os missings da base.

        self.df_data["Day"] = (pd.to_datetime(self.df_data["Day"])).dt.normalize()
        self.df_data = self.df_data[self.df_data["Day"] >= data_inicio_treino]
        self.df_data["Estadia_media_navios_hs"] = self.df_data[
            "Estadia_media_navios_hs"
        ].fillna(self.df_data["Estadia_media_navios_hs"].mean())
        self.df_data = self.df_data.dropna()
        self.df_data = self.df_data[
            self.df_data["Estadia_media_navios_hs"] != 0
        ].dropna()

        # Abaixo, criamos as variáveis que buscam captar a sazonalidade das
        # nossas variáveis dependentes.
        # A variavel Ano informa qual é o ano associado a cada observacao.
        # A variavel Mes informa o mes variando de 1 a 12.
        # A variavel Semana informa a semana, variando de 1 a 52.

        self.df_data["Ano"] = self.df_data["Day"].dt.year
        self.df_data["Mes"] = self.df_data["Day"].dt.month
        self.df_data["Semana"] = self.df_data["Day"].dt.week

        # A variavel Trimestre informa valores de 1 a 4, associados aos
        # trimestres do ano.

        self.df_data["Trimestre"] = np.where(
            self.df_data["Mes"] <= 3,
            1,
            np.where(
                self.df_data["Mes"] <= 6, 2, np.where(self.df_data["Mes"] <= 9, 3, 4)
            ),
        )

        # A variavel Quinzena varia de 1 a 104 associadas as quinzenas do ano.

        self.df_data["Quinzena"] = (self.df_data["Semana"] - 1) // 2 + 1

        # A variavel Periodo Chuvoso e uma dummy com valor 0 para o periodo
        # seco e valor 1 para o periodo chuvoso.

        self.df_data["Periodo chuvoso"] = np.where(
            self.df_data["Port"] == "Ponta Madeira",
            np.where((self.df_data["Mes"] >= 1) & (self.df_data["Mes"] <= 6), 1, 0),
            np.where((self.df_data["Mes"] >= 11) | (self.df_data["Mes"] <= 3), 1, 0),
        )
        self.df_data.pop("Port")
        self.df_data.set_index("Day", inplace=True)

        return [self.df_data, var_x, var_controle]


class TuningProphet:
    """A classe TuningProphet faz o tuning dos modelos Prophet utilizados,
    cada um, para prever uma das nossas variáveis dependentes. O essa variável
    prophet é incorporada a nossa base de dados e utilizada como variável
    independente dos modelos de random forest. Entretanto, essa variável
    prophet nao é utilizada na geracao dos gráfícos do shap, pois a
    interpretacao dos shap values nao fariam sentido.

    Args:

        df: Output da classe TratamentoDados.

        DATA_INICIO_TESTE: Data a partir da qual inicia nossa base de teste
        dos modelos.

        DATA_FIM_TESTE: Data que delimita o fim da base de teste dos
        modelos.

        hiperparametros_prophet: Valores de hiperparametros que permitem as
        melhores estimacoes do prophet.

        tuning_: Parametro que determina se sera feito o tuning ou se vamos utilizar
        valores fixos para os hiperparametros do prophet. Para tuning_ = 1 fazemos
        o tuning, para tuning_ = 0 utilizamos valores padrao.

    Returns:

        df: Dataframe com os dados completos.

        pred_d: Dataframe com a previsao da variável de custo de demurrage
        utilizando o prophet.

        pred_t: Dataframe com a previsao da variável de tempo de estadia
        utilizando o prophet.

        pred_v: Dataframe com a previsao da variável de volume embarcado
        utilizando o prophet.

    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        data_inicio_teste: str,
        data_fim_teste: str,
        hiperparametros_prophet: dict,
        tuning_: int,
    ):
        self.df_data = df_data
        self.data_inicio_teste = data_inicio_teste
        self.data_fim_teste = data_fim_teste
        self.hiperparametros_prophet = hiperparametros_prophet
        self.tuning_ = tuning_

    def resultados(self):
        """
        Vamos criar a variavel prophet, que sera acrescentada ao random
        forest. Teremos uma estimacao de prophet associada a cada variavel
        independentes. prophet_d para a Multa_Demurrage, prophet_t para a
        Estadia_media_navios_hs, prophet_v para o Volume_Embarcado.
        Temos que colocar as variáveis no formato a ser utilizado pelo
        prophet, com a variavel dependente com o nome y e a variável tempo
        como ds.
        """

        data_inicio_treino = "2019-07-01"

        prophet_d, prophet_t, prophet_v = (
            self.df_data[["Multa_Demurrage"]],
            self.df_data[["Estadia_media_navios_hs"]],
            self.df_data[["Volume_Embarcado"]],
        )

        prophet_d["Day"], prophet_t["Day"], prophet_v["Day"] = (
            prophet_d.index,
            prophet_t.index,
            prophet_v.index,
        )

        prophet_d.rename(columns={"Multa_Demurrage": "y", "Day": "ds"}, inplace=True)

        prophet_t.rename(
            columns={"Estadia_media_navios_hs": "y", "Day": "ds"}, inplace=True
        )

        prophet_v.rename(columns={"Volume_Embarcado": "y", "Day": "ds"}, inplace=True)

        prophet_d, prophet_t, prophet_v = (
            prophet_d[["y", "ds"]],
            prophet_t[["y", "ds"]],
            prophet_v[["y", "ds"]],
        )

        if self.tuning_ == 1:

            # Definida a funcao que faz o tunning, vamos aplicar o tunning a
            # estimacao de cada uma das variaveis dependentes.

            parameters_demurrage = tuning_prophet(
                self.df_data,
                "Multa_Demurrage",
                self.data_inicio_teste,
                self.data_fim_teste,
            )
            parameters_estadia = tuning_prophet(
                self.df_data,
                "Estadia_media_navios_hs",
                self.data_inicio_teste,
                self.data_fim_teste,
            )
            parameters_volume = tuning_prophet(
                self.df_data,
                "Volume_Embarcado",
                self.data_inicio_teste,
                self.data_fim_teste,
            )

            # Vamos salvar os hiperparametros.

            self.hiperparametros_prophet["Multa_Demurrage"]["changepoint_prior_scale"] = parameters_demurrage["changepoint_prior_scale"]
            self.hiperparametros_prophet["Multa_Demurrage"]["holidays_prior_scale"] = parameters_demurrage["holidays_prior_scale"]
            self.hiperparametros_prophet["Multa_Demurrage"]["n_changepoints"] = parameters_demurrage["n_changepoints"]
            self.hiperparametros_prophet["Multa_Demurrage"]["seasonality_mode"] = parameters_demurrage["seasonality_mode"]

            self.hiperparametros_prophet["Estadia_media_navios_hs"]["changepoint_prior_scale"] = parameters_estadia["changepoint_prior_scale"]
            self.hiperparametros_prophet["Estadia_media_navios_hs"]["holidays_prior_scale"] = parameters_estadia["holidays_prior_scale"]
            self.hiperparametros_prophet["Estadia_media_navios_hs"]["n_changepoints"] = parameters_estadia["n_changepoints"]
            self.hiperparametros_prophet["Estadia_media_navios_hs"]["seasonality_mode"] = parameters_estadia["seasonality_mode"]

            self.hiperparametros_prophet["Volume_Embarcado"]["changepoint_prior_scale"] = parameters_volume["changepoint_prior_scale"]
            self.hiperparametros_prophet["Volume_Embarcado"]["holidays_prior_scale"] = parameters_volume["holidays_prior_scale"]
            self.hiperparametros_prophet["Volume_Embarcado"]["n_changepoints"] = parameters_volume["n_changepoints"]
            self.hiperparametros_prophet["Volume_Embarcado"]["seasonality_mode"] = parameters_volume["seasonality_mode"]

            # Modelo prophet para a variavel custo de demurrage.

            m_d = Prophet(
                changepoint_prior_scale=parameters_demurrage["changepoint_prior_scale"],
                holidays_prior_scale=parameters_demurrage["holidays_prior_scale"],
                n_changepoints=parameters_demurrage["n_changepoints"],
                seasonality_mode=parameters_demurrage["seasonality_mode"],
            )

            # Modelo prophet para a variavel tempo de estadia.

            m_t = Prophet(
                changepoint_prior_scale=parameters_estadia["changepoint_prior_scale"],
                holidays_prior_scale=parameters_estadia["holidays_prior_scale"],
                n_changepoints=parameters_estadia["n_changepoints"],
                seasonality_mode=parameters_estadia["seasonality_mode"],
            )

            # Modelo prophet para a variavel volume embarcado.

            m_v = Prophet(
                changepoint_prior_scale=parameters_volume["changepoint_prior_scale"],
                holidays_prior_scale=parameters_volume["holidays_prior_scale"],
                n_changepoints=parameters_volume["n_changepoints"],
                seasonality_mode=parameters_volume["seasonality_mode"],
            )

        elif self.tuning_ == 0:

            m_d = Prophet(
                changepoint_prior_scale=self.hiperparametros_prophet["Multa_Demurrage"]["changepoint_prior_scale"],
                holidays_prior_scale=self.hiperparametros_prophet["Multa_Demurrage"]["holidays_prior_scale"],
                n_changepoints=self.hiperparametros_prophet["Multa_Demurrage"]["n_changepoints"],
                seasonality_mode=self.hiperparametros_prophet["Multa_Demurrage"]["seasonality_mode"],
            )

            m_t = Prophet(
                changepoint_prior_scale=self.hiperparametros_prophet["Estadia_media_navios_hs"]["changepoint_prior_scale"],
                holidays_prior_scale=self.hiperparametros_prophet["Estadia_media_navios_hs"]["holidays_prior_scale"],
                n_changepoints=self.hiperparametros_prophet["Estadia_media_navios_hs"]["n_changepoints"],
                seasonality_mode=self.hiperparametros_prophet["Estadia_media_navios_hs"]["seasonality_mode"],
            )

            m_v = Prophet(
                changepoint_prior_scale=self.hiperparametros_prophet["Volume_Embarcado"]["changepoint_prior_scale"],
                holidays_prior_scale=self.hiperparametros_prophet["Volume_Embarcado"]["holidays_prior_scale"],
                n_changepoints=self.hiperparametros_prophet["Volume_Embarcado"]["n_changepoints"],
                seasonality_mode=self.hiperparametros_prophet["Volume_Embarcado"]["seasonality_mode"],
            )

        # Abaixo, vazemos o fit do modelo para cada uma das variaveis
        # dependentes.

        m_d.fit(prophet_d)
        m_t.fit(prophet_t)
        m_v.fit(prophet_v)

        # Utilizando os modelos treinados, vamos fazer as previsoes de cada uma
        # dessas variaveis para o periodo futuro.

        future_d = m_d.make_future_dataframe(
            periods=len(
                self.df_data[
                    (self.df_data.index >= self.data_inicio_teste)
                    ]
            ),
            freq="W",
        )

        future_t = m_t.make_future_dataframe(
            periods=len(
                self.df_data[
                    (self.df_data.index >= self.data_inicio_teste)
                ]
            ),
            freq="W",
        )

        future_v = m_v.make_future_dataframe(
            periods=len(
                self.df_data[
                    (self.df_data.index >= self.data_inicio_teste)
                ]
            ),
            freq="W",
        )

        forecast_d, forecast_t, forecast_v = (
            m_d.predict(future_d),
            m_t.predict(future_t),
            m_v.predict(future_v),
        )

        # Feitas as previsoes, vamos filtrar as variaveis com as previsoes
        # dentro da base de dados e modificar os nomes para acrescentar
        # na nossa base final.

        pred_d, pred_t, pred_v = (
            forecast_d[["ds", "yhat"]],
            forecast_t[["ds", "yhat"]],
            forecast_v[["ds", "yhat"]],
        )

        pred_d.rename(columns={"ds": "Day", "yhat": "prophet_d"}, inplace=True)
        pred_t.rename(columns={"ds": "Day", "yhat": "prophet_t"}, inplace=True)
        pred_v.rename(columns={"ds": "Day", "yhat": "prophet_v"}, inplace=True)

        pred_d, pred_t, pred_v = (
            pred_d.set_index("Day"),
            pred_t.set_index("Day"),
            pred_v.set_index("Day"),
        )

        # Por fim, vamos fazer o merge das variaveis previstas com o prophet
        # as demais variaveis da nossa base de dados.

        self.df_data = pd.merge(
            self.df_data,
            pd.merge(
                pred_d,
                pd.merge(pred_t, pred_v, left_index=True, right_index=True),
                left_index=True,
                right_index=True,
            ),
            left_index=True,
            right_index=True,
        )
        pred_d.reset_index(inplace=True)
        pred_t.reset_index(inplace=True)
        pred_v.reset_index(inplace=True)

        return [self.df_data, pred_d, pred_t, pred_v,self.hiperparametros_prophet]


class DataSplit:
    """A classe DataSplit separa a nossa base de dados entre base de treino e
    base de teste. Também cria bases apenas com dados de período chuvoso ou
    apenas período seco.

    Args:

        df: Output da classe TuningProphet.

        DATA_INICIO_TESTE: Data a partir da qual inicia nossa base de teste
        dos modelos.

        DATA_FIM_TESTE: Data que delimita o fim da base de teste dos
        modelos.

    Returns:

        train_features: Base de treino com as variáveis independentes para
        o modelo de random forest.

        train_label: Base de treino com as variáveis dependentes para o
        modelo de random forest.

        test_features: Base de teste com as variáveis independentes para o
        modelo de random forest.

        test_label: Base de teste com as variáveis dependentes para o
        modelo de random forest.

        train_features_seco: Base de treino com as variáveis independentes
        para o modelo de random forest, considerando apenas o período seco.

        train_label_seco: Base de treino com as variáveis dependentes para
        o modelo de random forest, considerando apenas o período seco.

        train_features_chuvoso: Base de treino com as variáveis
        independentes para o modelo de random forest, considerando apenas o
        período chuvoso.

        train_label_chuvoso: Base de teste com as variáveis dependentes
        para o modelo de random forest, considerando apenas o período chuvoso.

        var_demurrage: Lista com as variáveis independentes utilizadas no
        modelo que preve o custo de demurrage.

        var_estadia: Lista com as variáveis independentes utilizadas no
        modelo que preve o tempo de estadia.

        var_volume: Lista com as variáveis independentes utilizadas no
        modelo que preve o volume embarcado.

        df_train: Dataframe completo de treino.

        df_test: Dataframe complesto de teste.

        df_train_seco: Dataframe de treino, apenas com o período seco.

        df_train_chuvoso: Dataframe de treino, apenas com o período
        chuvoso.

    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        data_inicio_teste: str,
        data_fim_teste: str,
    ):
        self.df_data = df_data
        self.data_inicio_teste = data_inicio_teste
        self.data_fim_teste = data_fim_teste

    def resultados(self):
        """O código abaixo executa o split da base, alem de selecionar quais
        variaveis serao utilizadas.

        """
        data_inicio_treino = "2019-07-01"

        variaveis_demurrage = [
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "CAPACIDADE/Dwt",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
            "Lag 1 mes Capacidade",
            "Lag 2 meses Capacidade",
            "Lag 3 meses Capacidade",
            "Lag 1 OEE",
            "Lag 1 DISPONIBILIDADE",
        ]

        variaveis_estadia = [
            "DISPONIBILIDADE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
            "Lag 1 mes Capacidade",
            "Lag 2 meses Capacidade",
            "Lag 3 meses Capacidade",
        ]

        variaveis_volume = [
            "DISPONIBILIDADE",
            "UTILIZACAO",
            "OEE",
            "Dwt_K_total",
            "Dwt_K_medio",
            "Pcg-FOB",
            "Pcg-CFR",
            "Pcg-SPOT/FOB",
            "Pcg-Frota Dedicada/SPOT/FOB",
            "Pcg-Frota Dedicada/FOB",
            "Pcg-Frota Dedicada",
            "Pcg-PANAMAX",
            "Pcg-CAPE",
            "Pcg-VLOC",
            "Pcg-NEWCASTLE",
            "Pcg-VALEMAX",
            "Pcg-BABYCAPE",
            "Lag 1 mes numero de navios na fila",
            "Lag 2 meses numero de navios na fila",
            "Lag 3 meses numero de navios na fila",
        ]

        df_train = self.df_data[
            (self.df_data.index >= data_inicio_treino)
            & (self.df_data.index < self.data_inicio_teste)
        ]

        df_train_seco = self.df_data[
            (self.df_data.index >= data_inicio_treino)
            & (self.df_data.index < self.data_inicio_teste)
            & (self.df_data["Periodo chuvoso"] == 0)
        ][self.df_data.columns.difference(["Periodo chuvoso"])]

        df_train_chuvoso = self.df_data[
            (self.df_data.index >= data_inicio_treino)
            & (self.df_data.index < self.data_inicio_teste)
            & (self.df_data["Periodo chuvoso"] == 1)
        ][self.df_data.columns.difference(["Periodo chuvoso"])]

        df_test = self.df_data[(self.df_data.index >= self.data_inicio_teste)]

        print(df_train.shape)

        # Abaixo filtramos nas bases apenas as variaveis depedentes e
        # e independentes entre as bases de treino e teste.

        train_features = df_train[
            df_train.columns.difference(
                ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
            )
        ]

        train_label = df_train[
            ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
        ]

        test_features = df_test[
            df_test.columns.difference(
                ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
            )
        ]

        test_label = df_test[
            ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
        ]

        # Abaixo, separamos apenas o periodo seco.

        train_features_seco = df_train_seco[
            df_train_seco.columns.difference(
                ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
            )
        ]

        train_label_seco = df_train_seco[
            ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
        ]

        # Abaixo, separamos apenas o periodo chuvoso.

        train_features_chuvoso = df_train_chuvoso[
            df_train_chuvoso.columns.difference(
                ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
            )
        ]

        train_label_chuvoso = df_train_chuvoso[
            ["Multa_Demurrage", "Estadia_media_navios_hs", "Volume_Embarcado"]
        ]

        # Abaixo, temos uma lista para cada um dos 9 modelos estimados

        # As listas de variaveis independentes para os modelos de custo,
        # estadia e volume

        var_demurrage = variaveis_demurrage + ["prophet_d"]
        var_estadia = variaveis_estadia + ["prophet_t"]
        var_volume = variaveis_volume + ["prophet_v"]

        return [
            train_features,
            train_label,
            test_features,
            test_label,
            train_features_seco,
            train_label_seco,
            train_features_chuvoso,
            train_label_chuvoso,
            var_demurrage,
            var_estadia,
            var_volume,
            df_train,
            df_test,
            df_train_seco,
            df_train_chuvoso,
        ]


class Modelos:
    """A classe Modelos faz o tunning dos modelos de random forest para cada
    uma das variáveis dependentes (demurrage, tempo de estadia, volume), e para
    tres indicadores (OEE,UF,DF,CAPACIDADE,etc) baseados nas médias.

    Args:

        df: Output da classe DataSplit.

        PORTO: Porto que iremos considerar no modelo.

        var_demurrage: Output da classe DataSplit.

        var_estadia: Output da classe DataSplit.

        var_volume: Output da classe DataSplit.

        pred_d: Output da classe TuningProphet.

        pred_t: Output da classe TuningProphet.

        pred_v: Output da classe TuningProphet.

        train_features: Output da classe DataSplit.

        train_label: Output da classe DataSplit.

        test_features: Output da classe DataSplit.

        DATA_INICIO_TESTE: DATA_INICIO_TESTE = Data a partir da qual
        inicia nossa base de teste dos modelos.

        wd: String com o endereco do diretório de trabalho.

        save_rf: Endereco no qual vamos salvar os resultados das previsoes
        dos modelos de random forest.

        tuning_: Parametro que determina se faremos o tuning para os modelos de
        random forest ou nao. Se tuning_ = 1, fazemos tuning. Se tuning_ = 0,
        executamos com os valores padrao.

        hp_rf_prev: Hiperparametros de random forest salvos na última estimacao
        do modelo.

    Returns:

        resultado: Lista com um conjunto de 9 dataframes. Sao previsoes correspondentes
        a 3 variáves dependentes Custo de Demurrage, Tempo de Estadia, Volume
        Embarcado.

        hp_rf_prev: Lista com os hiperparametros da ultima estimacao do modelo.
    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        porto: str,
        var_demurrage: list,
        var_estadia: list,
        var_volume: list,
        pred_d: pd.DataFrame,
        pred_t: pd.DataFrame,
        pred_v: pd.DataFrame,
        train_features: pd.DataFrame,
        train_label: pd.DataFrame,
        test_features: pd.DataFrame,
        data_inicio_teste: str,
        w_d: str,
        save_rf: str,
        tuning_: int,
        hp_rf_prev: dict,
    ):
        self.df_data = df_data
        self.porto = porto
        self.var_demurrage = var_demurrage
        self.var_estadia = var_estadia
        self.var_volume = var_volume
        self.pred_d = pred_d
        self.pred_t = pred_t
        self.pred_v = pred_v
        self.train_features = train_features
        self.train_label = train_label
        self.test_features = test_features
        self.data_inicio_teste = data_inicio_teste
        self.w_d = w_d
        self.save_rf = save_rf
        self.tuning_ = tuning_
        self.hp_rf_prev = hp_rf_prev

    def resultado(self):
        """Abaixo fazemos o Tuning dos modelos de random forest para cada cenario."""

        resultado = []  # lista para salvar os resultados

        pier = "Total"

        for var in (
            ("Multa_Demurrage", self.var_demurrage, "Medias", self.pred_d),
            ("Estadia_media_navios_hs", self.var_estadia, "Medias", self.pred_t),
            ("Volume_Embarcado", self.var_volume, "Medias", self.pred_v)
        ):

            var_y = var[0]
            var_x = var[1]

            if self.tuning_ == 1:

                # Abaixo, definimos as listas de valores a serem testados para
                # cada um dos hiperparametros do random forest a serem utilizados
                # no tunning.

                n_estimators = [
                    int(x) for x in np.linspace(start=200, stop=2000, num=10)
                ]  # Number of trees in random forest
                # Number of features to consider at every split
                max_features = ["auto", "sqrt"]
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                max_depth.append(None)
                # Minimum number of samples required to split a node
                min_samples_split = [2, 5, 10]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]

                # Vamos utilizar random grid para fazer o random forest
                # O random grid testa uma amostre entre todas as combinacoes possiveis
                # dados os varlores definidos nas listas dos hiperparametros.
                # Nao vamos testar todos os valores possiveis para que o tempo de
                # execucao do modelo seja menor.

                random_grid = {
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": bootstrap,
                }

                # Primeiro vamos criar o modelo base do random forest

                r_f = RandomForestRegressor()

                r_f_random = RandomizedSearchCV(
                    estimator=r_f,
                    param_distributions=random_grid,
                    n_iter=100,
                    cv=3,
                    verbose=2,
                    random_state=42,
                    n_jobs=-1,
                )

                r_f_random.fit(self.train_features[var_x], self.train_label[var_y])

                # Com base nos valores do random search, podemos definir os valores
                # de hiperparametros que iremos testar em cada modelo.

                if r_f_random.best_params_["max_depth"] is None:
                    param_grid = {
                        "bootstrap": [r_f_random.best_params_["bootstrap"]],
                        "max_depth": [r_f_random.best_params_["max_depth"]],
                        "max_features": [r_f_random.best_params_["max_features"]],
                        "min_samples_leaf": [1, 2, 4],
                        "min_samples_split": [2, 5, 10],
                        "n_estimators": [r_f_random.best_params_["n_estimators"]]
                        + [r_f_random.best_params_["n_estimators"] + 100]
                        + [r_f_random.best_params_["n_estimators"] + 200],
                    }
                else:
                    param_grid = {
                        "bootstrap": [r_f_random.best_params_["bootstrap"]],
                        "max_depth": [r_f_random.best_params_["max_depth"]]
                        + [r_f_random.best_params_["max_depth"] + 10]
                        + [r_f_random.best_params_["max_depth"] + 20],
                        "max_features": [r_f_random.best_params_["max_features"]],
                        "min_samples_leaf": [1, 2, 4],
                        "min_samples_split": [2, 5, 10],
                        "n_estimators": [r_f_random.best_params_["n_estimators"]]
                        + [r_f_random.best_params_["n_estimators"] + 100]
                        + [r_f_random.best_params_["n_estimators"] + 200],
                    }

                # Novamente criamos o modelo base

                r_f = RandomForestRegressor()

                # Instanciamos o GridSearch

                grid_search = GridSearchCV(
                    estimator=r_f, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
                )

                # Fazemos o fit do grid_search com os dados.

                grid_search.fit(self.train_features[var_x], self.train_label[var_y])

                # Definimos o modelo com o melhores hiperparametros.

                r_f = RandomForestRegressor(
                    n_estimators=grid_search.best_params_["n_estimators"],
                    min_samples_split=grid_search.best_params_["min_samples_split"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    max_features=grid_search.best_params_["max_features"],
                    max_depth=grid_search.best_params_["max_depth"],
                    bootstrap=grid_search.best_params_["bootstrap"],
                    random_state=42,
                )

                self.hp_rf_prev[var_y]["n_estimators"] = grid_search.best_params_["n_estimators"]
                self.hp_rf_prev[var_y]["min_samples_split"] = grid_search.best_params_["min_samples_split"]
                self.hp_rf_prev[var_y]["min_samples_leaf"] = grid_search.best_params_["min_samples_leaf"]
                self.hp_rf_prev[var_y]["max_features"] = grid_search.best_params_["max_features"]
                self.hp_rf_prev[var_y]["max_depth"] = grid_search.best_params_["max_depth"]
                self.hp_rf_prev[var_y]["bootstrap"] = grid_search.best_params_["bootstrap"]
                self.hp_rf_prev[var_y]['random_state'] = 42

            elif self.tuning_ == 0:
                if var[0] == 'Multa_Demurrage' and self.porto == 'Guaiba e Sepetiba':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Multa_Demurrage' and self.porto == 'Ponta Madeira':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Multa_Demurrage' and self.porto == 'Tubarao':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Estadia_media_navios_hs' and self.porto == 'Guaiba e Sepetiba':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Estadia_media_navios_hs' and self.porto == 'Ponta Madeira':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Estadia_media_navios_hs' and self.porto == 'Tubarao':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Volume_Embarcado' and self.porto == 'Guaiba e Sepetiba':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Volume_Embarcado' and self.porto == 'Ponta Madeira':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )
                elif var[0] == 'Volume_Embarcado' and self.porto == 'Tubarao':
                    r_f = RandomForestRegressor(
                        n_estimators=self.hp_rf_prev[var_y]["n_estimators"],
                        min_samples_split=self.hp_rf_prev[var_y]["min_samples_split"],
                        min_samples_leaf=self.hp_rf_prev[var_y]["min_samples_leaf"],
                        max_features=self.hp_rf_prev[var_y]["max_features"],
                        max_depth=self.hp_rf_prev[var_y]["max_depth"],
                        bootstrap=self.hp_rf_prev[var_y]["bootstrap"],
                        random_state=self.hp_rf_prev[var_y]['random_state'],
                    )

            # Treinamos o modelo

            r_f.fit(self.train_features[var_x], self.train_label[var_y])

            # Fazemos a previsao

            df_forecast = r_f.predict(self.test_features[var_x])

            # Acrescentamos a variavel com os valores previstos a nossa base
            # variavel prevista com base na média dos valores passados.

            previsao_modelo = pd.merge(
                pd.merge(
                    self.df_data[[var_y]][
                        self.df_data.index >= self.data_inicio_teste
                    ].reset_index(level=0),
                    pd.DataFrame(df_forecast),
                    left_index=True,
                    right_index=True,
                ),
                var[3],
                on="Day",
                how="inner",
            )

            # Modificamos o nome da variavel prevista

            previsao_modelo.rename(columns={0: "r_f+Prophet"}, inplace=True)

            # Salvamos o resultado de previsao do nosso melhor modelo

            resultado.append(previsao_modelo)

            # Transformamos em um arquivo excel

            previsao_modelo.to_excel(
                self.w_d
                + self.save_rf
                + self.porto
                + " "
                + pier
                + " "
                + var_y
                + " "
                + var[2]
                + " "
                + ".xlsx"
            )

        return [resultado,self.hp_rf_prev]
