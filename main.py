"""O script abaixo faz previsoes de Demurrage, Volume e Tempo de Estadia
"""

# Etapa 1: importacao das bibliotecas necessárias

import os
import json
import shutil
import numpy as np
import pandas as pd
from datetime import date
from dotenv import load_dotenv

load_dotenv(
    "C:/Users/KG858HY/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/02 - Previsão de Demurrage/04-Dados/3-Scripts/env"
)

os.chdir(os.getenv("CAMINHO_PROJETO"))

from classes_shape import (
    Modelos,
    DataSplit,
    TuningProphet,
    TratamentoDados,
)
from classes_data_processing import (
    BaseUnica,
    Projecoes,
    Importacao,
    NovosNomes,
    Indicadores,
    Semanalizacao,
    VariaveisDependentes,
    Datas_Inicio_Fim_Teste,
    MergeDemurrageOperationalDesk,
)

from tratamento_dados_pbi import Clusterizacao

# Etapa 2: Definindo parametros

# Defina tuning_ = 1 se quiser fazer o tuning do modelo.
# Se quiser rodar com os hiperparametros da ultima estimacao,
# escolha tuing_ = 0

tuning_ = 1

# Lista de hiperparametros que utilizamos no prophet quando nao executamos o
# tuning do modelo.

with open(os.getenv("HIPERPARAMETROS")) as file:
    hp_prophet = json.load(file)
    hp_prophet = json.loads(hp_prophet)

# Vamos importar os hiperparametros utilizados na última vez que fizemos o tuning
# dos modelos preditivos de random forest

with open(os.getenv("HIPERPARAMETROS_RF_PREV")) as file:
    hp_rf_prev = json.load(file)
    hp_rf_prev = json.loads(hp_rf_prev)

# Vamos importar os hiperparametros utilizados na última vez que fizemos o tuning
# dos modelos clusterizacao de random forest

with open(os.getenv("HIPERPARAMETROS_RF_CLUSTER")) as file:
    hp_rf_cluster = json.load(file)
    hp_rf_cluster = json.loads(hp_rf_cluster)

# Caminho para salvar resultados random forest

SAVE_RF = "Resultados Random Forest/"

# Caminho onde estao salvas as bases que usamos de insumo para a base semanal

# Versão Cassio
WD1 = os.getenv("WD1")

# Caminho onde vamos salvar os dados de previsao dos modelos

# Versão Cassio
WD2 = os.getenv("WD2")

# Caminho para salvar as bases que sao insumo para o powerbi

WD3 = os.getenv("WD3")

# Etapa 3: importacao de dados

# Seguem os caminhos das pastas onde arquivasmos as bases utilizadas

demurrage_folder = os.getenv("DEMURRAGE_FOLDER")
op_folder = os.getenv("OP_FOLDER")
indicadores_pm_folder = os.getenv("INDICADORES_PM_FOLDER")
indicadores_s_folder = os.getenv("INDICADORES_S_FOLDER")
indicadores_g_folder = os.getenv("INDICADORES_G_FOLDER")
indicadores_t_folder = os.getenv("INDICADORES_T_FOLDER")
historico = pd.read_excel(os.getenv("HISTORICO_PREVISOES"))

# Na linha abaixo, carregamos uma classe que importa e junta as bases

bases_importadas = Importacao(
    demurrage_folder,
    op_folder,
    indicadores_pm_folder,
    indicadores_s_folder,
    indicadores_g_folder,
    indicadores_t_folder,
).dados()

(bases_demurrage, bases_op, indices_PM, indices_S, indices_G, indices_T) = (
    bases_importadas[0],
    bases_importadas[1],
    bases_importadas[2],
    bases_importadas[3],
    bases_importadas[4],
    bases_importadas[5],
)

dataset = pd.concat(bases_demurrage, axis=0, ignore_index=True)

operational_desk = pd.concat(bases_op, axis=0, ignore_index=True)

indices1 = indices_PM[[x for x in list(indices_PM.columns) if "X1." in x]]
indices1.columns = list(
    map(
        lambda x: x.replace("X1.", ""),
        [x for x in list(indices_PM.columns) if "X1." in x],
    )
)

indices2 = indices_PM[[x for x in list(indices_PM.columns) if "X3N." in x]]
indices2.columns = list(
    map(
        lambda x: x.replace("X3N.", ""),
        [x for x in list(indices_PM.columns) if "X3N." in x],
    )
)

indices3 = indices_PM[[x for x in list(indices_PM.columns) if "X3S." in x]]
indices3.columns = list(
    map(
        lambda x: x.replace("X3S.", ""),
        [x for x in list(indices_PM.columns) if "X3S." in x],
    )
)

indices4 = indices_PM[[x for x in list(indices_PM.columns) if "X4N." in x]]
indices4.columns = list(
    map(
        lambda x: x.replace("X4N.", ""),
        [x for x in list(indices_PM.columns) if "X4N." in x],
    )
)

indices5 = indices_PM[[x for x in list(indices_PM.columns) if "X4S." in x]]
indices5.columns = list(
    map(
        lambda x: x.replace("X4S.", ""),
        [x for x in list(indices_PM.columns) if "X4S." in x],
    )
)

indices6 = indices_S[[x for x in list(indices_S.columns) if "CNS1." in x]]
indices6.columns = list(
    map(
        lambda x: x.replace("CNS1.", ""),
        [x for x in list(indices_S.columns) if "CNS1." in x],
    )
)

indices7 = indices_G[[x for x in list(indices_G.columns) if "CNG2." in x]]
indices7.columns = list(
    map(
        lambda x: x.replace("CNG2.", ""),
        [x for x in list(indices_G.columns) if "CNG2." in x],
    )
)

indices8 = indices_T[[x for x in list(indices_T.columns) if "X02." in x]]
indices8.columns = list(
    map(
        lambda x: x.replace("X02.", ""),
        [x for x in list(indices_T.columns) if "X02." in x],
    )
)

indices9 = indices_T[[x for x in list(indices_T.columns) if "X1N." in x]]
indices9.columns = list(
    map(
        lambda x: x.replace("X1N.", ""),
        [x for x in list(indices_T.columns) if "X1N." in x],
    )
)

indices10 = indices_T[[x for x in list(indices_T.columns) if "X1S." in x]]
indices10.columns = list(
    map(
        lambda x: x.replace("X1S.", ""),
        [x for x in list(indices_T.columns) if "X1S." in x],
    )
)

# A classe abaixo determina os períodos que utilizaremos como teste para o modelo
# identificando até qual data os dados estao disponíveis em cada base.

datas_inicio_fim = Datas_Inicio_Fim_Teste(
    dataset,
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
).datas()

DATA_INICIO_TESTE, DATA_FIM_TESTE = datas_inicio_fim[0], datas_inicio_fim[1]

# Etapa 4: Unir os dados das bases de Demurrage e Operational Desk

nova_base = MergeDemurrageOperationalDesk(dataset, operational_desk).resultados()
(
    data_inicial_real,
    data_final_real,
    data_inicio_projecoes,
    data_final_projecoes,
    dataset_g,
    dataset_s,
    dataset_gs,
    dataset_pm,
    dataset_t,
    dataset,
    dataset_,
    operational_desk,
) = (
    nova_base[0],
    nova_base[1],
    nova_base[2],
    nova_base[3],
    nova_base[4],
    nova_base[5],
    nova_base[6],
    nova_base[7],
    nova_base[8],
    nova_base[9],
    nova_base[10],
    nova_base[11],
)

# Etapa 5: Vamos unificar todas as variaveis criadas em uma base

base_diaria = BaseUnica(
    dataset,
    dataset_,
    operational_desk,
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
).resultado()

# Etapa 6: Acrescentamos a base variaveis de indicadores: UF, DF, OEE e demais

indicadores = Indicadores(
    base_diaria,
    data_inicial_real,
    data_final_real,
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
).bases()
base_g, base_s, base_pm, base_t, base_gs, embarcado_g, embarcado_s = (
    indicadores[0],
    indicadores[1],
    indicadores[2],
    indicadores[3],
    indicadores[4],
    indicadores[5],
    indicadores[6],
)

# Etapa 8: Vamos acrescentar a variável de custo de demurrage e volume

demurrageestadia = VariaveisDependentes(
    dataset_, base_g, base_s, base_gs, base_pm, base_t
).resultados()
base_g, base_s, base_gs, base_pm, base_t, base_diaria = (
    demurrageestadia[0],
    demurrageestadia[1],
    demurrageestadia[2],
    demurrageestadia[3],
    demurrageestadia[4],
    demurrageestadia[5],
)

# Etapa 10: Vamos transformar a base a nível diária para uma base de dados a nível semanal

bases = Semanalizacao(base_diaria).base()
base_semanal, base_diaria = bases[0], bases[1]

# Etapa 11: Fazemos previsao para o período futuro com base em médias.
# Tambem  acrescentamos valores de COI e CMA

previsoes = Projecoes(
    base_diaria,
    base_semanal,
    data_final_real,
    data_inicial_real,
    data_inicio_projecoes,
    data_final_projecoes,
).result()
base_diaria, base_semanal = previsoes[0], previsoes[1]

# Etapa 13: Vamos mudar os nomos das variáveis

nomes = NovosNomes(base_diaria, base_semanal).base()

base_diaria, base_semanal = nomes[0], nomes[1]

base_diaria["Lag 1 mes numero de navios na fila"] = base_diaria.groupby(["Port", "Pier"])["Lag 1 mes numero de navios na fila"].ffill()
base_diaria["Lag 2 meses numero de navios na fila"] = base_diaria.groupby(["Port", "Pier"])["Lag 2 meses numero de navios na fila"].ffill()
base_diaria["Lag 3 meses numero de navios na fila"] = base_diaria.groupby(["Port", "Pier"])["Lag 3 meses numero de navios na fila"].ffill()
base_diaria["Lag 1 mes Capacidade"] = base_diaria.groupby(["Port", "Pier"])["Lag 1 mes Capacidade"].ffill()
base_diaria["Lag 2 meses Capacidade"] = base_diaria.groupby(["Port", "Pier"])["Lag 2 meses Capacidade"].ffill()
base_diaria["Lag 3 meses Capacidade"] = base_diaria.groupby(["Port", "Pier"])["Lag 3 meses Capacidade"].ffill()
base_diaria["Lag 1 OEE"] = base_diaria.groupby(["Port", "Pier"])["Lag 1 OEE"].ffill()
base_diaria["Lag 1 DISPONIBILIDADE"] = base_diaria.groupby(["Port", "Pier"])["Lag 1 DISPONIBILIDADE"].ffill()
base_diaria["DISPONIBILIDADE"] = base_diaria.groupby(["Port", "Pier"])["DISPONIBILIDADE"].ffill()
base_diaria["UTILIZACAO"] = base_diaria.groupby(["Port", "Pier"])["UTILIZACAO"].ffill()
base_diaria["OEE"] = base_diaria.groupby(["Port", "Pier"])["OEE"].ffill()
base_diaria["CAPACIDADE"] = base_diaria.groupby(["Port", "Pier"])["CAPACIDADE"].ffill()
base_diaria["TAXA_EFETIVA"] = base_diaria.groupby(["Port", "Pier"])["TAXA_EFETIVA"].ffill()

base_semanal["Lag 1 mes numero de navios na fila"] = base_semanal.groupby(["Port", "Pier"])["Lag 1 mes numero de navios na fila"].ffill()
base_semanal["Lag 2 meses numero de navios na fila"] = base_semanal.groupby(["Port", "Pier"])["Lag 2 meses numero de navios na fila"].ffill()
base_semanal["Lag 3 meses numero de navios na fila"] = base_semanal.groupby(["Port", "Pier"])["Lag 3 meses numero de navios na fila"].ffill()
base_semanal["Lag 1 mes Capacidade"] = base_semanal.groupby(["Port", "Pier"])["Lag 1 mes Capacidade"].ffill()
base_semanal["Lag 2 meses Capacidade"] = base_semanal.groupby(["Port", "Pier"])["Lag 2 meses Capacidade"].ffill()
base_semanal["Lag 3 meses Capacidade"] = base_semanal.groupby(["Port", "Pier"])["Lag 3 meses Capacidade"].ffill()
base_semanal["Lag 1 OEE"] = base_semanal.groupby(["Port", "Pier"])["Lag 1 OEE"].ffill()
base_semanal["Lag 1 DISPONIBILIDADE"] = base_semanal.groupby(["Port", "Pier"])["Lag 1 DISPONIBILIDADE"].ffill()
base_semanal["DISPONIBILIDADE"] = base_semanal.groupby(["Port", "Pier"])["DISPONIBILIDADE"].ffill()
base_semanal["UTILIZACAO"] = base_semanal.groupby(["Port", "Pier"])["UTILIZACAO"].ffill()
base_semanal["OEE"] = base_semanal.groupby(["Port", "Pier"])["OEE"].ffill()
base_semanal["CAPACIDADE"] = base_semanal.groupby(["Port", "Pier"])["CAPACIDADE"].ffill()
base_semanal["TAXA_EFETIVA"] = base_semanal.groupby(["Port", "Pier"])["TAXA_EFETIVA"].ffill()
    
# Etapa 14: Vamos exportar as bases diária e semanal

# "Quantity_t", na última interação virou Volume_Embarcado
base_diaria = base_diaria[
    [
        "Port",
        "Pier",
        "Multa_Demurrage",
        "Estadia_media_navios_hs",
        "Volume_Embarcado",
        "Quantity_t",
        "Lag 1 mes numero de navios na fila",
        "Lag 2 meses numero de navios na fila",
        "Lag 3 meses numero de navios na fila",
        "Lag 1 mes Capacidade",
        "Lag 2 meses Capacidade",
        "Lag 3 meses Capacidade",
        "Lag 1 OEE",
        "Lag 1 DISPONIBILIDADE",
        "Navios na fila",
        "Navios que chegaram",
        "Navios TCA",
        "Navios carregando",
        "Navios desatracando",
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
        "Dwt_K_total",
        "Dwt_K_medio",
        "Qtde/Dwt",
        "DISPONIBILIDADE",
        "UTILIZACAO",
        "OEE",
        "CAPACIDADE",
        "CAPACIDADE/Dwt",
        "TAXA_EFETIVA",
        "Mudanca Politica",
        "Soma Estadia em Horas",
        "Real/Previsto",
    ]
]

# "Quantity_t", na última interação virou Volume_Embarcado
base_semanal = base_semanal[
    [
        "Port",
        "Pier",
        "Multa_Demurrage",
        "Estadia_media_navios_hs",
        "Volume_Embarcado",
        "Quantity_t",
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
        "Dwt_K_total",
        "Dwt_K_medio",
        "Qtde/Dwt",
        "DISPONIBILIDADE",
        "UTILIZACAO",
        "OEE",
        "CAPACIDADE",
        "CAPACIDADE/Dwt",
        "TAXA_EFETIVA",
        "Mudanca Politica",
        "Período",
    ]
]

# Agora exportamos os dados

base_diaria.to_excel(
    os.getenv("CAMINHO_BASE_DIARIA")
    + date.today().strftime("%d%m%Y")
    + "-base_diaria.xlsx"
)

base_semanal.to_excel(
    os.getenv("CAMINHO_BASE_SEMANAL")
    + date.today().strftime("%d%m%Y")
    + "-base_semanal.xlsx"
)


# Etapa 15: Fazemos o tratamento de dados para previsao

df = base_semanal.reset_index()

semanal_prev = base_semanal.reset_index()
semanal_prev["Real/Previsto"] = np.where(
    semanal_prev["Day"] > pd.to_datetime(data_final_real), "Previsto", "Real"
)
semanal_sc = semanal_prev.copy()
semanal_sc["Período"] = "Seco e Chuvoso"
semanal_prev = pd.concat([semanal_prev, semanal_sc], axis=0, ignore_index=True)

PORTO = ["Ponta Madeira", "Guaiba e Sepetiba", "Tubarao"]

df_ = []

for i in range(len(PORTO)):
    df_.append(df[df["Port"] == PORTO[i]])

df = df_

df_ = []
var_x = []
var_controle = []

for i in range(len(PORTO)):
    dados_tratados = TratamentoDados(
        df[i],
        PORTO[i],
        DATA_INICIO_TESTE,
        DATA_FIM_TESTE,
    ).resultado()
    df_.append(dados_tratados[0])
    var_x.append(dados_tratados[1])
    var_controle.append(dados_tratados[2])

df = df_

# df, var_x, var_controle = dados_tratados[0], dados_tratados[1], dados_tratados[2]

# Etapa 16: Fazemos o tunning do Prophet

df_ = []
pred_d = []
pred_t = []
pred_v = []
hp = {}

for i in range(len(PORTO)):
    tuning = TuningProphet(
        df[i], DATA_INICIO_TESTE, DATA_FIM_TESTE, hp_prophet[PORTO[i]], tuning_
    ).resultados()
    df_.append(tuning[0])
    pred_d.append(tuning[1])
    pred_t.append(tuning[2])
    pred_v.append(tuning[3])
    hp[PORTO[i]] = tuning[4]

hp_prophet = json.dumps(hp)
with open(os.getenv("HIPERPARAMETROS"), "w", encoding="utf-8") as f:
    json.dump(hp_prophet, f, ensure_ascii=False, indent=4)
df = df_

# df, pred_d, pred_t, pred_v = tuning[0], tuning[1], tuning[2], tuning[3]

# Etapa 17: Vamos separar as bases de dados entre base de treino e teste

train_features = []
train_label = []
test_features = []
test_label = []
train_features_seco = []
train_label_seco = []
train_features_chuvoso = []
train_label_chuvoso = []
var_demurrage = []
var_estadia = []
var_volume = []
var_demurrage_prem = []
var_estadia_prem = []
var_volume_prem = []
var_demurrage_sim = []
var_estadia_sim = []
var_volume_sim = []
df_train = []
df_test = []
df_train_seco = []
df_train_chuvoso = []

for i in range(len(PORTO)):
    data_split = DataSplit(
        df[i],
        DATA_INICIO_TESTE,
        DATA_FIM_TESTE,
    ).resultados()
    train_features.append(data_split[0])
    train_label.append(data_split[1])
    test_features.append(data_split[2])
    test_label.append(data_split[3])
    train_features_seco.append(data_split[4])
    train_label_seco.append(data_split[5])
    train_features_chuvoso.append(data_split[6])
    train_label_chuvoso.append(data_split[7])
    var_demurrage.append(data_split[8])
    var_estadia.append(data_split[9])
    var_volume.append(data_split[10])
    df_train.append(data_split[11])
    df_test.append(data_split[12])
    df_train_seco.append(data_split[13])
    df_train_chuvoso.append(data_split[14])

# Etapa 18: Tuning dos modelos de random forest e previsao
resultado = []
hp = {}

for i in range(len(PORTO)):
    modelo = Modelos(
        df[i],
        PORTO[i],
        var_demurrage[i],
        var_estadia[i],
        var_volume[i],
        pred_d[i],
        pred_t[i],
        pred_v[i],
        train_features[i],
        train_label[i],
        test_features[i],
        DATA_INICIO_TESTE,
        WD2,
        SAVE_RF,
        tuning_,
        hp_rf_prev[PORTO[i]],
    ).resultado()
    resultado.append(modelo[0])
    hp[PORTO[i]] = modelo[1]

df_demurrage_pm = resultado[0][0]
df_estadia_pm = resultado[0][1]
df_volume_pm = resultado[0][2]
df_demurrage_gs = resultado[1][0]
df_estadia_gs = resultado[1][1]
df_volume_gs = resultado[1][2]
df_demurrage_tb = resultado[2][0]
df_estadia_tb = resultado[2][1]
df_volume_tb = resultado[2][2]

hp_rf_prev = json.dumps(hp)
with open(os.getenv("HIPERPARAMETROS_RF_PREV"), "w", encoding="utf-8") as f:
    json.dump(hp_rf_prev, f, ensure_ascii=False, indent=4)

# Etapa 19: Na última etapa, vamos calcular os clusters

# df_detalhes_historico = pd.read_excel(os.getenv("CAMINHO_DADOS_HISTORICO_DETALHES"))

cluster = Clusterizacao(
    base_semanal.reset_index(),
    base_diaria.reset_index(),
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
    WD3,
    hp_rf_cluster,
    historico,
).resultado()

(
    df_mensal,
    df_day,
    df_melted_mensal,
    df_min_max,
    df_melted,
    df_final,
    df_final_ok,
    hp,
    historico,
) = (
    cluster[0],
    cluster[1],
    cluster[2],
    cluster[3],
    cluster[4],
    cluster[5],
    cluster[6],
    cluster[7],
    cluster[8],
)

hp_rf_cluster = json.dumps(hp)
with open(os.getenv("HIPERPARAMETROS_RF_CLUSTER"), "w", encoding="utf-8") as f:
    json.dump(hp_rf_cluster, f, ensure_ascii=False, indent=4)

# Vamos salvar o histórico de previsoes

historico.to_excel(os.getenv("HISTORICO_PREVISOES"))

# Vamos salvar as bases na pasta Output referente ao dia da estimacao

df_mensal.to_excel(WD3 + "/Output " + str(date.today()) + "/base_previsao_mensal.xlsx")
df_day.to_excel(WD3 + "/Output " + str(date.today()) + "/base_diaria.xlsx")
df_melted_mensal.to_excel(WD3 + "/Output " + str(date.today()) + "/df_variaveis.xlsx")
df_min_max.to_excel(
    WD3 + "/Output " + str(date.today()) + "/base_previsao_mensal_min_max.xlsx"
)
df_melted.to_excel(WD3 + "/Output " + str(date.today()) + "/correlacao_bi.xlsx")
df_final.to_excel(WD3 + "/Output " + str(date.today()) + "/shap_values.xlsx")
df_final_ok.to_excel(WD3 + "/Output " + str(date.today()) + "/cluster.xlsx")
semanal_prev.to_excel(WD3 + "/Output " + str(date.today()) + "/base_semanal.xlsx")

# Vamos copiar as bases da pasta do dia para a pasta com os dados do powerbi

files = os.listdir(WD3 + "/Output " + str(date.today()))
for fname in files:
    shutil.copy2(
        os.path.join(WD3 + "/Output " + str(date.today()), fname), WD3 + "/Output"
    )
