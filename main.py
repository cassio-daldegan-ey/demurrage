"""O script abaixo faz previsoes de Demurrage, Volume e Tempo de Estadia
"""

# Etapa 1: importacao das bibliotecas necessárias

import os
import json
import pandas as pd
from datetime import date
from dotenv import load_dotenv

# absolute_path é a pasta onde está o projeto
absolute_path = "C:/Users/KG858HY/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/02 - Previsão de Demurrage/04-Dados/3-Scripts/Nova estrutura de pastas/"

# O arquivo env concentra todas as variáveis de ambiente
load_dotenv(absolute_path + "env")

os.chdir(absolute_path + os.getenv("CAMINHO_SCRIPTS"))

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
    CorrecaoNomes,
    Semanalizacao,
    VariaveisDependentes,
    Datas_Inicio_Fim_Teste,
    MergeDemurrageOperationalDesk,
)

from tratamento_dados_pbi import Clusterizacao, tratamento_semanal

# Etapa 2: Definindo parametros

# Defina tuning_ = True se quiser fazer o tuning do modelo.
# Se quiser rodar com os hiperparametros da ultima estimacao,
# escolha tuing_ = False

tuning_ = False

# Lista de hiperparametros que utilizamos no prophet quando nao executamos o
# tuning do modelo.

with open(absolute_path + os.getenv("CAMINHO_DATA") + "hiperparametros.json") as file:
    hp_prophet = json.load(file)
    hp_prophet = json.loads(hp_prophet)

# Vamos importar os hiperparametros utilizados na última vez que fizemos o tuning
# dos modelos preditivos de random forest

with open(absolute_path + os.getenv("CAMINHO_DATA") + "hp_rf_prev.json") as file:
    hp_rf_prev = json.load(file)
    hp_rf_prev = json.loads(hp_rf_prev)

# Vamos importar os hiperparametros utilizados na última vez que fizemos o tuning
# dos modelos clusterizacao de random forest

with open(absolute_path + os.getenv("CAMINHO_DATA") + "hp_rf_cluster.json") as file:
    hp_rf_cluster = json.load(file)
    hp_rf_cluster = json.loads(hp_rf_cluster)

# Etapa 3: importacao de dados

# Nas linhas abaixo, carregamos uma classe que importa e junta as bases

bases_demurrage, bases_op, indices_PM, indices_S, indices_G, indices_T = Importacao(
    absolute_path + os.getenv("DEMURRAGE_FOLDER"),
    absolute_path + os.getenv("OP_FOLDER"),
    absolute_path + os.getenv("INDICADORES_PM_FOLDER"),
    absolute_path + os.getenv("INDICADORES_S_FOLDER"),
    absolute_path + os.getenv("INDICADORES_G_FOLDER"),
    absolute_path + os.getenv("INDICADORES_T_FOLDER"),
).dados()

dataset = pd.concat(bases_demurrage, axis=0, ignore_index=True)

operational_desk = pd.concat(bases_op, axis=0, ignore_index=True)

indices = []

for df_string in (
    (indices_PM, "X1."),
    (indices_PM, "X3N."),
    (indices_PM, "X3S."),
    (indices_PM, "X4N."),
    (indices_PM, "X4S."),
    (indices_S, "CNS1."),
    (indices_G, "CNG2."),
    (indices_T, "X02."),
    (indices_T, "X1N."),
    (indices_T, "X1S."),
):
    indices.append(CorrecaoNomes(df_string[0], df_string[1]))

(
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
) = indices

# A classe abaixo determina os períodos que utilizaremos como teste para o modelo
# identificando até qual data os dados estao disponíveis em cada base.

DATA_INICIO_TESTE, DATA_FIM_TESTE = Datas_Inicio_Fim_Teste(
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

# Etapa 4: Unir os dados das bases de Demurrage e Operational Desk

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
) = MergeDemurrageOperationalDesk(dataset, operational_desk).resultados()

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

base_g, base_s, base_pm, base_t, base_gs, embarcado_g, embarcado_s = Indicadores(
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

# Etapa 8: Vamos acrescentar a variável de custo de demurrage e volume

base_g, base_s, base_gs, base_pm, base_t, base_diaria = VariaveisDependentes(
    dataset_, base_g, base_s, base_gs, base_pm, base_t
).resultados()

# Etapa 10: Vamos transformar a base a nível diária para uma base de dados a nível semanal

base_semanal, base_diaria = Semanalizacao(base_diaria).base()

# Etapa 11: Fazemos previsao para o período futuro com base em médias.
# Tambem  acrescentamos valores de COI e CMA

base_diaria, base_semanal = Projecoes(
    base_diaria,
    base_semanal,
    data_final_real,
    data_inicial_real,
    data_inicio_projecoes,
    data_final_projecoes,
).result()

# Etapa 13: Vamos mudar os nomos das variáveis

base_diaria, base_semanal = NovosNomes(base_diaria, base_semanal).base()

# Etapa 14: Vamos exportar as bases diária e semanal

with open(absolute_path + os.getenv("CAMINHO_DATA") + "variaveis_diarias.json") as file:
    variaveis_diarias = json.load(file)
    variaveis_diarias = json.loads(variaveis_diarias)

with open(
    absolute_path + os.getenv("CAMINHO_DATA") + "variaveis_semanais.json"
) as file:
    variaveis_semanais = json.load(file)
    variaveis_semanais = json.loads(variaveis_semanais)

base_diaria = base_diaria[variaveis_diarias]

base_semanal = base_semanal[variaveis_semanais]

# Agora exportamos os dados

base_diaria.to_excel(
    absolute_path
    + os.getenv("CAMINHO_BASE_DIARIA")
    + date.today().strftime("%d%m%Y")
    + "-base_diaria.xlsx"
)

base_semanal.to_excel(
    absolute_path
    + os.getenv("CAMINHO_BASE_SEMANAL")
    + date.today().strftime("%d%m%Y")
    + "-base_semanal.xlsx"
)


# Etapa 15: Fazemos o tratamento de dados para previsao

df = base_semanal.reset_index()

semanal_prev = tratamento_semanal(base_semanal,data_final_real)
semanal_prev.to_excel(
    absolute_path + os.getenv("BASES_BI") + "/Output" + "/base_semanal.xlsx"
)

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
        df[i], PORTO[i], DATA_INICIO_TESTE, DATA_FIM_TESTE,
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
with open(
    absolute_path + os.getenv("CAMINHO_DATA") + "hiperparametros.json",
    "w",
    encoding="utf-8",
) as f:
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
    data_split = DataSplit(df[i], DATA_INICIO_TESTE, DATA_FIM_TESTE,).resultados()
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
        absolute_path,
        os.getenv("RANDOM_FOREST"),
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
with open(
    absolute_path + os.getenv("CAMINHO_DATA") + "hp_rf_prev.json", "w", encoding="utf-8"
) as f:
    json.dump(hp_rf_prev, f, ensure_ascii=False, indent=4)

# Etapa 19: Na última etapa, vamos calcular os clusters

historico = pd.read_excel(absolute_path + os.getenv("HISTORICO_PREVISOES"))

(
    df_mensal,
    df_day,
    df_melted_mensal,
    df_min_max,
    df_melted,
    df_final,
    df_final_ok,
    hp,
) = Clusterizacao(
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
    absolute_path,
    hp_rf_cluster,
    historico,
    os.getenv("HISTORICO_PREVISOES"),
    os.getenv("BASES_BI"),
).resultado()

hp_rf_cluster = json.dumps(hp)
with open(
    absolute_path + os.getenv("CAMINHO_DATA") + "hp_rf_cluster.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(hp_rf_cluster, f, ensure_ascii=False, indent=4)
