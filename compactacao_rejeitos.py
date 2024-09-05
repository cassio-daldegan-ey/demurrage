# Vamos tratar a base de dados de compactacao de rejeitos

import os
import math
import calendar
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Vamos definir os parametros.

USUARIO = "KG858HY"

####################### Etapa 1: Juntando todas as bases de dados

# Pasta onde estao os arquivos que vamos utilizar

ROOT = "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/2-Recebidos/1-Vale/2-Bases de Dados Vale/20230525-Bases Rejeitos"
PATTERN = "*.xlsx"

# Vamos salvar todos os enderecos onde estao os arquivos que vamos importar

filenames = []

for path, subdirs, files in os.walk(ROOT):
    for name in files:
        if fnmatch(name, PATTERN):
            filenames.append(os.path.join(path, name))

filenames = [sub.replace("\\", "/") for sub in filenames]

# Vamos definir as listas com os dataframes utilizados na base final,
# lista com os dataframes com colunas fora do padrao,
# lista comos enderecos das bases com colunas fora do padrao e
# lista com as bases que nao conseguimos importar

list_of_dfs = []
bases_colunas_erradas = []
enderecos_bases_colunas_erradas = []
erro_importacao = []
bases_corretas = []
importacao_correta = []

# No loop abaixo fazemos a iteracao entre todas as abas de todas as bases
# Utilizando as bases nas abas geramos a nossa base de dados final.
# Salvamos tambem os enderecos das bases que tem colunas fora do padrao
# ou que nao conseguimos importar.

colunas_corretas = [
    "Operação",
    "Data Inicío",
    "Arquivo de origem",
    "Baia",
    "Tempo gasto",
    "Obs",
    "Faixa",
    "Alerta",
    "Horário Inicial",
    "Empresa",
    "Retrabalho",
    "Data inicial",
    "Status",
    "Camada",
    "Operador",
    "Equipamento",
    "Data final",
    "Nome Operador",
    "Horário final",
    "Turno",
    "Número de Retrabalho",
]

for file in filenames:
    try:
        f = pd.ExcelFile(file)
        importacao_correta.append(file)
        try:
            for sheet in f.sheet_names:  # vamos extrair cada aba
                y = sheet
                if sheet != "Banco de Dados":
                    df = f.parse(sheet)
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df["Baia"] = sheet
                    df["Arquivo de origem"] = file
                    if len(df.columns) != 0:
                        df = df.loc[:, ~df.columns.duplicated()].copy()
                        df = df[
                            df[
                                [
                                    value
                                    for value in df.columns
                                    if value in colunas_corretas
                                ]
                            ].columns
                        ]
                        # vamos deletar os missings
                        df = df.loc[:, df.columns.notna()]
                        df = df.dropna(axis=1, how="all")
                        df = df.dropna(axis=0, how="all")
                        if "Status" in list(df.columns):
                            df = df[df["Status"] != "Sem Operação"]
                        if "Operação" in list(df.columns):
                            df = df[df["Operação"] != "Sem Operação"]
                        # abaixo retiramos colunas repetidas
                        if len(set(list(df.columns))) == len(list(df.columns)):
                            df = df[
                                list(
                                    set(list(df.columns))
                                    & set(colunas_corretas + ["Operação"])
                                )
                            ]
                            list_of_dfs.append(df)
                            bases_corretas.append(file)
        except:
            erro_importacao.append(file)
    except:
        erro_importacao.append(file)

# Vamos ficar apenas com os enderecos nao repetidos

erro_importacao = list(set(erro_importacao))
bases_corretas = list(set(bases_corretas))

len(list_of_dfs)

# Vamos concatenar as bases

df[df == "-"] = np.nan

df = pd.concat(list_of_dfs, axis=0, ignore_index=True)

# vamos corrigir algumas variáveis de data que estao em formato errado

df["Data inicial"] = np.where(
    df["Data inicial"] == np.nan, df["Data Inicío"], df["Data inicial"]
)
del df["Data Inicío"]
for variavel in ("Data final", "Data inicial"):
    datas_erradas = []
    for i in range(len(df)):
        if type(df[variavel].iloc[i]) == str:
            df[variavel].iloc[i] = df[variavel].iloc[i].strip()
            try:
                if int(df[variavel].iloc[i][3:]) == "jan":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        1,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "fev":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        2,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "mar":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        3,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "abr":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        4,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "mai":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        5,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "jun":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        6,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "jul":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        7,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "ago":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        8,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "set":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        9,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "out":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        10,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "nov":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        11,
                        int(df[variavel].iloc[i][:2]),
                    )
                elif int(df[variavel].iloc[i][3:]) == "dez":
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        12,
                        int(df[variavel].iloc[i][:2]),
                    )
                else:
                    df[variavel].iloc[i] = datetime.datetime(
                        int(df["Arquivo de origem"].iloc[i].split("/")[-3]),
                        int(df[variavel].iloc[i][3:]),
                        int(df[variavel].iloc[i][:2]),
                    )
            except:
                datas_erradas.append(i)
        elif type(df[variavel].iloc[i]) == datetime.time:
            datas_erradas.append(i)
        elif (
            type(df[variavel].iloc[i]) != datetime.datetime
            and math.isnan(df[variavel].iloc[i]) == False
        ):
            datas_erradas.append(i)
    df = df.drop(index=datas_erradas)
    df = df.reset_index(drop=True)

datas = []
for data in range(len(df)):
    datas.append(
        datetime.datetime(
            int(df["Arquivo de origem"].iloc[data].split("/")[-3]),
            int(df["Arquivo de origem"].iloc[data].split("/")[-2][0:2]),
            int(
                calendar.monthrange(
                    int(df["Arquivo de origem"].iloc[data].split("/")[-3]),
                    int(df["Arquivo de origem"].iloc[data].split("/")[-2][0:2]),
                )[1]
            ),
        )
    )

for var in ("Data final", "Data inicial"):
    df = df[((df[var] >= min(datas)) & (df[var] <= max(datas))) | (df[var] == np.nan)]

for variavel in ["Horário final", "Horário Inicial"]:
    horarios_errados = []
    df = df.reset_index(drop=True)
    for i in range(len(df[variavel])):
        if type(df[variavel].iloc[i]) == float or type(df[variavel].iloc[i]) == int:
            if math.isnan(df[variavel].iloc[i]) == False:
                horarios_errados.append(i)
        elif type(df[variavel].iloc[i]) != datetime.time:
            horarios_errados.append(i)
    df = df.drop(index=horarios_errados)

df = df.reset_index(drop=True)

for var in ["Operação", "Baia", "Operador", "Operação", "Equipamento"]:
    for i in range(len(df)):
        df = df.reset_index(drop=True)
        if (type(df[var].iloc[i]) != str) and (pd.isna(df[var].iloc[i]) == False):
            df[var].iloc[i] = "ERRO"

for var in ["Data final", "Data inicial"]:
    for i in range(len(df)):
        df = df.reset_index(drop=True)
        if (type(df[var].iloc[i]) != datetime.datetime) and (
            pd.isna(df[var].iloc[i]) == False
        ):
            df[var].iloc[i] = "ERRO"

for var in ["Horário Inicial", "Horário final"]:
    for i in range(len(df)):
        df = df.reset_index(drop=True)
        if (type(df["Horário Inicial"].iloc[i]) != datetime.time) and (
            pd.isna(df["Horário Inicial"].iloc[i]) == False
        ):
            df[var].iloc[i] = "ERRO"

for var in list(df.columns):
    df = df[df[var] != "ERRO"]

df["Duracao em horas"] = np.nan
for i in range(len(df)):
    if (
        (pd.isna(df["Horário final"].iloc[i]) == False)
        and (pd.isna(df["Horário Inicial"].iloc[i]) == False)
        and (pd.isna(df["Data inicial"].iloc[i]) == False)
        and (pd.isna(df["Data final"].iloc[i]) == False)
    ):
        if df["Data final"].iloc[i] == df["Data inicial"].iloc[i]:
            df["Duracao em horas"].iloc[i] = (
                float(df["Horário final"].iloc[i].hour)
                + float(df["Horário final"].iloc[i].minute / 60)
            ) - (
                float(df["Horário Inicial"].iloc[i].hour)
                + float(df["Horário Inicial"].iloc[i].minute / 60)
            )
        elif df["Data final"].iloc[i] > df["Data inicial"].iloc[i]:
            df["Duracao em horas"].iloc[i] = (
                (
                    (
                        float(df["Data final"].iloc[i].day)
                        - float(df["Data inicial"].iloc[i].day)
                        - 1
                    )
                    * 24
                )
                + 24
                - (
                    float(df["Horário Inicial"].iloc[i].hour)
                    + (float(df["Horário final"].iloc[i].minute)) / 60
                )
                + (
                    float(df["Horário final"].iloc[i].hour)
                    + (float(df["Horário final"].iloc[i].minute) / 60)
                )
            )

# Segue os links das bases que nao foram importadas por erros

bases_erros_importacao = pd.DataFrame(
    erro_importacao, columns=["Links dos arquivos com erros ao importar"]
)
bases_corretas = pd.DataFrame(
    bases_corretas, columns=["Links dos arquivos que foram corretamento importados"]
)
bases_erros_importacao.to_excel(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Bases/Bases com erros de importacao.xlsx"
)
bases_corretas.to_excel(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Bases/Bases corretas.xlsx"
)

# Segue os links das bases utilizadas no dataframe

bases_corretas = pd.DataFrame(
    bases_corretas, columns=["Links dos arquivos utilizados no dataframe"]
)
bases_corretas.to_excel(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Bases/Bases utilizadas no dataframe.xlsx"
)

# Vamos mudar os nomes de algumas categorias nas variáveis Equipamento, Operação
# e Baia.

df_ = df[(df["Status"] == "Em Operação") | (df["Status"] == "em Operação")][
    [
        "Baia",
        "Operação",
        "Operador",
        "Equipamento",
        "Status",
        "Duracao em horas",
        "Data inicial",
        "Data final",
        "Horário Inicial",
        "Horário final"
    ]
]

for nomes in (
    ["Trator Agrícola", "Trator"],
    ["Caminhão Pipa", "Caminhão"],
    ["Motoniveladora", "Motoniveladora"],
    ["Rolo Compactador", "Rolo"],
    ["Trator Esteira", "Trator"],
    ["Caminhão pipa", "Caminhão"],
    ["Trator de Pneu", "Trator"],
    ["Trator D10 4160", "Trator"],
    ["Trator 0454", "Trator"],
    ["Retroescaveira 0448", "Retroescavadeira"],
    ["Motoniveladora 1170", "Motoniveladora"],
    ["Rolo 0438", "Rolo"],
    ["Trator Agrícola 0525", "Trator"],
    ["Trator Agrícola 0528", "Trator"],
    ["Trator Agrícola 0517", "Trator"],
    ["Rolo 0447", "Rolo"],
    ["Motoniveladora 1168", "Motoniveladora"],
    ["Rolo 0446", "Rolo"],
    ["Motoniveladora 0556", "Motoniveladora"],
    ["Trator 0457", "Trator"],
    ["Pipa 0451", "Pipa"],
    ["Pipa 0468", "Pipa"],
    ["Rolo 0381", "Rolo"],
    ["Pipa 0442", "Pipa"],
    ["Trator 0450", "Trator"],
    ["Trator 60", "Trator"],
    ["Rolo 0429", "Rolo"],
    ["Rolo 1709", "Rolo"],
    ["Pá Carregadeira 6241", "Pá Carregadeira"],
    ["Trator 4710", "Trator"],
    ["Escavadeira 0514", "Escavadeira"],
    ["Pá Carregadeira 6242", "Pá Carregadeira"],
    ["Trator Agrícola 0526", "Trator"],
    ["Pipa 0443", "Pipa"],
    ["Pipa 0477", "Pipa"],
    ["Pipa 0441", "Pipa"],
    ["Motoniveladora 1169", "Motoniveladora"],
    ["motoniveladora 1169", "Motoniveladora"],
    ["motoniveladora 1170", "Motoniveladora"],
    ["ROLO 1709", "Rolo"],
    ["Trator 0451", "Trator"],
    ["Motoniveladora 0557", "Motoniveladora"],
    ["Pipa 0476", "Pipa"],
    ["Motoniveladora 1162", "Motoniveladora"],
    ["Motonivéladora 1169", "Motoniveladora"],
    ["ROLO 1708", "Rolo"],
    ["Rolo 1708", "Rolo"],
    ["ROLO 1710", "Rolo"],
    ["Trator D10 4161", "Trator"],
    ["Trator  0457", "Trator"],
    ["Rolo17 08", "Rolo"],
    ["TRATOR 62", "Trator"],
    ["trator 4162", "Trator"],
    ["Rolo 17 09", "Rolo"],
    ["Trator D10 4162", "Trator"],
    ["Trator 4162", "Trator"],
    ["Traor 4162", "Trator"],
    ["RC 1708", "Rolo"],
    ["RC 1709", "Rolo"],
    ["ROLO 17 09", "Rolo"],
    ["Motonivéladora  1169", "Motoniveladora"],
    ["Motonivéladora", "Motoniveladora"],
    ["Motonivéladora  11 69", "Motoniveladora"],
    ["Pipa 0470", "Pipa"],
    ["ROLO 17 08", "Rolo"],
    ["R0LO 1709", "Rolo"],
    ["Rolo 12t 0446", "Rolo"],
    ["Rolo 12t 0438", "Rolo"],
    ["Rolo 20t 1709", "Rolo"],
    ["Rolo 12t 0447", "Rolo"],
    ["Rolo 20t 1708", "Rolo"],
    ["Pipa 0469", "Pipa"],
    ["Trator D10 4163", "Trator"],
    ["trator Agrícola 0528", "Trator"],
    ["trator Agrícola 0517", "Trator"],
    ["Rolo 12t 0429", "Rolo"],
    ["Rolo 12t 0381", "Rolo"],
    ["Rolo 20t 1710", "Rolo"],
    ["Trator Agrícola 0519", "Trator"],
    ["Trator Agrícola 0527", "Trator"],
    ["Trator Agrícola 0518", "Trator"],
    ["Trator Agrícola 0530", "Trator"],
    ["Trator Agrícola 0529", "Trator"],
    ["Trator 0458", "Trator"],
    ["Rolo 12t 0439", "Rolo"],
    ["Pipa 0452", "Pipa"],
    ["Trator 4161", "Trator"],
    ["Retroescavadeira 0668", "Retroescavadeira"],
    ["Trator 4711", "Trator"],
    ["Trator 0455", "Trator"],
    ["Motoniveladora 1172", "Motoniveladora"],
    ["Trator D10 0461", "Trator"],
    ["Pipa 0448", "Pipa"],
    ["Motoniveladora 1171", "Motoniveladora"],
    ["Trator 4160", "Trator"],
    ["Motoniveladora 11 63", "Motoniveladora"],
    ["Trator Agrícola 0685", "Trator"],
    ["Rolo 12t 0445", "Rolo"],
    ["Trator Agrícola 0516", "Trator"],
    ["Trator Agrícola  0685", "Trator"],
    ["Trator Agricola 685", "Trator"],
    ["Pipa 0475", "Pipa"],
    ["trator agricola 0685", "Trator"],
    ["Trator agricola 0685", "Trator"],
    ["Trator Agricola 0685", "Trator"],
    ["Trator 0525", "Trator"],
    ["Trator 0528", "Trator"],
    ["Trator 0517", "Trator"],
    ["Trator 0526", "Trator"],
    ["Trator 0519", "Trator"],
    ["Trator 0527", "Trator"],
    ["Trator 0518", "Trator"],
    ["Trator 0530", "Trator"],
    ["Trator 0529", "Trator"],
    ["Trator 0685", "Trator"],
    ["Trator 0516", "Trator"],
    ["Trator  0685", "Trator"],
    ["Motoniveladora  11 69", "Motoniveladora"],
    ["Escavadeira", "Retroescavadeira"],
):
    df_["Equipamento"] = df_["Equipamento"].str.replace(nomes[0], nomes[1])

df_["Equipamento"] = np.where(
    (df_["Equipamento"] == "Não Aplica")
    | (df_["Equipamento"] == "Lucas Aparecido")
    | (df_["Equipamento"] == "MN1163")
    | (df_["Equipamento"] == "D10 4161")
    | (df_["Equipamento"] == "Filipe Henrique Martins")
    | (df_["Equipamento"] == "nA")
    | (df_["Equipamento"] == "TE4161"),
    np.nan,
    df_["Equipamento"],
)

for operacao in (
    ["Resultado do ensaio de Hilf", "Resultado do ensaio de Hilf"],
    ["Gradeamento", "Gradeamento"],
    ["Adição de água", "Adição de água"],
    ["Regularização", "Regularização"],
    ["Compactação", "Compactação"],
    ["Coleta de amostra para Hilf", "Coleta de amostra para Hilf"],
    ["Marcação Topográfica", "Marcação Topográfica"],
    ["Resultado do ensaio da Prévia", "Resultado do ensaio da Prévia"],
    ["Conferência de espessura", "Conferência de espessura"],
    ["Espalhamento", "Espalhamento"],
    ["Coleta de amostra para Prévia", "Coleta de amostra para Prévia"],
    ["Riscar a Camada", "Riscar camada"],
    ["Recebimento do Material", "Recebimento do Material"],
    ["Primitiva", "Primitiva"],
    ["Coleta deamostra para Hilf", "Coleta de amostra para Hilf"],
    ["Outro", "Outro"],
    ["Troca de Turno", "Outro"],
    ["Selagem", "Selagem"],
    ["Nivelamento", "Nivelamento"],
    ["Fechamento", "Fechamento"],
    ["coleta de amostra para Prévia", "Coleta de amostra para Prévia"],
    ["Camada aberta", "Camada aberta"],
    ["compactação", "Compactação"],
    ["Umidificação", "Umidificação"],
    ["marcação Topográfica", "Marcação Topográfica"],
    ["coleta de amostra para Hilf", "Coleta de amostra para Hilf"],
    ["Gradiamento", "Gradeamento"],
    ["riscar a Camada", "Riscar camada"],
    ["COLeta de amostra para Prévia", "Coleta de amostra para Prévia"],
    ["Coleta  amostra de Prévia", "Coleta de amostra para Prévia"],
    ["Coleta amostra de Hilf", "Coleta de amostra para Hilf"],
    ["regularização", "Regularização"],
    ["Coleta de amostra Hilf", "Coleta de amostra para Hilf"],
    ["Coleta de amostra Prévia", "Coleta de amostra para Prévia"],
    ["Resultado de Hilf", "Resultado do ensaio de Hilf"],
    ["Oscilação de Terrain/GPS", "Oscilação de Terrain/GPS"],
    ["Escarificação", "Escarificação"],
    ["Sem sinal GPS/Terrain", "Oscilação de Terrain/GPS"],
    ["Coleta amostra para Prévia", "Coleta de amostra para Prévia"],
    ["Coleta amostra para Hilf", "Coleta de amostra para Hilf"],
    ["Canto de Lâmina", "Canto de Lâmina"],
    ["Coleta amostra Prévia", "Coleta de amostra para Prévia"],
    ["Coleta amostra Hilf", "Coleta de amostra para Hilf"],
    ["Riscar camada", "Riscar camada"],
    ["Resultado ensaio de Hilf", "Resultado do ensaio de Hilf"],
    ["Riscando", "Riscar camada"],
    ["Riscando camada", "Riscar camada"],
    ["COMPactação", "Compactação"],
    ["umidificação", "Umidificação"],
    ["Riscando Camada", "Riscar camada"],
    ["recebimento do Material", "Recebimento do Material"],
    ["Riscar camada camada", "Riscar camada"],
    ["Riscar camada Camada", "Riscar camada"],
    ["Oscilação de Terrain/GPS", "Oscilação de Terrain ou GPS"],
):
    df_["Operação"] = df_["Operação"].str.replace(operacao[0], operacao[1])

df_["Operação"] = np.where(df_["Operação"] == "Em Operação", np.nan, df_["Operação"])

df_["Banco"] = df_["Baia"]
for nome in (
    ["Baia India", "1030"],
    ["Baia Juliet", "1030"],
    ["Baia Kilo", "1020"],
    ["Baia Lima", "1020"],
    ["Baia Mike", "1020"],
    ["Baia Quebec", "1020"],
    ["Baia Romeu", "1020"],
    ["Baia Alfa", "1030"],
    ["Baia Bravo", "1030"],
    ["Baia November", "1020"],
    ["Baia Papa ", "1020"],
    ["Faixa Transversal", "Outros"],
    ["Baia Charlie", "1030"],
    ["Baia Delta", "1030"],
    ["Baia Eco", "1030"],
    ["Baia Fox", "1030"],
    ["Baia Golf", "1030"],
    ["Baia Transversal", "Outros"],
    ["Baia kilo", "1020"],
    ["Baia Oscar", "1020"],
    ["Baia Papa", "1020"],
    ["Baia banco 1040", "1040"],
    ["Baia Cianita", "1040"],
    ["Baia Diamante", "1040"],
    ["Baia Esmeralda", "1040"],
    ["Baia Fosfato", "1040"],
    ["Baia Granada", "1040"],
    ["Baia Hematita", "1040"],
    ["Cianita", "1040"],
    ["Diamante", "1040"],
    ["Granada", "1040"],
    ["BAUXITA Banco 1040", "1040"],
    ["CIANITA Banco 1040", "1040"],
    ["DIAMANTE Banco 1040", "1040"],
    ["ESMERALDA Banco 1040", "1040"],
    ["HEMATITA Banco 1040 ", "1040"],
    ["FOSFATO Banco 1040", "1040"],
    ["GRANADA Banco 1040 ", "1040"],
    ["Hematita Banco 1040", "1040"],
    ["AGATA 1040 ", "1040"],
    ["BAUXITA 1040", "1040"],
    ["CIANITA 1040", "1040"],
    ["DIAMANTE 1040", "1040"],
    ["ESMERALDA 1040", "1040"],
    ["FOSFATO 1040", "1040"],
    ["GRANADA 1040", "1040"],
    ["HEMATITA 1040", "1040"],
    ["AGATA 1040", "1040"],
    ["GRANADA 1040 ", "1040"],
    ["HEMATITA 1040 ", "1040"],
    ["Baia Hemetita", "1040"],
    ["Hematita", "1040"],
    ["Baia Agata", "1040"],
    ["Baia Bauxita", "1040"],
    ["Baia November ", "1020"],
    ["Baia Whisky", "1020"],
    ["Sierra", "1020"],
    ["Victor", "1020"],
    ["Wisk", "1020"],
    ["Baia Victor ", "1020"],
    ["Baia Victor", "1020"],
    ["Alfa", "1030"],
    ["Bravo", "1030"],
    ["Charlie", "1030"],
    ["Delta", "1030"],
    ["Eco", "1030"],
    ["Fox", "1030"],
    ["Golf", "1030"],
    ["Hotel", "1030"],
    ["Baia Hotel", "1030"],
    ["India", "1030"],
    ["Juliet", "1030"],
    ["Baia Uniforme", "Outros"],
    ["Baia Uniforme ", "Outros"],
    ["Baia Tango", "1020"],
    ["Baia Wisky", "1020"],
    ["Baia Whsky", "1020"],
    ["Baia Itabirito", "1010"],
    ["Baia Tango ", "1020"],
    ["Baia Krypitonita", "1010"],
    ["Baia Limonita", "1010"],
    ["Baia Magnetita", "1010"],
    ["Baia Niobio", "1010"],
    ["Baia Opala", "1010"],
    ["Baia Itabirito ", "1010"],
    ["Baia Jade", "1010"],
    ["Baia Magnetita ", "1010"],
    ["Baia Pirita", "1010"],
    ["Baia Quartzo", "1010"],
    ["Baia 1030", "1030"],
    ["Outros ", "Outros"],
    ["1040 ", "1040"],
    ["1020 ", "1020"],
    ["Baia 1020 ", "1020"],
    ["Baia 1020", "1020"],
    ["Baia 1020y", "1010"],
    ["1020y", "1020"],
    ["1010 ", "1010"],
):
    df_["Banco"] = df_["Banco"].str.replace(nome[0], nome[1])

df_["Baia"] = df_["Baia"].apply(str.upper)

for nome in (
    ["BAIA ", ""],
    ["BANCO ", ""],
    ["  ", " "],
    [" 1040", ""],
    ["WISK", "WHISKY"],
    ["WISKY", "WHISKY"],
    ["WHSKY", "WHISKY"],
    ["WHSKYY", "WHISKY"],
    ["WHISKYY", "WHISKY"],
    ["HEMETITA", "HEMATITA"],
):
    df_["Baia"] = df_["Baia"].str.replace(nome[0], nome[1])

df_["Baia"] = df_["Baia"].str.rstrip()
df_["Baia"] = df_["Baia"].str.lstrip()

for var in ['Data inicial', 'Data final']:
    df_[var] = (
        pd.to_datetime(df_[var], dayfirst=True)
    ).dt.normalize()

df_['Ano-Semana'] = df_['Data inicial'].dt.strftime('%Y-%U')

# Terminado o tratamento da base, vamos exportar-la

df_.to_excel("C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Bases/Base final.xlsx")

########## ETAPA 2 - ESTATÍSTICAS DESCRITIVAS

# Vamos entao calcular os valores de soma e média de horas gastas por cada tipo
# de equipamento em cada dia. Primeiro os boxplots e depois os histogramas.

equipamento_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Equipamento"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)
equipamento_media = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Equipamento"])["Duracao em horas"]
    .agg("mean")
    .reset_index()
)

# Boxplots

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = equipamento_soma
df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Equipamento"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Equipamento")
ax = sns.boxplot(x=df_2["Equipamento"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Equipamento"])["Duracao em horas"].median().values
plt.xlabel("Equipamento", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title("Soma de tempo despendido diariamente por equipamento no dia", fontsize=20)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Equipamentos/Soma/"
    + "BoxPlot soma "
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = equipamento_media
df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Equipamento"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Equipamento")
ax = sns.boxplot(x=df_2["Equipamento"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Equipamento"])["Duracao em horas"].median().values
plt.xlabel("Equipamento", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title("Média de tempo despendido diariamente por equipamento no dia", fontsize=20)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Equipamentos/Média/"
    + "BoxPlot média "
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas

fig, axs = plt.subplots(4, 2, figsize=(8, 10))
df_2 = equipamento_soma
df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Equipamento"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Equipamento"] == "Trator"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Trator"}, inplace=True)
try:
    sns.histplot(data=base_1, x="Trator", kde=True, ax=axs[0, 0])
except:
    sns.histplot(data=base_1, x="Trator", kde=True, ax=axs[0, 0], bins="sturges")
base_2 = df_2[df_2["Equipamento"] == "Caminhão"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Caminhão"}, inplace=True)
try:
    sns.histplot(data=base_2, x="Caminhão", kde=True, ax=axs[0, 1])
except:
    sns.histplot(data=base_2, x="Caminhão", kde=True, ax=axs[0, 1], bins="sturges")
base_3 = df_2[df_2["Equipamento"] == "Motoniveladora"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Motoniveladora"}, inplace=True)
try:
    sns.histplot(data=base_3, x="Motoniveladora", kde=True, ax=axs[1, 0])
except:
    sns.histplot(
        data=base_3, x="Motoniveladora", kde=True, ax=axs[1, 0], bins="sturges"
    )
base_4 = df_2[df_2["Equipamento"] == "Rolo"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Rolo"}, inplace=True)
try:
    sns.histplot(data=base_4, x="Rolo", kde=True, ax=axs[1, 1])
except:
    sns.histplot(data=base_4, x="Rolo", kde=True, ax=axs[1, 1], bins="sturges")
base_5 = df_2[df_2["Equipamento"] == "Retroescavadeira"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Retroescavadeira"}, inplace=True)
try:
    sns.histplot(data=base_5, x="Retroescavadeira", kde=True, ax=axs[2, 0])
except:
    sns.histplot(
        data=base_5, x="Retroescavadeira", kde=True, ax=axs[2, 0], bins="sturges"
    )
base_6 = df_2[df_2["Equipamento"] == "Pipa"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Pipa"}, inplace=True)
try:
    sns.histplot(data=base_6, x="Pipa", kde=True, ax=axs[2, 1])
except:
    sns.histplot(data=base_6, x="Pipa", kde=True, ax=axs[2, 1], bins="sturges")
base_7 = df_2[df_2["Equipamento"] == "Pá Carregadeira"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "Pá Carregadeira"}, inplace=True)
try:
    sns.histplot(data=base_7, x="Pá Carregadeira", kde=True, ax=axs[3, 0])
except:
    sns.histplot(
        data=base_7, x="Pá Carregadeira", kde=True, ax=axs[3, 0], bins="sturges"
    )
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Equipamentos/Soma/"
    + "Histograma soma "
    + " Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(4, 2, figsize=(8, 10))
df_2 = equipamento_media
df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Equipamento"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Equipamento"] == "Trator"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Trator"}, inplace=True)
try:
    sns.histplot(data=base_1, x="Trator", kde=True, ax=axs[0, 0])
except:
    sns.histplot(data=base_1, x="Trator", kde=True, ax=axs[0, 0], bins="sturges")
base_2 = df_2[df_2["Equipamento"] == "Caminhão"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Caminhão"}, inplace=True)
try:
    sns.histplot(data=base_2, x="Caminhão", kde=True, ax=axs[0, 1])
except:
    sns.histplot(data=base_2, x="Caminhão", kde=True, ax=axs[0, 1], bins="sturges")
base_3 = df_2[df_2["Equipamento"] == "Motoniveladora"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Motoniveladora"}, inplace=True)
try:
    sns.histplot(data=base_3, x="Motoniveladora", kde=True, ax=axs[1, 0])
except:
    sns.histplot(
        data=base_3, x="Motoniveladora", kde=True, ax=axs[1, 0], bins="sturges"
    )
base_4 = df_2[df_2["Equipamento"] == "Rolo"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Rolo"}, inplace=True)
try:
    sns.histplot(data=base_4, x="Rolo", kde=True, ax=axs[1, 1])
except:
    sns.histplot(data=base_4, x="Rolo", kde=True, ax=axs[1, 1], bins="sturges")
base_5 = df_2[df_2["Equipamento"] == "Retroescavadeira"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Retroescavadeira"}, inplace=True)
try:
    sns.histplot(data=base_5, x="Retroescavadeira", kde=True, ax=axs[2, 0])
except:
    sns.histplot(
        data=base_5, x="Retroescavadeira", kde=True, ax=axs[2, 0], bins="sturges"
    )
base_6 = df_2[df_2["Equipamento"] == "Pipa"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Pipa"}, inplace=True)
try:
    sns.histplot(data=base_6, x="Pipa", kde=True, ax=axs[2, 1])
except:
    sns.histplot(data=base_6, x="Pipa", kde=True, ax=axs[2, 1], bins="sturges")
base_7 = df_2[df_2["Equipamento"] == "Pá Carregadeira"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "Pá Carregadeira"}, inplace=True)
try:
    sns.histplot(data=base_7, x="Pá Carregadeira", kde=True, ax=axs[3, 0])
except:
    sns.histplot(
        data=base_7, x="Pá Carregadeira", kde=True, ax=axs[3, 0], bins="sturges"
    )
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Equipamentos/Média/"
    + "Histograma média "
    + " Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Em sequencia, calculamos soma e média de nomero de horas gastas por todas as
# operacoes executadas em um mesmo dia em determinada baia. Como temos várias
# operacoes simultaneas, no caso da soma, os resultados serao superiores a 24
# Primeiro os boxplots e depois os histogramas

baia_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Baia"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)
baia_media = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Baia"])["Duracao em horas"]
    .agg("mean")
    .reset_index()
)

# Boxplots para soma

plt.figure(figsize=(30, 10))
sns.set(style="darkgrid")
df_2 = baia_soma[baia_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title("Total de horas despendido diariamente entre operação no dia", fontsize=20)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Soma/"
    + "BoxPlot soma "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas para soma

fig, axs = plt.subplots(7, 2, figsize=(8, 10))
df_2 = baia_soma[baia_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_alfa = df_2[df_2["Baia"] == "ALFA"][["Duracao em horas"]]
base_alfa.rename(columns={"Duracao em horas": "ALFA"}, inplace=True)
sns.histplot(data=base_alfa, x="ALFA", kde=True, ax=axs[0, 0])
base_faixa = df_2[df_2["Baia"] == "FAIXA TRANSVERSAL"][["Duracao em horas"]]
base_faixa.rename(columns={"Duracao em horas": "FAIXA TRANSVERSAL"}, inplace=True)
sns.histplot(data=base_faixa, x="FAIXA TRANSVERSAL", kde=True, ax=axs[0, 1])
base_mike = df_2[df_2["Baia"] == "MIKE"][["Duracao em horas"]]
base_mike.rename(columns={"Duracao em horas": "MIKE"}, inplace=True)
sns.histplot(data=base_mike, x="MIKE", kde=True, ax=axs[1, 0])
base_golf = df_2[df_2["Baia"] == "GOLF"][["Duracao em horas"]]
base_golf.rename(columns={"Duracao em horas": "GOLF"}, inplace=True)
sns.histplot(data=base_golf, x="GOLF", kde=True, ax=axs[1, 1])
base_charlie = df_2[df_2["Baia"] == "CHARLIE"][["Duracao em horas"]]
base_charlie.rename(columns={"Duracao em horas": "CHARLIE"}, inplace=True)
sns.histplot(data=base_charlie, x="CHARLIE", kde=True, ax=axs[2, 0])
base_bravo = df_2[df_2["Baia"] == "BRAVO"][["Duracao em horas"]]
base_bravo.rename(columns={"Duracao em horas": "BRAVO"}, inplace=True)
sns.histplot(data=base_bravo, x="BRAVO", kde=True, ax=axs[2, 1])
base_india = df_2[df_2["Baia"] == "INDIA"][["Duracao em horas"]]
base_india.rename(columns={"Duracao em horas": "INDIA"}, inplace=True)
sns.histplot(data=base_india, x="INDIA", kde=True, ax=axs[3, 0])
base_juliet = df_2[df_2["Baia"] == "JULIET"][["Duracao em horas"]]
base_juliet.rename(columns={"Duracao em horas": "JULIET"}, inplace=True)
sns.histplot(data=base_juliet, x="JULIET", kde=True, ax=axs[3, 1])
base_lima = df_2[df_2["Baia"] == "LIMA"][["Duracao em horas"]]
base_lima.rename(columns={"Duracao em horas": "LIMA"}, inplace=True)
sns.histplot(data=base_lima, x="LIMA", kde=True, ax=axs[4, 0])
base_quebec = df_2[df_2["Baia"] == "QUEBEC"][["Duracao em horas"]]
base_quebec.rename(columns={"Duracao em horas": "QUEBEC"}, inplace=True)
sns.histplot(data=base_quebec, x="QUEBEC", kde=True, ax=axs[4, 1])
base_eco = df_2[df_2["Baia"] == "ECO"][["Duracao em horas"]]
base_eco.rename(columns={"Duracao em horas": "ECO"}, inplace=True)
sns.histplot(data=base_eco, x="ECO", kde=True, ax=axs[5, 0])
base_fox = df_2[df_2["Baia"] == "FOX"][["Duracao em horas"]]
base_fox.rename(columns={"Duracao em horas": "FOX"}, inplace=True)
sns.histplot(data=base_fox, x="FOX", kde=True, ax=axs[5, 1])
base_hotel = df_2[df_2["Baia"] == "HOTEL"][["Duracao em horas"]]
base_hotel.rename(columns={"Duracao em horas": "HOTEL"}, inplace=True)
sns.histplot(data=base_hotel, x="HOTEL", kde=True, ax=axs[6, 0])
base_delta = df_2[df_2["Baia"] == "DELTA"][["Duracao em horas"]]
base_delta.rename(columns={"Duracao em horas": "DELTA"}, inplace=True)
sns.histplot(data=base_delta, x="DELTA", kde=True, ax=axs[6, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Soma/"
    + "Histograma soma parte 1 "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(6, 2, figsize=(8, 10))
df_2 = baia_soma[baia_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_romeu = df_2[df_2["Baia"] == "ROMEU"][["Duracao em horas"]]
base_romeu.rename(columns={"Duracao em horas": "ROMEU"}, inplace=True)
sns.histplot(data=base_romeu, x="ROMEU", kde=True, ax=axs[0, 0])
base_kilo = df_2[df_2["Baia"] == "KILO"][["Duracao em horas"]]
base_kilo.rename(columns={"Duracao em horas": "KILO"}, inplace=True)
sns.histplot(data=base_kilo, x="KILO", kde=True, ax=axs[0, 1])
base_november = df_2[df_2["Baia"] == "NOVEMBER"][["Duracao em horas"]]
base_november.rename(columns={"Duracao em horas": "NOVEMBER"}, inplace=True)
sns.histplot(data=base_november, x="NOVEMBER", kde=True, ax=axs[1, 0])
base_papa = df_2[df_2["Baia"] == "PAPA"][["Duracao em horas"]]
base_papa.rename(columns={"Duracao em horas": "PAPA"}, inplace=True)
sns.histplot(data=base_papa, x="PAPA", kde=True, ax=axs[1, 1])
base_oscar = df_2[df_2["Baia"] == "OSCAR"][["Duracao em horas"]]
base_oscar.rename(columns={"Duracao em horas": "OSCAR"}, inplace=True)
sns.histplot(data=base_oscar, x="OSCAR", kde=True, ax=axs[2, 0])
base_cianita = df_2[df_2["Baia"] == "CIANITA"][["Duracao em horas"]]
base_cianita.rename(columns={"Duracao em horas": "CIANITA"}, inplace=True)
sns.histplot(data=base_cianita, x="CIANITA", kde=True, ax=axs[2, 1])
base_diamante = df_2[df_2["Baia"] == "DIAMANTE"][["Duracao em horas"]]
base_diamante.rename(columns={"Duracao em horas": "DIAMANTE"}, inplace=True)
sns.histplot(data=base_diamante, x="DIAMANTE", kde=True, ax=axs[3, 0])
base_fosfato = df_2[df_2["Baia"] == "FOSFATO"][["Duracao em horas"]]
base_fosfato.rename(columns={"Duracao em horas": "FOSFATO"}, inplace=True)
sns.histplot(data=base_fosfato, x="FOSFATO", kde=True, ax=axs[3, 1])
base_hematita = df_2[df_2["Baia"] == "HEMATITA"][["Duracao em horas"]]
base_hematita.rename(columns={"Duracao em horas": "HEMATITA"}, inplace=True)
sns.histplot(data=base_hematita, x="HEMATITA", kde=True, ax=axs[4, 0])
base_granada = df_2[df_2["Baia"] == "GRANADA"][["Duracao em horas"]]
base_granada.rename(columns={"Duracao em horas": "GRANADA"}, inplace=True)
sns.histplot(data=base_granada, x="GRANADA", kde=True, ax=axs[4, 1])
base_esmeralda = df_2[df_2["Baia"] == "ESMERALDA"][["Duracao em horas"]]
base_esmeralda.rename(columns={"Duracao em horas": "ESMERALDA"}, inplace=True)
sns.histplot(data=base_esmeralda, x="ESMERALDA", kde=True, ax=axs[5, 0])
base_agata = df_2[df_2["Baia"] == "AGATA"][["Duracao em horas"]]
base_agata.rename(columns={"Duracao em horas": "AGATA"}, inplace=True)
sns.histplot(data=base_agata, x="AGATA", kde=True, ax=axs[5, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Soma/"
    + "Histograma soma parte 2 "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot para média

plt.figure(figsize=(30, 10))
sns.set(style="darkgrid")
df_2 = baia_media[baia_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title("Média de horas despendido diariamente entre operação no dia", fontsize=20)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Média/"
    + "BoxPlot média "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas para média

fig, axs = plt.subplots(7, 2, figsize=(8, 10))
df_2 = baia_media[baia_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_alfa = df_2[df_2["Baia"] == "ALFA"][["Duracao em horas"]]
base_alfa.rename(columns={"Duracao em horas": "ALFA"}, inplace=True)
sns.histplot(data=base_alfa, x="ALFA", kde=True, ax=axs[0, 0])
base_faixa = df_2[df_2["Baia"] == "FAIXA TRANSVERSAL"][["Duracao em horas"]]
base_faixa.rename(columns={"Duracao em horas": "FAIXA TRANSVERSAL"}, inplace=True)
sns.histplot(data=base_faixa, x="FAIXA TRANSVERSAL", kde=True, ax=axs[0, 1])
base_mike = df_2[df_2["Baia"] == "MIKE"][["Duracao em horas"]]
base_mike.rename(columns={"Duracao em horas": "MIKE"}, inplace=True)
sns.histplot(data=base_mike, x="MIKE", kde=True, ax=axs[1, 0])
base_golf = df_2[df_2["Baia"] == "GOLF"][["Duracao em horas"]]
base_golf.rename(columns={"Duracao em horas": "GOLF"}, inplace=True)
sns.histplot(data=base_golf, x="GOLF", kde=True, ax=axs[1, 1])
base_charlie = df_2[df_2["Baia"] == "CHARLIE"][["Duracao em horas"]]
base_charlie.rename(columns={"Duracao em horas": "CHARLIE"}, inplace=True)
sns.histplot(data=base_charlie, x="CHARLIE", kde=True, ax=axs[2, 0])
base_bravo = df_2[df_2["Baia"] == "BRAVO"][["Duracao em horas"]]
base_bravo.rename(columns={"Duracao em horas": "BRAVO"}, inplace=True)
sns.histplot(data=base_bravo, x="BRAVO", kde=True, ax=axs[2, 1])
base_india = df_2[df_2["Baia"] == "INDIA"][["Duracao em horas"]]
base_india.rename(columns={"Duracao em horas": "INDIA"}, inplace=True)
sns.histplot(data=base_india, x="INDIA", kde=True, ax=axs[3, 0])
base_juliet = df_2[df_2["Baia"] == "JULIET"][["Duracao em horas"]]
base_juliet.rename(columns={"Duracao em horas": "JULIET"}, inplace=True)
sns.histplot(data=base_juliet, x="JULIET", kde=True, ax=axs[3, 1])
base_lima = df_2[df_2["Baia"] == "LIMA"][["Duracao em horas"]]
base_lima.rename(columns={"Duracao em horas": "LIMA"}, inplace=True)
sns.histplot(data=base_lima, x="LIMA", kde=True, ax=axs[4, 0])
base_quebec = df_2[df_2["Baia"] == "QUEBEC"][["Duracao em horas"]]
base_quebec.rename(columns={"Duracao em horas": "QUEBEC"}, inplace=True)
sns.histplot(data=base_quebec, x="QUEBEC", kde=True, ax=axs[4, 1])
base_eco = df_2[df_2["Baia"] == "ECO"][["Duracao em horas"]]
base_eco.rename(columns={"Duracao em horas": "ECO"}, inplace=True)
sns.histplot(data=base_eco, x="ECO", kde=True, ax=axs[5, 0])
base_fox = df_2[df_2["Baia"] == "FOX"][["Duracao em horas"]]
base_fox.rename(columns={"Duracao em horas": "FOX"}, inplace=True)
sns.histplot(data=base_fox, x="FOX", kde=True, ax=axs[5, 1])
base_hotel = df_2[df_2["Baia"] == "HOTEL"][["Duracao em horas"]]
base_hotel.rename(columns={"Duracao em horas": "HOTEL"}, inplace=True)
sns.histplot(data=base_hotel, x="HOTEL", kde=True, ax=axs[6, 0])
base_delta = df_2[df_2["Baia"] == "DELTA"][["Duracao em horas"]]
base_delta.rename(columns={"Duracao em horas": "DELTA"}, inplace=True)
sns.histplot(data=base_delta, x="DELTA", kde=True, ax=axs[6, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Média/"
    + "Histograma media parte 1 "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(6, 2, figsize=(8, 10))
df_2 = baia_media[baia_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Baia"]
].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
sns.histplot(data=base_romeu, x="ROMEU", kde=True, ax=axs[0, 0])
base_kilo = df_2[df_2["Baia"] == "KILO"][["Duracao em horas"]]
base_kilo.rename(columns={"Duracao em horas": "KILO"}, inplace=True)
sns.histplot(data=base_kilo, x="KILO", kde=True, ax=axs[0, 1])
base_november = df_2[df_2["Baia"] == "NOVEMBER"][["Duracao em horas"]]
base_november.rename(columns={"Duracao em horas": "NOVEMBER"}, inplace=True)
sns.histplot(data=base_november, x="NOVEMBER", kde=True, ax=axs[1, 0])
base_papa = df_2[df_2["Baia"] == "PAPA"][["Duracao em horas"]]
base_papa.rename(columns={"Duracao em horas": "PAPA"}, inplace=True)
sns.histplot(data=base_papa, x="PAPA", kde=True, ax=axs[1, 1])
base_oscar = df_2[df_2["Baia"] == "OSCAR"][["Duracao em horas"]]
base_oscar.rename(columns={"Duracao em horas": "OSCAR"}, inplace=True)
sns.histplot(data=base_oscar, x="OSCAR", kde=True, ax=axs[2, 0])
base_cianita = df_2[df_2["Baia"] == "CIANITA"][["Duracao em horas"]]
base_cianita.rename(columns={"Duracao em horas": "CIANITA"}, inplace=True)
sns.histplot(data=base_cianita, x="CIANITA", kde=True, ax=axs[2, 1])
base_diamante = df_2[df_2["Baia"] == "DIAMANTE"][["Duracao em horas"]]
base_diamante.rename(columns={"Duracao em horas": "DIAMANTE"}, inplace=True)
sns.histplot(data=base_diamante, x="DIAMANTE", kde=True, ax=axs[3, 0])
base_fosfato = df_2[df_2["Baia"] == "FOSFATO"][["Duracao em horas"]]
base_fosfato.rename(columns={"Duracao em horas": "FOSFATO"}, inplace=True)
sns.histplot(data=base_fosfato, x="FOSFATO", kde=True, ax=axs[3, 1])
base_hematita = df_2[df_2["Baia"] == "HEMATITA"][["Duracao em horas"]]
base_hematita.rename(columns={"Duracao em horas": "HEMATITA"}, inplace=True)
sns.histplot(data=base_hematita, x="HEMATITA", kde=True, ax=axs[4, 0])
base_granada = df_2[df_2["Baia"] == "GRANADA"][["Duracao em horas"]]
base_granada.rename(columns={"Duracao em horas": "GRANADA"}, inplace=True)
sns.histplot(data=base_granada, x="GRANADA", kde=True, ax=axs[4, 1])
base_esmeralda = df_2[df_2["Baia"] == "ESMERALDA"][["Duracao em horas"]]
base_esmeralda.rename(columns={"Duracao em horas": "ESMERALDA"}, inplace=True)
sns.histplot(data=base_esmeralda, x="ESMERALDA", kde=True, ax=axs[5, 0])
base_agata = df_2[df_2["Baia"] == "AGATA"][["Duracao em horas"]]
base_agata.rename(columns={"Duracao em horas": "AGATA"}, inplace=True)
sns.histplot(data=base_agata, x="AGATA", kde=True, ax=axs[5, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Baia/Média/"
    + "Histograma media parte 2 "
    + "Baia"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Vou fazer um Pareto

operacao_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Operação"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)

operacao_soma = operacao_soma.sort_values(by="Duracao em horas", ascending=False)
operacao_soma['cumperc'] = operacao_soma["Duracao em horas"].cumsum()/operacao_soma["Duracao em horas"].sum()*100

color1 = 'steelblue'
color2 = 'red'
line_size = 4
fig, ax = plt.subplots()
ax.bar(operacao_soma["Operação"], operacao_soma["Duracao em horas"], color=color1)
ax2 = ax.twinx()
ax2.plot(operacao_soma["Operação"], operacao_soma['cumperc'], color=color2, marker="D", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis='y', colors=color1)
ax2.tick_params(axis='y', colors=color2)
plt.show()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Porcentagem de Tempo por Operacao/"
    + "Pareto" + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Para operacoes vamos fazer 3 tipos de gráficos. Para além dos de soma e média
# de horas despendidas em cada operacao, faremos também o de porcentagem de horas
# totais gastas em cada tipo de operacao. Considerando como horas totas a soma de todas
# as horas despendidas no dia.

operacao = df_[
    (df_["Data inicial"] == df_["Data final"])
    & (df_["Duracao em horas"] > 0)
    & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
]
operacao["Data inicial"] = operacao["Data inicial"].astype(str)
operacao["Data final"] = operacao["Data final"].astype(str)
operacao = (
    operacao[
        (operacao["Data inicial"] == operacao["Data final"])
        & (operacao["Duracao em horas"] > 0)
        & (operacao["Duracao em horas"] < operacao["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Baia", "Operação"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)
tempo_total = df_[
    (df_["Data inicial"] == df_["Data final"])
    & (df_["Duracao em horas"] > 0)
    & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
]
tempo_total["Data inicial"] = tempo_total["Data inicial"].astype(str)
tempo_total["Data final"] = tempo_total["Data final"].astype(str)
tempo_total = (
    tempo_total[
        (tempo_total["Data inicial"] == tempo_total["Data final"])
        & (tempo_total["Duracao em horas"] > 0)
        & (
            tempo_total["Duracao em horas"]
            < tempo_total["Duracao em horas"].quantile(0.99)
        )
    ]
    .groupby(["Data inicial", "Baia"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)
tempo_total.rename(columns={"Duracao em horas": "Total por dia"}, inplace=True)
operacao = pd.merge(operacao, tempo_total, how="left", on=["Data inicial", "Baia"])
operacao["Duracao em horas"] = operacao["Duracao em horas"] / operacao["Total por dia"]

# Boxplot de porcentagem

plt.figure(figsize=(30, 10))
sns.set(style="darkgrid")
df_2 = operacao[operacao["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Operação")
count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
count["Operação novo"] = np.nan
for i in range(len(count)):
    count["Operação novo"].iloc[i] = (
        count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Operação"] = df_2["Operação"].str.replace(
        list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
plt.xlabel("Operação", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Porcentagem", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Porcentagem de tempo despendido diariamente por operação por baia no dia",
    fontsize=20,
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Porcentagem de Tempo por Operacao/"
    + "BoxPlot "
    + "Operação"
    + " porcentagem de tempo"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas de porcentagem

fig, axs = plt.subplots(6, 2, figsize=(8, 10))
df_2 = operacao[operacao["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True)
sns.histplot(data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True)
sns.histplot(data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][["Duracao em horas"]]
base_7.rename(
    columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
)
sns.histplot(data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "Conferência de espessura"}, inplace=True)
sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
    ["Duracao em horas"]
]
base_10.rename(
    columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
)
sns.histplot(data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1])
base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
base_12.rename(columns={"Duracao em horas": "Recebimento do Material"}, inplace=True)
sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Porcentagem de Tempo por Operacao/"
    + "Histograma parte 1 "
    + "Operação"
    + " porcentagem de tempo"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = operacao[operacao["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
sns.histplot(data=base_1, x="Primitiva", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
sns.histplot(data=base_2, x="Outro", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
sns.histplot(data=base_3, x="Selagem", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
sns.histplot(data=base_4, x="Nivelamento", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
sns.histplot(data=base_5, x="Fechamento", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
sns.histplot(data=base_6, x="Camada aberta", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Operação"] == "Umidificação"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "Umidificação"}, inplace=True)
sns.histplot(data=base_7, x="Umidificação", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True)
sns.histplot(data=base_8, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
sns.histplot(data=base_9, x="Escarificação", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
sns.histplot(data=base_10, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Porcentagem de Tempo por Operacao/"
    + "Histograma parte 2 "
    + "Operação"
    + " porcentagem de tempo"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de soma

operacao = df_[
    (df_["Data inicial"] == df_["Data final"])
    & (df_["Duracao em horas"] > 0)
    & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
]
operacao_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Operação"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)
operacao_media = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Operação"])["Duracao em horas"]
    .agg("mean")
    .reset_index()
)

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Operação")
count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
count["Operação novo"] = np.nan
for i in range(len(count)):
    count["Operação novo"].iloc[i] = (
        count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Operação"] = df_2["Operação"].str.replace(
        list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
plt.xlabel("Operação", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Soma de tempo despendido diariamente por operação por baia no dia", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Soma/"
    + "BoxPlot soma "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de soma

fig, axs = plt.subplots(6, 2, figsize=(8, 10))
df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True)
sns.histplot(data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True)
sns.histplot(data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][["Duracao em horas"]]
base_7.rename(
    columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
)
sns.histplot(data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "Conferência de espessura"}, inplace=True)
sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
    ["Duracao em horas"]
]
base_10.rename(
    columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
)
sns.histplot(data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1])
base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
base_12.rename(columns={"Duracao em horas": "Recebimento do Material"}, inplace=True)
sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Soma/"
    + "Histograma soma parte 1 "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][["Duracao em horas"]]
base_19.rename(
    columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
)
sns.histplot(data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0])
base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
base_20.rename(columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True)
sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Soma/"
    + "Histograma soma parte 2 "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de média

operacao_media = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Operação"])["Duracao em horas"]
    .agg("mean")
    .reset_index()
)

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Operação")
count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
count["Operação novo"] = np.nan
for i in range(len(count)):
    count["Operação novo"].iloc[i] = (
        count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Operação"] = df_2["Operação"].str.replace(
        list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
plt.xlabel("Operação", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Média de tempo despendido diariamente por operação por baia no dia", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Média/"
    + "BoxPlot média "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de média

fig, axs = plt.subplots(6, 2, figsize=(8, 10))
df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True)
sns.histplot(data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True)
sns.histplot(data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][["Duracao em horas"]]
base_7.rename(
    columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
)
sns.histplot(data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "Conferência de espessura"}, inplace=True)
sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
    ["Duracao em horas"]
]
base_10.rename(
    columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
)
sns.histplot(data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1])
base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
base_12.rename(columns={"Duracao em horas": "Recebimento do Material"}, inplace=True)
sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Média/"
    + "Histograma média parte 1 "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
    ["Duracao em horas", "Operação"]
].dropna()
df_2["Operação"] = np.where(
    (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
    "Umidificação/Adição de água",
    df_2["Operação"],
)
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][["Duracao em horas"]]
base_19.rename(
    columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
)
sns.histplot(data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0])
base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
base_20.rename(columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True)
sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Por Operacao/Média/"
    + "Histograma média parte 2 "
    + "Operação"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Vamos fazer agora os gráficos de soma e média de horas gastas nas operacoes diárias
# por baia, sem fazer distincao do tipo de operacao.

for baia in [
    "HOTEL",
    "INDIA",
    "JULIET",
    "KILO",
    "LIMA",
    "MIKE",
    "QUEBEC",
    "ROMEU",
    "ALFA",
    "BRAVO",
    "NOVEMBER",
    "PAPA",
    "FAIXA TRANSVERSAL",
    "CHARLIE",
    "DELTA",
    "ECO",
    "FOX",
    "GOLF",
    "OSCAR",
    "CIANITA",
    "DIAMANTE",
    "ESMERALDA",
    "FOSFATO",
    "GRANADA",
    "HEMATITA",
    "BAUXITA",
    "AGATA",
    "WHISKY",
    "VICTOR",
    "UNIFORME",
    "TANGO",
]:
    operacao_soma = (
        df_[
            (df_["Data inicial"] == df_["Data final"])
            & (df_["Duracao em horas"] > 0)
            & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
        ]
        .groupby(["Data inicial", "Operação", "Baia"])["Duracao em horas"]
        .agg("sum")
        .reset_index()
    )
    operacao_soma = operacao_soma[operacao_soma["Baia"] == baia]
    operacao_media = (
        df_[
            (df_["Data inicial"] == df_["Data final"])
            & (df_["Duracao em horas"] > 0)
            & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
        ]
        .groupby(["Data inicial", "Operação", "Baia"])["Duracao em horas"]
        .agg("mean")
        .reset_index()
    )
    operacao_media = operacao_media[operacao_media["Baia"] == baia]

    # Boxplot de soma

    plt.figure(figsize=(35, 10))
    sns.set(style="darkgrid")
    df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    df_2 = df_2.sort_values("Operação")
    count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
    count["Operação novo"] = np.nan
    for i in range(len(count)):
        count["Operação novo"].iloc[i] = (
            count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
        )
    for i in range(len(count)):
        df_2["Operação"] = df_2["Operação"].str.replace(
            list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
        )
    categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
    df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
    ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
    medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
    plt.xlabel("Operação", fontsize=20)
    plt.xticks(rotation="vertical", fontsize=20)
    plt.ylabel("Horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Soma de tempo despendido diariamente por operação, baia " + baia, fontsize=20
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Soma/"
        + "BoxPlot soma "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Histogramas de soma

    fig, axs = plt.subplots(6, 2, figsize=(8, 10))
    df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][
        ["Duracao em horas"]
    ]
    base_1.rename(
        columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0]
        )
    except:
        sns.histplot(
            data=base_1,
            x="Resultado do ensaio de Hilf",
            kde=True,
            ax=axs[0, 0],
            bins="sturges",
        )
    base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
    base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
    try:
        sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(
            data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1], bins="sturges"
        )
    base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
    base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
    try:
        sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(
            data=base_3, x="Regularização", kde=True, ax=axs[1, 0], bins="sturges"
        )
    base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
    base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
    try:
        sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_4, x="Compactação", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][
        ["Duracao em horas"]
    ]
    base_5.rename(
        columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0]
        )
    except:
        sns.histplot(
            data=base_5,
            x="Coleta de amostra para Hilf",
            kde=True,
            ax=axs[2, 0],
            bins="sturges",
        )
    base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
    base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
    try:
        sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_6,
            x="Marcação Topográfica",
            kde=True,
            ax=axs[2, 1],
            bins="sturges",
        )
    base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][
        ["Duracao em horas"]
    ]
    base_7.rename(
        columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_7,
            x="Resultado do ensaio da Prévia",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
    base_8.rename(
        columns={"Duracao em horas": "Conferência de espessura"}, inplace=True
    )
    try:
        sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_8,
            x="Conferência de espessura",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
    base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
    try:
        sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
        ["Duracao em horas"]
    ]
    base_10.rename(
        columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1]
        )
    except:
        sns.histplot(
            data=base_10,
            x="Coleta de amostra para Prévia",
            kde=True,
            ax=axs[4, 1],
            bins="sturges",
        )
    base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
    base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
    try:
        sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
    except:
        sns.histplot(
            data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0], bins="sturges"
        )
    base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
    base_12.rename(
        columns={"Duracao em horas": "Recebimento do Material"}, inplace=True
    )
    try:
        sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
    except:
        sns.histplot(
            data=base_12,
            x="Recebimento do Material",
            kde=True,
            ax=axs[5, 1],
            bins="sturges",
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Soma/"
        + "Histograma soma parte 1 "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(5, 2, figsize=(8, 10))
    df_2 = operacao_soma[operacao_soma["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
    base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
    try:
        sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
    except:
        sns.histplot(
            data=base_13, x="Primitiva", kde=True, ax=axs[0, 0], bins="sturges"
        )
    base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
    base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
    try:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1], bins="sturges")
    base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
    base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
    try:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0], bins="sturges")
    base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
    base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
    try:
        sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
    base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
    try:
        sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
    except:
        sns.histplot(
            data=base_17, x="Fechamento", kde=True, ax=axs[2, 0], bins="sturges"
        )
    base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
    base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
    try:
        sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1], bins="sturges"
        )
    base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][
        ["Duracao em horas"]
    ]
    base_19.rename(
        columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_19,
            x="Umidificação/Adição de água",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
    base_20.rename(
        columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True
    )
    try:
        sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_20,
            x="Oscilação de Terrain/GPS",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
    base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
    try:
        sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_21, x="Escarificação", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
    base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
    try:
        sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
    except:
        sns.histplot(
            data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1], bins="sturges"
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Soma/"
        + "Histograma soma parte 2 "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Boxplot de média

    plt.figure(figsize=(35, 10))
    sns.set(style="darkgrid")
    df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    df_2 = df_2.sort_values("Operação")
    count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
    count["Operação novo"] = np.nan
    for i in range(len(count)):
        count["Operação novo"].iloc[i] = (
            count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
        )
    for i in range(len(count)):
        df_2["Operação"] = df_2["Operação"].str.replace(
            list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
        )
    categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
    df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
    ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
    medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
    plt.xlabel("Operação", fontsize=20)
    plt.xticks(rotation="vertical", fontsize=20)
    plt.ylabel("Horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Média de tempo despendido diariamente por operação, baia " + baia, fontsize=20
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Média/"
        + "BoxPlot média "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Histogramas de média

    fig, axs = plt.subplots(6, 2, figsize=(8, 10))
    df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][
        ["Duracao em horas"]
    ]
    base_1.rename(
        columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0]
        )
    except:
        sns.histplot(
            data=base_1,
            x="Resultado do ensaio de Hilf",
            kde=True,
            ax=axs[0, 0],
            bins="sturges",
        )
    base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
    base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
    try:
        sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(
            data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1], bins="sturges"
        )
    base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
    base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
    try:
        sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(
            data=base_3, x="Regularização", kde=True, ax=axs[1, 0], bins="sturges"
        )
    base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
    base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
    try:
        sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_4, x="Compactação", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][
        ["Duracao em horas"]
    ]
    base_5.rename(
        columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0]
        )
    except:
        sns.histplot(
            data=base_5,
            x="Coleta de amostra para Hilf",
            kde=True,
            ax=axs[2, 0],
            bins="sturges",
        )
    base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
    base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
    try:
        sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_6,
            x="Marcação Topográfica",
            kde=True,
            ax=axs[2, 1],
            bins="sturges",
        )
    base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][
        ["Duracao em horas"]
    ]
    base_7.rename(
        columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_7,
            x="Resultado do ensaio da Prévia",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
    base_8.rename(
        columns={"Duracao em horas": "Conferência de espessura"}, inplace=True
    )
    try:
        sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_8,
            x="Conferência de espessura",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
    base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
    try:
        sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
        ["Duracao em horas"]
    ]
    base_10.rename(
        columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1]
        )
    except:
        sns.histplot(
            data=base_10,
            x="Coleta de amostra para Prévia",
            kde=True,
            ax=axs[4, 1],
            bins="sturges",
        )
    base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
    base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
    try:
        sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
    except:
        sns.histplot(
            data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0], bins="sturges"
        )
    base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
    base_12.rename(
        columns={"Duracao em horas": "Recebimento do Material"}, inplace=True
    )
    try:
        sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
    except:
        sns.histplot(
            data=base_12,
            x="Recebimento do Material",
            kde=True,
            ax=axs[5, 1],
            bins="sturges",
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Média/"
        + "Histograma média parte 1 "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(5, 2, figsize=(8, 10))
    df_2 = operacao_media[operacao_media["Duracao em horas"] > 0][
        ["Duracao em horas", "Operação"]
    ].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
    base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
    try:
        sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
    except:
        sns.histplot(
            data=base_13, x="Primitiva", kde=True, ax=axs[0, 0], bins="sturges"
        )
    base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
    base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
    try:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1], bins="sturges")
    base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
    base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
    try:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0], bins="sturges")
    base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
    base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
    try:
        sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
    base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
    try:
        sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
    except:
        sns.histplot(
            data=base_17, x="Fechamento", kde=True, ax=axs[2, 0], bins="sturges"
        )
    base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
    base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
    try:
        sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1], bins="sturges"
        )
    base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][
        ["Duracao em horas"]
    ]
    base_19.rename(
        columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_19,
            x="Umidificação/Adição de água",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
    base_20.rename(
        columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True
    )
    try:
        sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_20,
            x="Oscilação de Terrain/GPS",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
    base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
    try:
        sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_21, x="Escarificação", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
    base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
    try:
        sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
    except:
        sns.histplot(
            data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1], bins="sturges"
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Baia/Média/"
        + "Histograma média parte 2 "
        + baia
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# Vamos gerar esses mesmos resultados de soma e médias por bancos

banco_media = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Banco", "Baia", "Operação"])["Duracao em horas"]
    .agg("mean")
    .reset_index()
)
banco_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Data inicial", "Banco", "Baia", "Operação"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)

# Vamos fazer as mesmas análises de bosxplot e histograma das operacoes agora
# considerando o banco como unidade de análise

for nome in ["1010", "1020", "1030", "1040"]:
    plt.figure(figsize=(35, 10))
    sns.set(style="darkgrid")
    df_2 = banco_soma[banco_soma["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    df_2 = df_2.sort_values("Operação")
    count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
    count["Operação novo"] = np.nan
    for i in range(len(count)):
        count["Operação novo"].iloc[i] = (
            count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
        )
    for i in range(len(count)):
        df_2["Operação"] = df_2["Operação"].str.replace(
            list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
        )
    categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
    df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
    ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
    medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
    plt.xlabel("Operação", fontsize=20)
    plt.xticks(rotation="vertical", fontsize=20)
    plt.ylabel("Horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Soma de tempo despendido diariamente por operação por baia no dia, banco "
        + nome,
        fontsize=20,
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Soma/"
        + "BoxPlot soma "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(6, 2, figsize=(8, 10))
    df_2 = banco_soma[banco_soma["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][
        ["Duracao em horas"]
    ]
    base_1.rename(
        columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0]
        )
    except:
        sns.histplot(
            data=base_1,
            x="Resultado do ensaio de Hilf",
            kde=True,
            ax=axs[0, 0],
            bins="sturges",
        )
    base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
    base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
    try:
        sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(
            data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1], bins="sturges"
        )
    base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
    base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
    try:
        sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(
            data=base_3, x="Regularização", kde=True, ax=axs[1, 0], bins="sturges"
        )
    base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
    base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
    try:
        sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_4, x="Compactação", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][
        ["Duracao em horas"]
    ]
    base_5.rename(
        columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0]
        )
    except:
        sns.histplot(
            data=base_5,
            x="Coleta de amostra para Hilf",
            kde=True,
            ax=axs[2, 0],
            bins="sturges",
        )
    base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
    base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
    try:
        sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_6,
            x="Marcação Topográfica",
            kde=True,
            ax=axs[2, 1],
            bins="sturges",
        )
    base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][
        ["Duracao em horas"]
    ]
    base_7.rename(
        columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_7,
            x="Resultado do ensaio da Prévia",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
    base_8.rename(
        columns={"Duracao em horas": "Conferência de espessura"}, inplace=True
    )
    try:
        sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_8,
            x="Conferência de espessura",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
    base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
    try:
        sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
        ["Duracao em horas"]
    ]
    base_10.rename(
        columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1]
        )
    except:
        sns.histplot(
            data=base_10,
            x="Coleta de amostra para Prévia",
            kde=True,
            ax=axs[4, 1],
            bins="sturges",
        )
    base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
    base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
    try:
        sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
    except:
        sns.histplot(
            data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0], bins="sturges"
        )
    base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
    base_12.rename(
        columns={"Duracao em horas": "Recebimento do Material"}, inplace=True
    )
    try:
        sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
    except:
        sns.histplot(
            data=base_12,
            x="Recebimento do Material",
            kde=True,
            ax=axs[5, 1],
            bins="sturges",
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Soma/"
        + "Histograma soma parte 1 "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(5, 2, figsize=(8, 10))
    df_2 = banco_soma[banco_soma["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
    base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
    try:
        sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
    except:
        sns.histplot(
            data=base_13, x="Primitiva", kde=True, ax=axs[0, 0], bins="sturges"
        )
    base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
    base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
    try:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1], bins="sturges")
    base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
    base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
    try:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0], bins="sturges")
    base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
    base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
    try:
        sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
    base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
    try:
        sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
    except:
        sns.histplot(
            data=base_17, x="Fechamento", kde=True, ax=axs[2, 0], bins="sturges"
        )
    base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
    base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
    try:
        sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1], bins="sturges"
        )
    base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][
        ["Duracao em horas"]
    ]
    base_19.rename(
        columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_19,
            x="Umidificação/Adição de água",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
    base_20.rename(
        columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True
    )
    try:
        sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_20,
            x="Oscilação de Terrain/GPS",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
    base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
    try:
        sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_21, x="Escarificação", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
    base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
    try:
        sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
    except:
        sns.histplot(
            data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1], bins="sturges"
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Soma/"
        + "Histograma soma parte 2 "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Agora, a média

    plt.figure(figsize=(35, 10))
    sns.set(style="darkgrid")
    df_2 = banco_media[banco_media["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    df_2 = df_2.sort_values("Operação")
    count = df_2.groupby(["Operação"])["Duracao em horas"].agg(["count"]).reset_index()
    count["Operação novo"] = np.nan
    for i in range(len(count)):
        count["Operação novo"].iloc[i] = (
            count["Operação"].iloc[i] + " - " + str(count["count"].iloc[i])
        )
    for i in range(len(count)):
        df_2["Operação"] = df_2["Operação"].str.replace(
            list(df_2["Operação"].unique())[i], list(count["Operação novo"].unique())[i]
        )
    categorias_utilizadas = list(count[count["count"] >= 5]["Operação novo"].unique())
    df_2 = df_2[df_2["Operação"].isin(categorias_utilizadas)]
    ax = sns.boxplot(x=df_2["Operação"], y=df_2["Duracao em horas"], showfliers=False)
    medians = df_2.groupby(["Operação"])["Duracao em horas"].median().values
    plt.xlabel("Operação", fontsize=20)
    plt.xticks(rotation="vertical", fontsize=20)
    plt.ylabel("Horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Média de tempo despendido diariamente por operação no dia, baua " + nome,
        fontsize=20,
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Média/"
        + "BoxPlot média "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(6, 2, figsize=(8, 10))
    df_2 = banco_media[banco_media["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_1 = df_2[df_2["Operação"] == "Resultado do ensaio de Hilf"][
        ["Duracao em horas"]
    ]
    base_1.rename(
        columns={"Duracao em horas": "Resultado do ensaio de Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_1, x="Resultado do ensaio de Hilf", kde=True, ax=axs[0, 0]
        )
    except:
        sns.histplot(
            data=base_1,
            x="Resultado do ensaio de Hilf",
            kde=True,
            ax=axs[0, 0],
            bins="sturges",
        )
    base_2 = df_2[df_2["Operação"] == "Gradeamento"][["Duracao em horas"]]
    base_2.rename(columns={"Duracao em horas": "Gradeamento"}, inplace=True)
    try:
        sns.histplot(data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(
            data=base_2, x="Gradeamento", kde=True, ax=axs[0, 1], bins="sturges"
        )
    base_3 = df_2[df_2["Operação"] == "Regularização"][["Duracao em horas"]]
    base_3.rename(columns={"Duracao em horas": "Regularização"}, inplace=True)
    try:
        sns.histplot(data=base_3, x="Regularização", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(
            data=base_3, x="Regularização", kde=True, ax=axs[1, 0], bins="sturges"
        )
    base_4 = df_2[df_2["Operação"] == "Compactação"][["Duracao em horas"]]
    base_4.rename(columns={"Duracao em horas": "Compactação"}, inplace=True)
    try:
        sns.histplot(data=base_4, x="Compactação", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_4, x="Compactação", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_5 = df_2[df_2["Operação"] == "Coleta de amostra para Hilf"][
        ["Duracao em horas"]
    ]
    base_5.rename(
        columns={"Duracao em horas": "Coleta de amostra para Hilf"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_5, x="Coleta de amostra para Hilf", kde=True, ax=axs[2, 0]
        )
    except:
        sns.histplot(
            data=base_5,
            x="Coleta de amostra para Hilf",
            kde=True,
            ax=axs[2, 0],
            bins="sturges",
        )
    base_6 = df_2[df_2["Operação"] == "Marcação Topográfica"][["Duracao em horas"]]
    base_6.rename(columns={"Duracao em horas": "Marcação Topográfica"}, inplace=True)
    try:
        sns.histplot(data=base_6, x="Marcação Topográfica", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_6,
            x="Marcação Topográfica",
            kde=True,
            ax=axs[2, 1],
            bins="sturges",
        )
    base_7 = df_2[df_2["Operação"] == "Resultado do ensaio da Prévia"][
        ["Duracao em horas"]
    ]
    base_7.rename(
        columns={"Duracao em horas": "Resultado do ensaio da Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_7, x="Resultado do ensaio da Prévia", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_7,
            x="Resultado do ensaio da Prévia",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_8 = df_2[df_2["Operação"] == "Conferência de espessura"][["Duracao em horas"]]
    base_8.rename(
        columns={"Duracao em horas": "Conferência de espessura"}, inplace=True
    )
    try:
        sns.histplot(data=base_8, x="Conferência de espessura", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_8,
            x="Conferência de espessura",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_9 = df_2[df_2["Operação"] == "Espalhamento"][["Duracao em horas"]]
    base_9.rename(columns={"Duracao em horas": "Espalhamento"}, inplace=True)
    try:
        sns.histplot(data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_9, x="Espalhamento", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_10 = df_2[df_2["Operação"] == "Coleta de amostra para Prévia"][
        ["Duracao em horas"]
    ]
    base_10.rename(
        columns={"Duracao em horas": "Coleta de amostra para Prévia"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_10, x="Coleta de amostra para Prévia", kde=True, ax=axs[4, 1]
        )
    except:
        sns.histplot(
            data=base_10,
            x="Coleta de amostra para Prévia",
            kde=True,
            ax=axs[4, 1],
            bins="sturges",
        )
    base_11 = df_2[df_2["Operação"] == "Riscar camada"][["Duracao em horas"]]
    base_11.rename(columns={"Duracao em horas": "Riscar camada"}, inplace=True)
    try:
        sns.histplot(data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0])
    except:
        sns.histplot(
            data=base_11, x="Riscar camada", kde=True, ax=axs[5, 0], bins="sturges"
        )
    base_12 = df_2[df_2["Operação"] == "Recebimento do Material"][["Duracao em horas"]]
    base_12.rename(
        columns={"Duracao em horas": "Recebimento do Material"}, inplace=True
    )
    try:
        sns.histplot(data=base_12, x="Recebimento do Material", kde=True, ax=axs[5, 1])
    except:
        sns.histplot(
            data=base_12,
            x="Recebimento do Material",
            kde=True,
            ax=axs[5, 1],
            bins="sturges",
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Média/"
        + "Histograma média parte 1 "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, axs = plt.subplots(5, 2, figsize=(8, 10))
    df_2 = banco_media[banco_media["Banco"] == nome]
    df_2 = df_2[df_2["Duracao em horas"] > 0][["Duracao em horas", "Operação"]].dropna()
    df_2["Operação"] = np.where(
        (df_2["Operação"] == "Adição de água") | (df_2["Operação"] == "Umidificação"),
        "Umidificação/Adição de água",
        df_2["Operação"],
    )
    df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
    base_13 = df_2[df_2["Operação"] == "Primitiva"][["Duracao em horas"]]
    base_13.rename(columns={"Duracao em horas": "Primitiva"}, inplace=True)
    try:
        sns.histplot(data=base_13, x="Primitiva", kde=True, ax=axs[0, 0])
    except:
        sns.histplot(
            data=base_13, x="Primitiva", kde=True, ax=axs[0, 0], bins="sturges"
        )
    base_14 = df_2[df_2["Operação"] == "Outro"][["Duracao em horas"]]
    base_14.rename(columns={"Duracao em horas": "Outro"}, inplace=True)
    try:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1])
    except:
        sns.histplot(data=base_14, x="Outro", kde=True, ax=axs[0, 1], bins="sturges")
    base_15 = df_2[df_2["Operação"] == "Selagem"][["Duracao em horas"]]
    base_15.rename(columns={"Duracao em horas": "Selagem"}, inplace=True)
    try:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0])
    except:
        sns.histplot(data=base_15, x="Selagem", kde=True, ax=axs[1, 0], bins="sturges")
    base_16 = df_2[df_2["Operação"] == "Nivelamento"][["Duracao em horas"]]
    base_16.rename(columns={"Duracao em horas": "Nivelamento"}, inplace=True)
    try:
        sns.histplot(data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1])
    except:
        sns.histplot(
            data=base_16, x="Nivelamento", kde=True, ax=axs[1, 1], bins="sturges"
        )
    base_17 = df_2[df_2["Operação"] == "Fechamento"][["Duracao em horas"]]
    base_17.rename(columns={"Duracao em horas": "Fechamento"}, inplace=True)
    try:
        sns.histplot(data=base_17, x="Fechamento", kde=True, ax=axs[2, 0])
    except:
        sns.histplot(
            data=base_17, x="Fechamento", kde=True, ax=axs[2, 0], bins="sturges"
        )
    base_18 = df_2[df_2["Operação"] == "Camada aberta"][["Duracao em horas"]]
    base_18.rename(columns={"Duracao em horas": "Camada aberta"}, inplace=True)
    try:
        sns.histplot(data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1])
    except:
        sns.histplot(
            data=base_18, x="Camada aberta", kde=True, ax=axs[2, 1], bins="sturges"
        )
    base_19 = df_2[df_2["Operação"] == "Umidificação/Adição de água"][
        ["Duracao em horas"]
    ]
    base_19.rename(
        columns={"Duracao em horas": "Umidificação/Adição de água"}, inplace=True
    )
    try:
        sns.histplot(
            data=base_19, x="Umidificação/Adição de água", kde=True, ax=axs[3, 0]
        )
    except:
        sns.histplot(
            data=base_19,
            x="Umidificação/Adição de água",
            kde=True,
            ax=axs[3, 0],
            bins="sturges",
        )
    base_20 = df_2[df_2["Operação"] == "Oscilação de Terrain/GPS"][["Duracao em horas"]]
    base_20.rename(
        columns={"Duracao em horas": "Oscilação de Terrain/GPS"}, inplace=True
    )
    try:
        sns.histplot(data=base_20, x="Oscilação de Terrain/GPS", kde=True, ax=axs[3, 1])
    except:
        sns.histplot(
            data=base_20,
            x="Oscilação de Terrain/GPS",
            kde=True,
            ax=axs[3, 1],
            bins="sturges",
        )
    base_21 = df_2[df_2["Operação"] == "Escarificação"][["Duracao em horas"]]
    base_21.rename(columns={"Duracao em horas": "Escarificação"}, inplace=True)
    try:
        sns.histplot(data=base_21, x="Escarificação", kde=True, ax=axs[4, 0])
    except:
        sns.histplot(
            data=base_21, x="Escarificação", kde=True, ax=axs[4, 0], bins="sturges"
        )
    base_22 = df_2[df_2["Operação"] == "Canto de Lâmina"][["Duracao em horas"]]
    base_22.rename(columns={"Duracao em horas": "Canto de Lâmina"}, inplace=True)
    try:
        sns.histplot(data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1])
    except:
        sns.histplot(
            data=base_22, x="Canto de Lâmina", kde=True, ax=axs[4, 1], bins="sturges"
        )
    fig.tight_layout()
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco/Média/"
        + "Histograma média parte 2 "
        + nome
        + " Operação"
        + " Duracao em horas"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# Em seguida faremos a análise de soma e média de tempo dispoendido entre operacoes
# para cada banco separadamente, explicitando as baias que fazem parte de cada banco
# Lembrando que nas análises anteriores considerávamos apenas banco ou apenas
# baia, agora vamos considerar os dois

# Boxplot de médias para o Banco 1010

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1010")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Média de tempo despendido diariamente por baia no banco " + "1010", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "BoxPlot média "
    + "1010"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas de médias para o Banco 1010

fig, axs = plt.subplots(4, 2, figsize=(8, 10))
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1010")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "KRYPITONITA"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "KRYPITONITA"}, inplace=True)
sns.histplot(data=base_1, x="KRYPITONITA", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "LIMONITA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "LIMONITA"}, inplace=True)
sns.histplot(data=base_2, x="LIMONITA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "NIOBIO"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "NIOBIO"}, inplace=True)
sns.histplot(data=base_3, x="NIOBIO", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "OPALA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "OPALA"}, inplace=True)
sns.histplot(data=base_4, x="OPALA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "ITABIRITO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "ITABIRITO"}, inplace=True)
sns.histplot(data=base_5, x="ITABIRITO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "QUARTZO"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "QUARTZO"}, inplace=True)
sns.histplot(data=base_6, x="QUARTZO", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "JADE"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "JADE"}, inplace=True)
sns.histplot(data=base_7, x="JADE", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "MAGNETITA"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "MAGNETITA"}, inplace=True)
sns.histplot(data=base_8, x="MAGNETITA", kde=True, ax=axs[3, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "Histograma média "
    + "Banco "
    + "1010"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de soma para o Banco 1010

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1010")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Soma de tempo despendido diariamente por baia no banco " + "1010", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "BoxPlot soma "
    + "1010"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de soma para o Banco 1010

fig, axs = plt.subplots(4, 2, figsize=(8, 10))
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1010")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "KRYPITONITA"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "KRYPITONITA"}, inplace=True)
sns.histplot(data=base_1, x="KRYPITONITA", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "LIMONITA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "LIMONITA"}, inplace=True)
sns.histplot(data=base_2, x="LIMONITA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "NIOBIO"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "NIOBIO"}, inplace=True)
sns.histplot(data=base_3, x="NIOBIO", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "OPALA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "OPALA"}, inplace=True)
sns.histplot(data=base_4, x="OPALA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "ITABIRITO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "ITABIRITO"}, inplace=True)
sns.histplot(data=base_5, x="ITABIRITO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "QUARTZO"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "QUARTZO"}, inplace=True)
sns.histplot(data=base_6, x="QUARTZO", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "JADE"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "JADE"}, inplace=True)
sns.histplot(data=base_7, x="JADE", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "MAGNETITA"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "MAGNETITA"}, inplace=True)
sns.histplot(data=base_8, x="MAGNETITA", kde=True, ax=axs[3, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "Histograma soma "
    + "Banco "
    + "1010"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de médias para o Banco 1020

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1020")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Média de tempo despendido diariamente por baia no banco " + "1020", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "BoxPlot média "
    + "1020"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas de médias para o Banco 1020

fig, axs = plt.subplots(7, 2, figsize=(8, 10))
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1020")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "KILO"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "KILO"}, inplace=True)
sns.histplot(data=base_1, x="KILO", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "LIMA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "LIMA"}, inplace=True)
sns.histplot(data=base_2, x="LIMA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "MIKE"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "MIKE"}, inplace=True)
sns.histplot(data=base_3, x="MIKE", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "QUEBEC"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "QUEBEC"}, inplace=True)
sns.histplot(data=base_4, x="QUEBEC", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "ROMEU"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "ROMEU"}, inplace=True)
sns.histplot(data=base_5, x="ROMEU", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "NOVEMBER"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "NOVEMBER"}, inplace=True)
sns.histplot(data=base_6, x="NOVEMBER", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "PAPA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "PAPA"}, inplace=True)
sns.histplot(data=base_7, x="PAPA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "OSCAR"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "OSCAR"}, inplace=True)
sns.histplot(data=base_8, x="OSCAR", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "WHISKY"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "WHISKY"}, inplace=True)
sns.histplot(data=base_9, x="WHISKY", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "SIERRA"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "SIERRA"}, inplace=True)
sns.histplot(data=base_10, x="SIERRA", kde=True, ax=axs[4, 1])
base_11 = df_2[df_2["Baia"] == "VICTOR"][["Duracao em horas"]]
base_11.rename(columns={"Duracao em horas": "VICTOR"}, inplace=True)
sns.histplot(data=base_11, x="VICTOR", kde=True, ax=axs[5, 0])
base_12 = df_2[df_2["Baia"] == "WISK"][["Duracao em horas"]]
base_12.rename(columns={"Duracao em horas": "WISK"}, inplace=True)
sns.histplot(data=base_12, x="WISK", kde=True, ax=axs[5, 1])
base_13 = df_2[df_2["Baia"] == "TANGO"][["Duracao em horas"]]
base_13.rename(columns={"Duracao em horas": "TANGO"}, inplace=True)
sns.histplot(data=base_13, x="TANGO", kde=True, ax=axs[6, 0])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "Histograma média "
    + "Banco "
    + "1020"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de soma para o Banco 1020

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1020")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Soma de tempo despendido diariamente por baia no banco " + "1020", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "BoxPlot soma "
    + "1020"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histogramas de soma para o Banco 1020

fig, axs = plt.subplots(7, 2, figsize=(8, 10))
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1020")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "KILO"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "KILO"}, inplace=True)
sns.histplot(data=base_1, x="KILO", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "LIMA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "LIMA"}, inplace=True)
sns.histplot(data=base_2, x="LIMA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "MIKE"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "MIKE"}, inplace=True)
sns.histplot(data=base_3, x="MIKE", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "QUEBEC"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "QUEBEC"}, inplace=True)
sns.histplot(data=base_4, x="QUEBEC", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "ROMEU"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "ROMEU"}, inplace=True)
sns.histplot(data=base_5, x="ROMEU", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "NOVEMBER"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "NOVEMBER"}, inplace=True)
sns.histplot(data=base_6, x="NOVEMBER", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "PAPA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "PAPA"}, inplace=True)
sns.histplot(data=base_7, x="PAPA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "OSCAR"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "OSCAR"}, inplace=True)
sns.histplot(data=base_8, x="OSCAR", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "WHISKY"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "WHISKY"}, inplace=True)
sns.histplot(data=base_9, x="WHISKY", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "SIERRA"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "SIERRA"}, inplace=True)
sns.histplot(data=base_10, x="SIERRA", kde=True, ax=axs[4, 1])
base_11 = df_2[df_2["Baia"] == "VICTOR"][["Duracao em horas"]]
base_11.rename(columns={"Duracao em horas": "VICTOR"}, inplace=True)
sns.histplot(data=base_11, x="VICTOR", kde=True, ax=axs[5, 0])
base_12 = df_2[df_2["Baia"] == "WISK"][["Duracao em horas"]]
base_12.rename(columns={"Duracao em horas": "WISK"}, inplace=True)
sns.histplot(data=base_12, x="WISK", kde=True, ax=axs[5, 1])
base_13 = df_2[df_2["Baia"] == "TANGO"][["Duracao em horas"]]
base_13.rename(columns={"Duracao em horas": "TANGO"}, inplace=True)
sns.histplot(data=base_13, x="TANGO", kde=True, ax=axs[6, 0])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "Histograma soma "
    + "Banco "
    + "1020"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de médias para o Banco 1030

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1030")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Média de tempo despendido diariamente por baia no banco " + "1030", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "BoxPlot media "
    + "1030"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de médias para o Banco 1030

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1030")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "HOTEL"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "HOTEL"}, inplace=True)
sns.histplot(data=base_1, x="HOTEL", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "INDIA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "INDIA"}, inplace=True)
sns.histplot(data=base_2, x="INDIA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "JULIET"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "JULIET"}, inplace=True)
sns.histplot(data=base_3, x="JULIET", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "ALFA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "ALFA"}, inplace=True)
sns.histplot(data=base_4, x="ALFA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "BRAVO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "BRAVO"}, inplace=True)
sns.histplot(data=base_5, x="BRAVO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "CHARLIE"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "CHARLIE"}, inplace=True)
sns.histplot(data=base_6, x="CHARLIE", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "DELTA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "DELTA"}, inplace=True)
sns.histplot(data=base_7, x="DELTA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "ECO"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "ECO"}, inplace=True)
sns.histplot(data=base_8, x="ECO", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "GOLF"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "GOLF"}, inplace=True)
sns.histplot(data=base_9, x="GOLF", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "FOX"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "FOX"}, inplace=True)
sns.histplot(data=base_10, x="FOX", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "Histograma média "
    + "Banco "
    + "1030"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de soma para o Banco 1030

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1030")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Soma de tempo despendido diariamente por baia no banco " + "1030", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "BoxPlot soma "
    + "1030"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de soma para o Banco 1030

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1030")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "HOTEL"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "HOTEL"}, inplace=True)
sns.histplot(data=base_1, x="HOTEL", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "INDIA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "INDIA"}, inplace=True)
sns.histplot(data=base_2, x="INDIA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "JULIET"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "JULIET"}, inplace=True)
sns.histplot(data=base_3, x="JULIET", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "ALFA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "ALFA"}, inplace=True)
sns.histplot(data=base_4, x="ALFA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "BRAVO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "BRAVO"}, inplace=True)
sns.histplot(data=base_5, x="BRAVO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "CHARLIE"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "CHARLIE"}, inplace=True)
sns.histplot(data=base_6, x="CHARLIE", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "DELTA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "DELTA"}, inplace=True)
sns.histplot(data=base_7, x="DELTA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "ECO"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "ECO"}, inplace=True)
sns.histplot(data=base_8, x="ECO", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "GOLF"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "GOLF"}, inplace=True)
sns.histplot(data=base_9, x="GOLF", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "FOX"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "FOX"}, inplace=True)
sns.histplot(data=base_10, x="FOX", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "Histograma soma "
    + "Banco "
    + "1030"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de médias para o Banco 1040

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1040")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Média de tempo despendido diariamente por baia no banco " + "1040", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "BoxPlot média "
    + "1040"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de médias para o Banco 1040

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = banco_media[
    (banco_media["Duracao em horas"] > 0) & (banco_media["Banco"] == "1040")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "1040"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "1040"}, inplace=True)
sns.histplot(data=base_1, x="1040", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "CIANITA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "CIANITA"}, inplace=True)
sns.histplot(data=base_2, x="CIANITA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "DIAMANTE"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "DIAMANTE"}, inplace=True)
sns.histplot(data=base_3, x="DIAMANTE", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "ESMERALDA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "ESMERALDA"}, inplace=True)
sns.histplot(data=base_4, x="ESMERALDA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "FOSFATO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "FOSFATO"}, inplace=True)
sns.histplot(data=base_5, x="FOSFATO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "GRANADA"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "GRANADA"}, inplace=True)
sns.histplot(data=base_6, x="GRANADA", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "HEMATITA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "HEMATITA"}, inplace=True)
sns.histplot(data=base_7, x="HEMATITA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "BAUXITA"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "BAUXITA"}, inplace=True)
sns.histplot(data=base_8, x="BAUXITA", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "AGATA"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "AGATA"}, inplace=True)
sns.histplot(data=base_9, x="AGATA", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "HEMETITA"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "HEMETITA"}, inplace=True)
sns.histplot(data=base_10, x="HEMETITA", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Média/"
    + "Histograma média "
    + "Banco "
    + "1040"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Boxplot de soma para o Banco 1040

plt.figure(figsize=(35, 10))
sns.set(style="darkgrid")
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1040")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
df_2 = df_2.sort_values("Baia")
count = df_2.groupby(["Baia"])["Duracao em horas"].agg(["count"]).reset_index()
count["Baia novo"] = np.nan
for i in range(len(count)):
    count["Baia novo"].iloc[i] = (
        count["Baia"].iloc[i] + " - " + str(count["count"].iloc[i])
    )
for i in range(len(count)):
    df_2["Baia"] = df_2["Baia"].str.replace(
        list(df_2["Baia"].unique())[i], list(count["Baia novo"].unique())[i]
    )
categorias_utilizadas = list(count[count["count"] >= 5]["Baia novo"].unique())
df_2 = df_2[df_2["Baia"].isin(categorias_utilizadas)]
ax = sns.boxplot(x=df_2["Baia"], y=df_2["Duracao em horas"], showfliers=False)
medians = df_2.groupby(["Baia"])["Duracao em horas"].median().values
plt.xlabel("Baia", fontsize=20)
plt.xticks(rotation="vertical", fontsize=20)
plt.ylabel("Horas", fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "Soma de tempo despendido diariamente por baia no banco " + "1040", fontsize=20
)
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "BoxPlot soma "
    + "1040"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# Histograma de soma para o Banco 1040

fig, axs = plt.subplots(5, 2, figsize=(8, 10))
df_2 = banco_soma[
    (banco_soma["Duracao em horas"] > 0) & (banco_soma["Banco"] == "1040")
][["Duracao em horas", "Banco", "Baia"]].dropna()
df_2 = df_2[df_2["Duracao em horas"] < df_2["Duracao em horas"].quantile(0.99)]
base_1 = df_2[df_2["Baia"] == "1040"][["Duracao em horas"]]
base_1.rename(columns={"Duracao em horas": "1040"}, inplace=True)
sns.histplot(data=base_1, x="1040", kde=True, ax=axs[0, 0])
base_2 = df_2[df_2["Baia"] == "CIANITA"][["Duracao em horas"]]
base_2.rename(columns={"Duracao em horas": "CIANITA"}, inplace=True)
sns.histplot(data=base_2, x="CIANITA", kde=True, ax=axs[0, 1])
base_3 = df_2[df_2["Baia"] == "DIAMANTE"][["Duracao em horas"]]
base_3.rename(columns={"Duracao em horas": "DIAMANTE"}, inplace=True)
sns.histplot(data=base_3, x="DIAMANTE", kde=True, ax=axs[1, 0])
base_4 = df_2[df_2["Baia"] == "ESMERALDA"][["Duracao em horas"]]
base_4.rename(columns={"Duracao em horas": "ESMERALDA"}, inplace=True)
sns.histplot(data=base_4, x="ESMERALDA", kde=True, ax=axs[1, 1])
base_5 = df_2[df_2["Baia"] == "FOSFATO"][["Duracao em horas"]]
base_5.rename(columns={"Duracao em horas": "FOSFATO"}, inplace=True)
sns.histplot(data=base_5, x="FOSFATO", kde=True, ax=axs[2, 0])
base_6 = df_2[df_2["Baia"] == "GRANADA"][["Duracao em horas"]]
base_6.rename(columns={"Duracao em horas": "GRANADA"}, inplace=True)
sns.histplot(data=base_6, x="GRANADA", kde=True, ax=axs[2, 1])
base_7 = df_2[df_2["Baia"] == "HEMATITA"][["Duracao em horas"]]
base_7.rename(columns={"Duracao em horas": "HEMATITA"}, inplace=True)
sns.histplot(data=base_7, x="HEMATITA", kde=True, ax=axs[3, 0])
base_8 = df_2[df_2["Baia"] == "BAUXITA"][["Duracao em horas"]]
base_8.rename(columns={"Duracao em horas": "BAUXITA"}, inplace=True)
sns.histplot(data=base_8, x="BAUXITA", kde=True, ax=axs[3, 1])
base_9 = df_2[df_2["Baia"] == "AGATA"][["Duracao em horas"]]
base_9.rename(columns={"Duracao em horas": "AGATA"}, inplace=True)
sns.histplot(data=base_9, x="AGATA", kde=True, ax=axs[4, 0])
base_10 = df_2[df_2["Baia"] == "HEMETITA"][["Duracao em horas"]]
base_10.rename(columns={"Duracao em horas": "HEMETITA"}, inplace=True)
sns.histplot(data=base_10, x="HEMETITA", kde=True, ax=axs[4, 1])
fig.tight_layout()
plt.savefig(
    "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Operacoes por Banco e Baia/Soma/"
    + "Histograma soma "
    + "Banco "
    + "1040"
    + " Duracao em horas"
    + ".png",
    bbox_inches="tight",
)
plt.show()
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()


# Por fim, vamos fazer um gráfico de barras emplilhadas. Mas Vamos considerar
# apenas as operacoes que concentram ao menos 80% do total de horas gastas

# Para selecionar esses dados precisamos gerar um gráfico de pareto

operacao_soma = (
    df_[
        (df_["Data inicial"] == df_["Data final"])
        & (df_["Duracao em horas"] > 0)
        & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
    ]
    .groupby(["Operação"])["Duracao em horas"]
    .agg("sum")
    .reset_index()
)

operacao_soma = operacao_soma.sort_values(by="Duracao em horas", ascending=False)
operacao_soma['cumperc'] = operacao_soma["Duracao em horas"].cumsum()/operacao_soma["Duracao em horas"].sum()*100

color1 = 'steelblue'
color2 = 'red'
line_size = 4
fig, ax = plt.subplots()
ax.bar(operacao_soma["Operação"], operacao_soma["Duracao em horas"], color=color1)
ax2 = ax.twinx()
ax2.plot(operacao_soma["Operação"], operacao_soma['cumperc'], color=color2, marker="D", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis='y', colors=color1)
ax2.tick_params(axis='y', colors=color2)
plt.show()

# Vamos filtrar apenas as operacoes que, no acumulado, representam 80% das horas

outros = ['Umidificação','Primitiva','Marcação Topográfica','Outro','Coleta de amostra para Hilf','Coleta de amostra para Prévia','Selagem','Fechamento','Adição de água','Riscar camada','Resultado do ensaio da Prévia','Resultado do ensaio de Hilf','Escarificação','Conferência de espessura','Oscilação de Terrain ou GPS','Canto de Lâmina','Camada aberta']

for BAIA in ['HOTEL','INDIA','JULIET','KILO','LIMA','MIKE','QUEBEC','ROMEU','ALFA','BRAVO','NOVEMBER','PAPA','FAIXA TRANSVERSAL','CHARLIE','DELTA','ECO','FOX','GOLF','OSCAR','CIANITA','DIAMANTE','ESMERALDA','FOSFATO','GRANADA','HEMATITA','BAUXITA','AGATA']:
    x = BAIA
    # Vamos fazer o para as 6 operacoes mais importantes e colocar as demais em outros
    
    operacao_soma_semanal = (
        df_[
            (df_["Data inicial"] == df_["Data final"])
            & (df_["Duracao em horas"] > 0)
            & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
        ]
        .groupby(["Ano-Semana",'Baia',"Operação"])["Duracao em horas"]
        .agg("sum")
        .reset_index()
    )

    for op in outros:
        operacao_soma_semanal["Operação"] = np.where(operacao_soma_semanal["Operação"] == op,"Outro",operacao_soma_semanal["Operação"])

    operacao_soma_semanal = (
        operacao_soma_semanal.groupby(["Ano-Semana",'Baia',"Operação"])["Duracao em horas"]
        .agg("sum")
        .reset_index()
    )
    df_2 = operacao_soma_semanal[operacao_soma_semanal['Baia'] == BAIA].dropna()
    
    for semana in list(df_2["Ano-Semana"].unique()):
        if list(df_2[df_2["Ano-Semana"] == semana]["Operação"].unique()) != list(df_2["Operação"].unique()):
            for operacao in list(set(list(df_2["Operação"].unique())).difference(list(df_2[df_2["Ano-Semana"] == semana]["Operação"].unique()))):
                obs = {"Ano-Semana":semana,"Duracao em horas":0,"Operação":operacao}
                df_2 = df_2.append(obs, ignore_index = True)
    df_2 = df_2.sort_values(by=["Ano-Semana","Operação"], ascending=True).reset_index(drop=True)

    stud_no = pd.DataFrame({
        list(df_2["Operação"].unique())[0]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[0]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[1]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[1]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[2]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[2]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[3]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[3]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[4]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[4]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[5]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[5]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[6]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[6]]["Duracao em horas"])
        },
        index = list(df_2["Ano-Semana"].unique()))
    stud_no.plot (kind = 'bar', stacked = True)
    plt.xticks(rotation="vertical", fontsize=10)
    plt.ylabel("Soma de horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Barras 80% operacoes mais relevantes, baia "+BAIA,
        fontsize=20,
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Barras/Baia/"
        + BAIA +" Operacoes mais relevantes"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

for BANCO in ['1010','1020','1030','1040']:
    x = BANCO
    # Vamos fazer o para as 6 operacoes mais importantes e colocar as demais em outros
    
    operacao_soma_semanal = (
        df_[
            (df_["Data inicial"] == df_["Data final"])
            & (df_["Duracao em horas"] > 0)
            & (df_["Duracao em horas"] < df_["Duracao em horas"].quantile(0.99))
        ]
        .groupby(["Ano-Semana",'Banco',"Operação"])["Duracao em horas"]
        .agg("sum")
        .reset_index()
    )

    for op in outros:
        operacao_soma_semanal["Operação"] = np.where(operacao_soma_semanal["Operação"] == op,"Outro",operacao_soma_semanal["Operação"])

    operacao_soma_semanal = (
        operacao_soma_semanal.groupby(["Ano-Semana",'Banco',"Operação"])["Duracao em horas"]
        .agg("sum")
        .reset_index()
    )
    df_2 = operacao_soma_semanal[operacao_soma_semanal['Banco'] == BANCO].dropna()
    
    for semana in list(df_2["Ano-Semana"].unique()):
        if list(df_2[df_2["Ano-Semana"] == semana]["Operação"].unique()) != list(df_2["Operação"].unique()):
            for operacao in list(set(list(df_2["Operação"].unique())).difference(list(df_2[df_2["Ano-Semana"] == semana]["Operação"].unique()))):
                obs = {"Ano-Semana":semana,"Duracao em horas":0,"Operação":operacao}
                df_2 = df_2.append(obs, ignore_index = True)
    df_2 = df_2.sort_values(by=["Ano-Semana","Operação"], ascending=True).reset_index(drop=True)

    stud_no = pd.DataFrame({
        list(df_2["Operação"].unique())[0]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[0]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[1]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[1]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[2]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[2]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[3]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[3]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[4]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[4]]["Duracao em horas"]),
        list(df_2["Operação"].unique())[5]: list(df_2[df_2["Operação"] == list(df_2["Operação"].unique())[5]]["Duracao em horas"])
        },
        index = list(df_2["Ano-Semana"].unique()))
    stud_no.plot (kind = 'bar', stacked = True)
    plt.xticks(rotation="vertical", fontsize=10)
    plt.ylabel("Soma de horas", fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        "Barras 80% operacoes mais relevantes "+BANCO,
        fontsize=20,
    )
    plt.savefig(
        "C:/Users/" + USUARIO + "/EY/Vale - AI Center (Projeto) - 9. Minimização Demurrage/4-Dados/2-Dados Tratados/6-Rejeitos/Graficos/Barras/Banco/"
        + BANCO + " Operacoes mais relevantes"
        + ".png",
        bbox_inches="tight",
    )
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()