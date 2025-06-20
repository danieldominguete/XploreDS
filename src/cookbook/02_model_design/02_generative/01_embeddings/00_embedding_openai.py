"""
Xplore DS :: Script Template for OpenAI Embedding Generation
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[5]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    save_dataframe_to_parquet,
)
from xploreds.data_handler.dataframe import cast_columns_type_by_prefix

# ==================================================================================
# Setup do script

script_name = os.path.basename(__file__)

# Variaveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Criando estrutura de execucao local
env = XploreDSLocalhost(run_folder=project_folder)

# Criando estrutura de logs
log = XploreDSLogging(project_root=project_folder, script_name=script_name)
log.init_run()

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

log.title("Script setup")

# Configuracao de dados de entrada

# Parametros de operacao
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Configuracao de dados de saida
output_file_path = "data/tutorial/open_ai_embedding.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================
log.title("Loading datasets")
text_example = ["This is an example text to be embedded."]

# ==================================================================================
# Pré-processamento dos dados
# ==================================================================================
log.title("Applying preprocessing steps")

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Applying business rules")

data = pd.DataFrame(text_example, columns=["txt_text"])

# Gerando embeddings para cada texto e adicionando como colunas
log.info("Generating embeddings for all texts")
embeddings = []
for _, row in data.iterrows():
    embedding = embeddings_model.embed_query(row["txt_text"])
    embeddings.append(embedding)

embeddings_df = pd.DataFrame(
    embeddings.tolist(),
    columns=[f"num_embedding_{i}" for i in range(len(embeddings[0]))],
)
data = pd.concat([data, embeddings_df], axis=1)


db = Chroma.from_documents(
    documents=data["txt_text"].tolist(),
    embedding=embeddings,
    persist_directory="data/tutorial/chroma_db",
)

query = "What is an example text?"
results = db.similarity_search(query, k=5)
log.info(f"Results for query '{query}': {results}")

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Casting columns to appropriate types")
data = cast_columns_type_by_prefix(
    data=data,
    log=log,
)

log.subtitle("Saving dataframe to file")
save_dataframe_to_parquet(
    data=data,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
