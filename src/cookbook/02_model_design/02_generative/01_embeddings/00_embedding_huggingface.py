"""
Xplore DS :: Script Template for Hugging Face Embedding Generation
"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

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
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

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
embeddings_model = HuggingFaceEmbeddings(model_name="snowflake-arctic-embed-m")

# Configuracao de dados de saida
output_file_path = "data/tutorial/hf_embedding.parquet"

# ==================================================================================
# Carregando base de dados
# ==================================================================================
log.title("Loading datasets")
text_example = [
    "This is an example text to be embedded.",
    "Here is another example text for embedding.",
]

# ==================================================================================
# Pré-processamento dos dados
# ==================================================================================
log.title("Applying preprocessing steps")

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Applying business rules")

data = pd.DataFrame(text_example, columns=["txt_text"])

# Gerando embeddings para cada texto e adicionando como colunas em um pandas
log.info("Generating embeddings for all texts")
embeddings = embeddings_model.embed_documents(data["txt_text"].tolist())
embeddings_df = pd.DataFrame(
    embeddings,
    columns=[f"num_embedding_{i}" for i in range(len(embeddings[0]))],
)
data = pd.concat([data, embeddings_df], axis=1)

# Gerando embeddings e armazenando em um banco de dados vetorial
log.info("Storing embeddings in a vector database")

client = chromadb.Client()
collection = client.create_collection(name="docs")
for i, row in data.iterrows():
    collection.add(
        ids=[str(i)],
        documents=[row["txt_text"]],
        embeddings=[
            row[[f"num_embedding_{j}" for j in range(len(embeddings[0]))]].tolist()
        ],
        metadatas=[{"index": i}],
    )


# recuperando os dados do banco de dados vetorial
log.info("Retrieving data from the vector database")
input = "Is this another example text to be embedded?"
input_embedding = embeddings_model.embed_query(input)
results = collection.query(
    query_embeddings=[input_embedding],
    n_results=5,
)
log.info(f"Retrieved {len(results['ids'][0])} results from the vector database")
log.info("Results:")
for i, doc_id in enumerate(results["ids"][0]):
    log.info(f"Result {i + 1}:")
    log.info(f"ID: {doc_id}")
    log.info(f"Document: {results['documents'][0][i]}")
    log.info(f"Metadata: {results['metadatas'][0][i]}")

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Casting columns to appropriate types")
data = cast_columns_type_by_prefix(
    data=data,
    # log=log,
)

log.subtitle("Saving dataframe to file")
save_dataframe_to_parquet(
    data=data,
    file_path=output_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
