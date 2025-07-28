"""
Xplore DS :: Feature Book for E-commerce Reviews Dataset

Uso de modelos Hugging Face para gerar embeddings:
    - Criar uma conta no Hugging Face.
    - Acesse https://huggingface.co/settings/tokens.
    - Crie um token de acesso com permissões de leitura.
    - Defina a variável de ambiente HUGGINGFACE_ACCESS_TOKEN com o token gerado
    - Solicite acesso ao modelo Meta Llama 3-8B Instruct:
    - Se você não tiver acesso ao modelo Meta Llama 3-8B Instruct
    - Acesse https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.
    - Clique em "Request Access" ou "Solicitar acesso".
    - Aguarde aprovação do time da Meta/Hugging Face.
    - Após aprovação, use seu token normalmente.

"""

# Importando bibliotecas nativas
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[4]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_parquet,
    save_dataframe_to_parquet,
)
from xploreds.data_handler.dataframe import rename_columns, cast_columns_type_by_prefix
from huggingface_hub import login


# ==================================================================================
# Setup do script
script_name = os.path.basename(__file__)

# Variaveis de ambiente
load_dotenv()

# Verificando se a variável de ambiente HUGGINGFACE_TOKEN está definida
hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if hf_token is None:
    print("HUGGINGFACE_ACCESS_TOKEN environment variable not set.")
    sys.exit(1)


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
input_reviews_file_path = "data/ecommerce/curated/olist_reviews_curated_dataset.parquet"

# Configuracao de dados de saida
output_dataset_file_path = (
    "data/ecommerce/stage/olist_reviews_feature_book_dataset.parquet"
)


# ==================================================================================
# Funcoes auxiliares
def get_llm_embedding(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    embedding = embeddings_model.embed_query(text)
    return embedding


# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

data = load_dataframe_from_parquet(
    file_path=input_reviews_file_path,
    selected_columns=[
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
    log=log,
)

# ==================================================================================
# Pré-processamento de dados
# ==================================================================================
log.title("Preprocessing datasets")

log.subtitle("Removing duplicates")

log.info("Removing duplicates from reviews dataset")
data = data.drop_duplicates(subset=["order_id"], keep="first")
log.info(f"Dataframe shape after removing duplicates: {data.shape[0]} rows")

log.subtitle("Rename columns")

data = rename_columns(
    data=data,
    columns_to_rename={
        "order_id": "txt_order_id",
        "review_score": "num_review_score",
        "review_comment_title": "txt_review_comment_title",
        "review_comment_message": "txt_review_comment_message",
        "review_creation_date": "dta_review_creation_date",
        "review_answer_timestamp": "tsp_review_answer_timestamp",
    },
    log=log,
)

log.subtitle("Casting columns to appropriate types")
data = cast_columns_type_by_prefix(
    data=data,
    log=log,
)


# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Applying business rules")

log.subtitle("Generating embeddings for txt_review_comment_title")

# Inicializa o objeto de embeddings do LangChain
login(token=hf_token)
embeddings_model = HuggingFaceEmbeddings(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
)

log.info("Generating LLM embeddings for review titles")
data["vec_review_comment_title_embedding"] = data["txt_review_comment_title"].apply(
    get_llm_embedding
)
log.info("LLM embeddings generated for txt_review_comment_title")


# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")

log.subtitle("Saving dataframe")
save_dataframe_to_parquet(
    data=data,
    file_path=output_dataset_file_path,
    log=log,
)

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
