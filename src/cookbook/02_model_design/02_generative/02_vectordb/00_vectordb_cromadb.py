"""
Xplore DS :: General cookbook for ChromaDB vector database

"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[5]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging
from xploreds.data_handler.file import (
    load_dataframe_from_csv,
    save_dataframe_to_parquet,
)
from xploreds.data_handler.dataframe import cast_columns_type_by_prefix

# ==================================================================================
# Setup do script

script_name = os.path.basename(__file__)

# Variaveis de ambiente
load_dotenv()

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
input_file_path = "data/tutorial/txt/speech.txt"

# Parametros de operacao

# Configuracao de dados de saida
output_file_path = "data/tutorial/chroma_vector_db/speech_chroma_vector_db"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")

loader = TextLoader(input_file_path, encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, length_function=len
)
texts = text_splitter.split_documents(documents)

# ==================================================================================
# Pré-processamento dos dados
# ==================================================================================
log.title("Applying preprocessing steps")

embedding_model = OllamaEmbeddings(model="llama3.2")
db = Chroma.from_documents(texts, embedding_model, persist_directory=output_file_path)

# ==================================================================================
# Regras de negócio
# ==================================================================================
log.title("Applying business rules")

query = "What is the main topic of the speech?"

# Usando similarity search para encontrar o documento mais relevante
log.info("Performing similarity search")
docs = db.similarity_search(query, k=1)
log.info(f"Query: {query}")
for doc in docs:
    log.info(f"Document: {doc.page_content}")
    log.info(f"Metadata: {doc.metadata}")

# Usando retrieval para buscar o documento mais relevante
log.info("Performing retrieval")
retriever = db.as_retriever()
docs = retriever.invoke(query)
log.info(f"Query: {query}")
for doc in docs:
    log.info(f"Document: {doc.page_content}")
    log.info(f"Metadata: {doc.metadata}")

# Usando similarity search com score
log.info("Performing similarity search with score")
docs_with_score = db.similarity_search_with_score(query, k=1)
log.info(f"Query: {query}")
for doc, score in docs_with_score:
    log.info(f"Document: {doc.page_content}")
    log.info(f"Metadata: {doc.metadata}")
    log.info(f"Score: {score}")

# Buscando pelo embedding de uma consulta
log.info("Performing similarity search with embedding")
input_embedding = embedding_model.embed_query(query)
docs = db.similarity_search_by_vector(input_embedding, k=1)
log.info(f"Query: {query}")
for doc in docs:
    log.info(f"Document: {doc.page_content}")
    log.info(f"Metadata: {doc.metadata}")


# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================

log.title("Saving output artifacts")


new_db = Chroma(
    persist_directory=output_file_path,
    embedding_function=embedding_model,
)
log.info("Verifying if the saved database can be loaded correctly")
assert new_db is not None, "Failed to load the saved CHROMA vector database."
log.info("Chroma vector database loaded successfully from the saved file.")

# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
