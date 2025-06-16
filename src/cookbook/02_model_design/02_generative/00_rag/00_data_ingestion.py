"""
Xplore DS :: Cookbook to produce a RAG solution

Data Ingestion Script
This script is designed to ingest data for a RAG (Retrieval-Augmented Generation) solution.

Reference:
https://python.langchain.com/docs/concepts/document_loaders/

"""

# Importando bibliotecas nativas
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[5]
sys.path.append(str(project_folder))

# Importando biblioteca Xplore DS
from xploreds.environment.environment import XploreDSLocalhost
from xploreds.environment.logging import XploreDSLogging

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
input_text_dataset_path = "data/credit-g/raw/adhoc_files/txt/speech.txt"
input_pdf_dataset_path = "data/credit-g/raw/adhoc_files/pdf/attention.pdf"
input_web_page_dataset_path = "https://pt.wikipedia.org/wiki/Michael_Jordan"

# ==================================================================================
# Carregando base de dados
# ==================================================================================

log.title("Loading datasets")
txt_loader = TextLoader(input_text_dataset_path, encoding="utf-8")
txt_documents = txt_loader.load()

pdf_loader = PyPDFLoader(input_pdf_dataset_path)
pdf_documents = pdf_loader.load()

web_page_loader = WebBaseLoader(input_web_page_dataset_path)
web_page_documents = web_page_loader.load()

# ==================================================================================
# Split texts into chunks
# ==================================================================================

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_txt_docs = text_splitter.split_documents(txt_documents)
final_pdf_docs = text_splitter.split_documents(pdf_documents)
final_web_page_docs = text_splitter.split_documents(web_page_documents)

# ==================================================================================
# Split texts into chunks
# ==================================================================================

text_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=500, chunk_overlap=50
)
final_txt_docs = text_splitter.split_documents(txt_documents)
final_pdf_docs = text_splitter.split_documents(pdf_documents)
final_web_page_docs = text_splitter.split_documents(web_page_documents)

# ==================================================================================
# Salvando artefatos de saida
# ==================================================================================


# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
