import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
from flask import jsonify, request
import psutil
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

import config

#MADI

from transformers import AutoTokenizer, GenerationConfig, pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
callbacks = [StreamingStdOutCallbackHandler()]

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


logger = hivemind.get_logger(__file__)
#-----------------------------------
from transformers import TextStreamer,TextIteratorStreamer
from typing import Dict, Union, Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import tracing_enabled
from langchain.llms import OpenAI

def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        try:
            loader = loader_class(file_path)
        except Exception as e:
            logger.warning(f"ignoring a malformed file, filename: {file_path}, err: {e}")
            raise(e)
            
    else:
        raise ValueError("Document type is undefined")
    try: 
        return loader.load()[0]
    except Exception as e:
        logger.warning(f"ignoring a malformed file, filename: {file_path}, err: {e}")
        raise(e)


def load_document_batch(filepaths):
    logger.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as e:
                logger.warning(f"ignoring a malformed file, filename: {future.exception().args[0]}, err: {e}")
                continue

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs
def update_from_local():
    # Load documents and split in chunks
    logger.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logger.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,

    )
    logger.info(f"Knowledge DB Updated with private Data !!")
    return "OK"

# First, define custom callback handler implementations
class MyCustomHandlerOne(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        #print(f"on_llm_start {serialized['name']}")
        logger.info(f"on_llm_start: YES")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        logger.info(f"on_new_token {token}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        #print(f"on_chain_start {serialized['name']}")
        logger.info(f"on_chain_start: YES")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        #print(f"on_tool_start {serialized['name']}")
        logger.info(f"on_tool_start: YES")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        logger.info(f"on_agent_action {action}")


class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start (I'm the second handler!!) {serialized['name']}")


# Instantiate the handlers
handler1 = MyCustomHandlerOne()
handler2 = MyCustomHandlerTwo()
#------------------------------------
os.environ['HF_ACCESS_TOKEN'] = "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"
#process.env.HF_ACCESS_TOKEN = 'hf_...';
access_token = "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"

models = {}
for model_info in config.MODELS:
    logger.info(f"Loading tokenizer for {model_info.repo}")
    tokenizer = AutoTokenizer.from_pretrained(model_info.repo, add_bos_token=False, use_fast=True,token=access_token,)

    logger.info(f"Loading model {model_info.repo} with adapter {model_info.adapter} and dtype {config.TORCH_DTYPE}")
    # We set use_fast=False since LlamaTokenizerFast takes a long time to init
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_info.repo,
        active_adapter=model_info.adapter,
        torch_dtype=config.TORCH_DTYPE,
        initial_peers=config.INITIAL_PEERS,
        max_retries=3,
        token=access_token,
    )
    model = model.to(config.DEVICE)
    generation_config = GenerationConfig.from_pretrained(model_info.repo)
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE})
    model_name = model_info.name
    if model_name is None:  # Use default name based on model/repo repo
        model_name = model_info.adapter if model_info.adapter is not None else model_info.repo
    models[model_name] = model,tokenizer,generation_config,embeddings #local_llm,embeddings
#update_from_local()
logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
