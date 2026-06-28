from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder
import gradio as gr

from config import (
    LLM_MODEL_ID,
    EMBEDDING_MODEL_ID,
    WATSONX_URL,
    PROJECT_ID,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_TRUNCATE_TOKENS,
)

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


def get_llm():
    """Initialize and return the WatsonxLLM model."""
    parameters = {
        GenParams.MAX_NEW_TOKENS: MAX_NEW_TOKENS,
        GenParams.TEMPERATURE: TEMPERATURE,
    }
    watsonx_llm = WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=parameters,
    )
    return watsonx_llm


def watsonx_embedding():
    """Initialize and return the WatsonxEmbeddings model."""
    embed_params = {
        "truncate_input_tokens": EMBED_TRUNCATE_TOKENS
    }
    embedding = WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=embed_params,
    )
    return embedding


def document_loader(file):
    """Load and parse a PDF document from a given file path."""
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    """Split loaded documents into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


def vector_database(chunks):
    """Create a Chroma vector store from document chunks."""
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


def retriever(file):
    """Build a retriever from the uploaded PDF file."""
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


def retriever_qa(file, query):
    """Run the RAG pipeline and return an answer to the user's query."""
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True
    )
    response = qa.invoke({"query": query})
    return response['result']


# Gradio UI
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=['.pdf'],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here..."
        )
    ],
    outputs=gr.Textbox(label="Generated Response"),
    title="Watsonx.ai RAG — Document QA Bot",
    description="Upload a PDF document and ask any question. The bot answers using only your document."
)

if __name__ == "__main__":
    rag_application.launch(share=True)
