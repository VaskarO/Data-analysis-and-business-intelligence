import os
import time
import langchain
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = 'open ai key'
llm = OpenAI(temprature=0.9, max_tokens = 500)

loaders =  UnstructuredURLLoader(urls= ['url1', 'url2'])

data = loaders.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap= 200
)

docs = text_spliter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectorindex_openai = FAISS.from_docuemnts(docs, embeddings)