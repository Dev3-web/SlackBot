# embed_knowledge_base.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = (
    "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"
)

loader = PyPDFLoader("docs/my_knowledge.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_index")
