from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

retriever = db.as_retriever()
llm = ChatOpenAI()

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
def query_knowledge_base(question: str):
    return qa.run(question)
