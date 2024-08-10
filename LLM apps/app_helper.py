from langchain.llms import GooglePalm
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]
llm = GooglePalm(google_api_key = api_key, temprature = 0.9)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path='dataset.csv', source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    vectordb.save_local(vectordb_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    
    prompt_tempate = ''' 

    Generate a answer based upon the given context. Try to provide the answer as far as possible from the responce section of the provided document. If answer not found, straight forward state "No response avaiable."
    '''

    CONTEXT : {context}

    QUESTION: {question}

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()