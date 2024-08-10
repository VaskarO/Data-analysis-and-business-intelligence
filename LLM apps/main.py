import streamlit as st
from app_helper import get_qa_chain, create_vector_db

st.title("Sampe QA application")
btn = st.button("Knowledgebase generation")
if btn:
    create_vector_db()

question = st.text_input("Your query...")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Response")
    st.write(response["result"])