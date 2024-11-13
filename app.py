from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from ingest import ingest_doc

load_dotenv()

@st.cache_resource
def get_retriver():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory='db',embedding_function=embeddings)
    retriver = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs ={"k":1,"score_threshold":0.2}
    )
    print("--------- Retriver created ------")
    return retriver

def llm_pipeline(query):
    model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash",temperature=0.2)
    retriver = get_retriver()
    relevant_docs = retriver.invoke(query)
    print("--------- relevant_docs ------")
    print(relevant_docs)
    prompt_temp = """You are a helpful assistance answer the below question from the given context.
                /nQuestion : {query} /nContext: /n {context}"""
    prompt = ChatPromptTemplate.from_template(prompt_temp)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"query":query,"context":relevant_docs})
    return str(response)

def main():
    st.title("🔎 Search your PDF 📑")
    file = st.file_uploader("Upload your PDF 📥",type = "pdf")
    if file is not None:
        print("--------- Inside File upload ------")
        ingest_doc(file)
        print("--------- File Uploaded ------")
        input_query = st.text_input("Write Your question & press submit.")
        if st.button("Submit🚀") and input != '':
            output = llm_pipeline(input_query)
            st.subheader("Query:")
            st.text(input_query)
            st.subheader("Response:")
            st.markdown(output)


if __name__ =="__main__":
    main()