from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables
load_dotenv()

# Cache the retriever function as a resource
@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory='db', embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5}
    )
    return retriever

# Define the main LLM pipeline
def llm_pipeline(query):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    retriever = get_retriever()  # Use the cached retriever resource
    relevant_docs = retriever.invoke(query)

    # Create context from retrieved documents
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Define the prompt template
    prompt_temp = """You are a helpful assistant. Answer the below question from the given context.
Question: {query}
Context:
{context}"""
    prompt = ChatPromptTemplate.from_template(prompt_temp)
    
    # Initialize the chain with prompt, model, and output parser
    chain = prompt | model | StrOutputParser()
    
    # Generate the response
    response = chain.invoke({"query": query, "context": context})
    return str(response)  # Ensure response is converted to a string for display

# Define the main Streamlit function
def main():
    st.title("Search your PDF")  # Corrected the typo here from 'tile' to 'title'
    
    # User input
    input_query = st.text_input("Write your question & press submit.")
    
    if st.button("Submit") and input_query:
        # Run the LLM pipeline and display the output
        output = llm_pipeline(input_query)
        st.subheader("Query:")
        st.text(input_query)
        st.subheader("Response:")
        st.text(output)

# Run the main function
if __name__ == "__main__":
    main()
