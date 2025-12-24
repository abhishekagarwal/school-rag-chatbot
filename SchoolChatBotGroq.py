import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if not os.path.exists("./chroma_db"):

    pdfLoader = PyPDFLoader('school_doc.pdf')
    pdfDocs = pdfLoader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    data = text_splitter.split_documents(pdfDocs)

    db = Chroma.from_documents(
    data,
    embeddings,
    persist_directory="./chroma_db"
    )
    db.persist()

else:
     db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )



st.title("School Chatbot Demo")
input_text = st.text_input("What question you want to ask?")

# llm = OllamaLLM(model = "phi4-mini")
llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Try to answer with minimum 3 lines
Answer the question using the provided context.
If the answer is partially available, answer as best as you can.
If it is completely missing, say ""Sorry, I don't have these details available. Please contact +91-1234567890 for more details"".

Context:
{context}

Question:
{question}
""")

outputParser = StrOutputParser() #to get direct answer

chain = prompt | llm | outputParser


if input_text:
    retrieved_results = db.similarity_search(input_text, k=4)
    context = "\n\n".join([d.page_content for d in retrieved_results])
    answer = chain.invoke({
        "context": context,
        "question": input_text
    })

    st.write(answer)

