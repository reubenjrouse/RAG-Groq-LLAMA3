import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key = os.getenv("GROQ_API_KEY")


st.title("RAG (Llama3-70b) with PDF uploads")
st.write("Upload PDFs and chat with their content")
api_key=st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(model = 'llama3-70b-8192', groq_api_key=api_key)
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files = True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            tempdf = f"./temp.pdf"
            with open(tempdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(tempdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            """ 
    Given a chat history and the latest user question which might
    reference context in the chat history formualte a standalone question
    which can be understood without the chat history. Do NOT answer the 
    question, just reformulate it if needed and otherwise return it as it is.
    """
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            """ 
            You are an assistant for question-answering tasks. 
            Use the following piees of retrieved context to answer
            the question. If ou dont know the answer, say that you
            dont know. Use three sentences maximum and keep the answer
            concise.

            {context}
    """
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable":{"session_id":session_id}
                    }
            )
        
        st.write(st.session_state.store)
        st.write('Assistant:',response['answer'])
        st.write("Chat History", session_history.messages)

else:
    st.warning("Enter Groq API")