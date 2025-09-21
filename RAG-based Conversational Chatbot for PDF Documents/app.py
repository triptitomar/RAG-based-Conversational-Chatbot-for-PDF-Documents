## RAG Q&A Conversation With PDF Including Chat History
from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load Hugging Face token
token = os.getenv("HF_TOKEN")
if token:
    os.environ['HF_TOKEN'] = token
else:
    raise ValueError("HF_TOKEN is not set in the environment.")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app title
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session ID input
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state for documents, retriever, chat_history, etc.
    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Choose PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            try:
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                st.session_state.documents.extend(docs)
                st.success(f"Loaded {uploaded_file.name} successfully")
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {e}")

    # Only create vectorstore/retriever if documents exist
    if st.session_state.documents and st.session_state.vectorstore is None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.success(f"{len(splits)} chunks created and embeddings generated.")

    # Only proceed if retriever is ready
    if st.session_state.retriever:
        # Create history-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, contextualize_q_prompt
        )

        # Create QA chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Wrap RAG chain with session-based chat history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda _: st.session_state.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User question input
        user_input = st.text_input("Your question:")
        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(response['answer'])

            st.write("Assistant:", response['answer'])
            st.write("Chat History:", st.session_state.chat_history.messages)
    else:
        st.info("Upload PDFs to enable question-answering.")
else:
    st.warning("Please enter the Groq API Key")
