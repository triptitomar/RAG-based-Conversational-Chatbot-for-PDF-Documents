# RAG-based-Conversational-Chatbot-for-PDF-Documents

Technology Stack:
-Languages & Frameworks: Python, Streamlit
-Libraries/Tools: LangChain, HuggingFace, PyTorch, Chroma, Groq API
-Other Tools: PDFLoader, RecursiveCharacterTextSplitter, Environment Variables (.env)

Key Features:
-Use bullet points, start with action verbs:
-Developed a RAG (Retrieval-Augmented Generation) pipeline to answer questions from uploaded PDFs.
-Implemented context-aware question reformulation using LangChain chat history.
-Created a Streamlit web app for interactive PDF uploads and real-time Q&A.
-Managed vector embeddings with HuggingFace and Chroma for document retrieval.
-Integrated Groq API LLM for concise, three-sentence responses.
-Ensured stateful chat history using st.session_state to track user interactions.
