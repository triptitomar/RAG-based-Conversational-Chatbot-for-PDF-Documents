## 📄 RAG-based Conversational Chatbot for PDF Documents

🛠️ Technology Stack:

Languages & Frameworks: Python, Streamlit
Libraries & Tools: LangChain, HuggingFace, PyTorch, Chroma, Groq API
Other Tools: PyPDFLoader, RecursiveCharacterTextSplitter, Environment Variables (.env)

✨ Key Features:

-Developed a RAG (Retrieval-Augmented Generation) pipeline to intelligently answer questions from uploaded PDFs.
-Implemented context-aware question reformulation leveraging LangChain chat history for accurate responses.
-Created a Streamlit web app enabling interactive PDF uploads and real-time Q&A.
-Managed vector embeddings with HuggingFace and Chroma to optimize document retrieval.
-Integrated Groq API LLM for concise, three-sentence answers.
-Ensured stateful chat history using st.session_state to track and maintain user interactions.

📌 Project Overview

This project allows users to upload PDF documents and interact with a conversational AI that answers questions based on the document content. The system intelligently understands context from previous messages, reformulates questions when necessary, and retrieves relevant information from the PDFs using vector embeddings.
