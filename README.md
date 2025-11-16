Here is the **clean README.md** version **without any symbols (no emojis, no ticks, no special icons)** — safe to paste directly into GitHub.

---

# GUVI Knowledge Retrieval Chatbot (RAG-Based AI Chatbot)

This project is an AI-powered Q&A system that answers questions about GUVI Courses, Zen Class, Certifications, and FAQs using Retrieval-Augmented Generation (RAG).

The chatbot retrieves accurate information from GUVI blogs, course descriptions, and FAQs, then generates natural language responses using the OpenAI GPT model.

---

## Project Overview

The GUVI Knowledge Retrieval Chatbot helps users get accurate answers related to:

* GUVI courses
* Zen Class
* Certification procedures
* FAQs
* GUVI platforms and services

The system uses:

* Retrieval-Augmented Generation (RAG)
* FAISS vector search
* SentenceTransformer embeddings
* OpenAI GPT model
* Streamlit interface

---

## Features

1. Intelligent question answering
2. RAG-based accurate responses
3. Preprocessed GUVI dataset
4. FAISS vector index for fast search
5. Clean and responsive Streamlit UI

---

## Project Structure

```
GUVI_RAG_CHATBOT/
│
├── app.py                     # Streamlit application
├── build_index.py             # Builds FAISS index and embeddings
├── retriever_llm.py           # Retrieval + LLM Answer Generation
├── embedding_generation.py    # Creates embeddings
├── data_preprocess.py         # Cleans text data
│
├── data/
│   ├── guvi_courses.txt
│   ├── guvi_blogs.txt
│   ├── guvi_faqs.txt
│   ├── cleaned_chunks.txt
│   ├── chunks_list.txt
│   └── faiss_index.bin
│
├── requirements.txt           # Required dependencies
└── README.md                  # Project documentation
```

---

## Tech Stack

### Backend and Machine Learning

* Python
* SentenceTransformer (all-MiniLM-L6-v2)
* FAISS CPU
* OpenAI GPT API

### Frontend

* Streamlit

### Data

* Local text documents
* FAISS binary vector index

---

## RAG Pipeline Workflow

1. Data Preprocessing

   * Loads GUVI text files
   * Cleans unwanted characters
   * Splits into text chunks

2. Embedding Generation

   * Uses SentenceTransformer to convert text chunks into vectors

3. FAISS Index Creation

   * Stores vector embeddings in faiss_index.bin

4. Query Processing

   * User enters question
   * System embeds the query
   * Retrieves top similar chunks from FAISS
   * Sends context to GPT model

5. Final Output

   * Displays final refined answer with retrieved context

---

## Installation and Setup

Clone the repository:

```
git clone https://github.com/Vadivukarasimoorthy/guvi_rag_chatbot
cd guvi_rag_chatbot
```

Install dependencies:

```
pip install -r requirements.txt
```

Add your OpenAI key in Streamlit secrets:

Create file:

```
~/.streamlit/secrets.toml
```

Add:

```
OPENAI_API_KEY = "your_api_key_here"
```

Run the application:

streamlit run app.py


## Example Questions to Ask

You can ask:

* What is Zen Class?
* What courses does GUVI offer?
* How do I get certified in GUVI?
* What can I do with Geekoins?
* What is GUVI’s refund policy?
* I cannot log in. What should I do?

## Model Used

* SentenceTransformer: all-MiniLM-L6-v2
* FAISS CPU Index for similarity search
* OpenAI ChatCompletion model for generating responses

## Future Enhancements

* Add chat history
* Add support for multiple data sources
* Deploy on Streamlit Cloud
* Add multilingual RAG support
* Improve dataset for complete GUVI website coverage

## Developer

Name: Vadivukarasi M
Project: GUVI Knowledge Retrieval Chatbot
Batch: CSR-AIML-C-WE-E-B1


