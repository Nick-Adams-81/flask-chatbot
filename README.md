# Tournament Director Assistant (TDA) Chatbot  

## Overview  
The **TDA Chatbot** is designed to assist with poker tournament rule inquiries. It leverages **LangChain** with **FAISS** for efficient document retrieval and **OpenAI's GPT-4o** for generating responses. The chatbot processes tournament rules from a text file, retrieves relevant rule sections, and ensures responses remain within the allowed context.  

## Features  
- **Natural Language Processing (NLP)**: Uses OpenAI's GPT-4o for intelligent responses.  
- **Vector Search with FAISS**: Retrieves relevant rule sections from the provided tournament rulebook.  
- **Context Guardrails**: Ensures responses stay within predefined rule topics.  
- **Benchmark Testing with MLflow**: Evaluates chatbot accuracy against predefined test questions.  
- **Heroku Deployment**: Designed for easy cloud deployment with a Flask backend.  

## How It Works  
1. Loads tournament rules from a `.txt` file.  
2. Splits the document into chunks for vector-based retrieval.  
3. Embeds text using OpenAI's `text-embedding-3-large`.  
4. Retrieves relevant rule sections using FAISS.  
5. Generates responses using GPT-4o.  
6. Logs benchmark tests and chatbot accuracy using MLflow.  

## Deployment  
The app is configured for **Heroku deployment** and requires setting up environment variables such as `OPENAI_API_KEY`.  

## Getting Started  
To install dependencies, run:  
```sh
pip install -r requirements.txt

