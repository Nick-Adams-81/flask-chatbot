import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .guardrails.context_guardrail import get_allowed_keywords, is_question_relevant, REFERENCE_TEXT, get_embedding
from .guardrails.safety_guardrail import safety_guardrail
from .data_ingestion import txt_loader

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def chat_bot(document_path, user_input):

    # Check if the user input contains harmful language using the safety guardrail.
    is_profane, profane_words = safety_guardrail(user_input)
    print(f"Profanity check result: {is_profane}, Detected words: {profane_words}")  # Debugging line

    if is_profane:
        return f"Please refrain from using harmful or offensive language. Detected words: {', '.join(profane_words)}"

    # Using our LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Load and process the document(s)
    document = txt_loader.txt_loader(document_path)

    # Retrieve chunked documents
    retriever = txt_loader.retriever(document)

    # Initialize the chat history
    chat_history = []

    with open(document_path, "r") as file:
        document_text = file.read()

    # Get allowed keywords for relevance check
    allowed_keywords = get_allowed_keywords(document_text)
    reference_embedding = get_embedding(REFERENCE_TEXT)

    # Check if the user's question is relevant using the guardrail
    if not is_question_relevant(user_input, allowed_keywords, reference_embedding):
        return "I can only answer questions related to TDA rules and rulings."

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """ You are an assistant for answering questions about 
        tournament poker rules. Use the provided context to respond. If the user query is not relevent,
        say you can't answer.If the answer isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.{context} 
        """),
        ("human", "{history}\nUser: {input}")
    ])

    # Create chain
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Format history as a single string
    history_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history])

    # Get response from RAG chain
    response = rag_chain.invoke({"input": user_input, "history": history_text})

    # Extract bot's answer
    bot_answer = response['answer']

    return bot_answer