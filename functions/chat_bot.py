import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .guardrails.context_guardrail import get_allowed_keywords, is_question_relevant, REFERENCE_TEXT, get_embedding, get_reference_embedding
from .guardrails.safety_guardrail import safety_guardrail
from .data_ingestion.txt_loader import txt_loader, retriever

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize chat history globally (or you can use a database/session for persisting)
chat_history = []

def chat_bot(document_path, user_input):
    # Ensure chat_history is global so it persists across calls
    global chat_history

    # Check if the user input contains harmful language using the safety guardrail
    is_profane, profane_words = safety_guardrail(user_input)
    if is_profane:
        return f"Please refrain from using harmful or offensive language. Detected words: {', '.join(profane_words)}"

    # Using our LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Load and process the document(s)
    document = txt_loader(document_path)

    # Retrieve chunked documents
    retrieve = retriever(document)

    # Read document text (optional, for relevance checks)
    with open(document_path, "r") as file:
        document_text = file.read()

    # Get allowed keywords for relevance check
    allowed_keywords = get_allowed_keywords(document_text)
    reference_embedding = get_reference_embedding()

    # Check if the user's question is relevant using the guardrail
    if not is_question_relevant(user_input, allowed_keywords, reference_embedding):
        return "I can only answer questions related to TDA rules and rulings."

    # Add the user input to the chat history
    chat_history.append({"user": user_input})

    # Format the chat history for the prompt
    history_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history if 'bot' in msg])

    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """ You are an assistant for answering questions about 
        tournament poker rules. Use the provided context to respond. If the user query is not relevant,
        say you can't answer. If the answer isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.{context} 
        """),
        ("human", "{history}\nUser: {input}")
    ])

    # Create chain
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retrieve, qa_chain)

    # Get response from RAG chain
    response = rag_chain.invoke({"input": user_input, "history": history_text})

    # Extract bot's answer
    bot_answer = response['answer']

    # Append bot response to chat history
    # Add the bot's response to the latest entry
    chat_history[-1]["bot"] = bot_answer

    return bot_answer
