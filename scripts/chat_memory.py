import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from data_retriever import data_lake
from dotenv import load_dotenv

load_dotenv("../.env")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

db, retriever, compression_retriever = data_lake()


@st.cache_resource()
def memory():
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True, output_key="answer"
    )
    return memory


con_memory = memory()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Prompt To Generate Search Query For Retriever
search_query = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

# Chain that takes conversation history and returns documents.
retriever_chain = create_history_aware_retriever(
    llm=llm, retriever=compression_retriever, prompt=search_query
)

# Prompt To Get Response From LLM Based on Chat History
fetch_answer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\\n\\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# Chain for passing a list of Documents to a model
document_chain = create_stuff_documents_chain(llm=llm, prompt=fetch_answer)

# Create retrieval chain that retrieves documents and then passes them on to the LLM
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

response = retrieval_chain.invoke(
    {
        "chat_history": con_memory.memory_variables,
        "input": "What is Cohere mainly used for?",
    }
)
print(response["answer"])

con_memory.load_memory_variables({})

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#adding-chat-history
conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
