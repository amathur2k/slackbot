# This script is running the bot with improved prompts locally in your terminal

import os
from typing import List
from langchain_core.prompts import PromptTemplate
import pinecone
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema.document import Document

embeddings = OpenAIEmbeddings()

os.environ["PINECONE_API_KEY"] = "pcsk_2MpYP6_DX2FpKzg2H4qbbykjkCB3pL6c5jQtGRhekVy3pQGpgcfTJ5zkgcVce5mW2TQeDK"
os.environ["PINECONE_ENV"] = "us-east-1"  
os.environ["PINECONE_INDEX"] = "test-index"

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])
vector_store = Pinecone(index=index, embedding=embeddings, text_key="text")

prompt_template = """You are a question-answering bot operating on Github issues and documentation pages for a product called connector builder. The documentation pages document what can be done, the issues document future plans and bugs. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. State were you got this information from (and the github issue number if applicable), but do only if you used the information in your answer.

{context}

Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
class ContextualRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)
        return [self.format_doc(doc) for doc in docs]

    def format_doc(self, doc: Document) -> Document:
        stream_type = doc.metadata.get("_airbyte_stream", "unknown")
        
        if stream_type == "item_collection":
            doc.page_content = f"Excerpt from documentation page: {doc.page_content}"
        elif stream_type == "issues":
            doc.page_content = f"Excerpt from Github issue: {doc.page_content}, issue number: {int(doc.metadata.get('number', 0)):d}, issue state: {doc.metadata.get('state', 'unknown')}"
        elif stream_type in ["threads", "channel_messages"]:
            doc.page_content = f"Excerpt from Slack thread: {doc.page_content}"
        else:
            doc.page_content = f"Excerpt from source: {doc.page_content}"
        
        return doc


qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=ContextualRetriever(vectorstore=vector_store),
    chain_type_kwargs={"prompt": prompt}
)

'''
print("Connector development help bot. What do you want to know?")
while True:
    query = input("What do you want to know?")
    answer = qa.run(query)
    print(answer)
    print("\nWhat else can I help you with:")
'''

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse


slack_web_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

handled_messages = {}

def process(client: SocketModeClient, socket_mode_request: SocketModeRequest):
    if socket_mode_request.type == "events_api":
        event = socket_mode_request.payload.get("event", {})
        client_msg_id = event.get("client_msg_id")
        if event.get("type") == "app_mention" and not handled_messages.get(client_msg_id):
            handled_messages[client_msg_id] = True
            channel_id = event.get("channel")
            text = event.get("text")
            print("I was asked :- ", text)
            #text = "What is the meaning of life?"
            #result = "Hello World"
            result = qa.run(text)
            print("I answered :- ", result)
            slack_web_client.chat_postMessage(channel=channel_id, text=result)
    
    return SocketModeResponse(envelope_id=socket_mode_request.envelope_id)

socket_mode_client = SocketModeClient(
    app_token=os.environ["SLACK_APP_TOKEN"], 
    web_client=slack_web_client
)
socket_mode_client.socket_mode_request_listeners.append(process)

socket_mode_client.connect()
print("listening")
from threading import Event
Event().wait()


