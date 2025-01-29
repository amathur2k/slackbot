import os
import pinecone
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

os.environ["PINECONE_ENV"] = "us-east-1"  
os.environ["PINECONE_INDEX"] = "fourth-index"



class SlackRAG:
    def __init__(self):
        # Initialize OpenAI
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX"])
        
        # Create vector store
        self.vectorstore = Pinecone(
            index=index,
            embedding=self.embeddings,
            text_key="text"
        )

        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Given the following Slack messages:

{context}

Question: {question}

Please answer the question based only on the provided Slack messages. Do not take into account any messages by or adressed to the user : U08BHCTUF7S 
Include a section on citations at the end of your answer, clearly highlighting the message sender and content you used.
If the information isn't available in the messages, please say so.
"""
        )

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a readable context string"""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            message = f"[Message {i}]\n"
            message += f"Content: {doc.page_content}\n"
            if metadata.get('channel'):
                message += f"Channel: {metadata['channel']}\n"
            if metadata.get('timestamp'):
                message += f"Time: {metadata['timestamp']}\n"
            formatted_docs.append(message)
        return "\n".join(formatted_docs)

    def query(self, question: str) -> Dict[str, str]:
        """Process a query and return answer with context"""
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(
            question,
            k=5  # Retrieve top 4 most relevant messages
        )

        if not docs:
            return {
                "answer": "No relevant information found in the Slack messages.",
                "context": ""
            }

        # Format context
        context = self.format_context(docs)

        # Generate response using LLM
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.llm.predict(prompt)

        return {
            "answer": response,
            "context": context
        }

slack_web_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
handled_messages = {}  
rag = SlackRAG()

def clean_slack_message(text: str) -> str:
    """Remove Slack user mention tags from message text"""
    # Remove any <@USER_ID> mentions from the text
    import re
    cleaned_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    return cleaned_text

def process(client: SocketModeClient, socket_mode_request: SocketModeRequest):
    if socket_mode_request.type == "events_api":
        event = socket_mode_request.payload.get("event", {})
        client_msg_id = event.get("client_msg_id")
        if event.get("type") == "app_mention" and not handled_messages.get(client_msg_id):
            handled_messages[client_msg_id] = True
            channel_id = event.get("channel")
            text = event.get("text")
            # Clean the message text
            cleaned_text = clean_slack_message(text)
            print("I was asked :- ", cleaned_text)

            result = rag.query(cleaned_text)
            print("I answered :- ", result)
            result = result["answer"]
            slack_web_client.chat_postMessage(channel=channel_id, text=result)
    
    return SocketModeResponse(envelope_id=socket_mode_request.envelope_id)

def main():
    # Usage example
    
    
    # Example query
    #result = rag.query("What was discussed about the project timeline?")
    #print("Answer:", result["answer"])
    #print("\nRelevant Messages:", result["context"])


    '''
    print("What do you want to know?")
    while True:
        query = input("What do you want to know?")
        answer = rag.query(query)
        print(answer["answer"])
        print("\nWhat else can I help you with:")
    '''

    
    socket_mode_client = SocketModeClient(
        app_token=os.environ["SLACK_APP_TOKEN"], 
        web_client=slack_web_client
    )
    socket_mode_client.socket_mode_request_listeners.append(process)

    socket_mode_client.connect()
    print("listening")
    from threading import Event
    Event().wait()
    

if __name__ == "__main__":
    main()
