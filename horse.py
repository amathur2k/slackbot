import os
import pinecone
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

os.environ["PINECONE_ENV"] = "us-east-1"  
os.environ["PINECONE_INDEX"] = "second-index"



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

Please answer the question based only on the provided Slack messages. 
Include citations to specific messages you used in your answer.
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
            k=4  # Retrieve top 4 most relevant messages
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

def main():
    # Usage example
    rag = SlackRAG()
    
    # Example query
    #result = rag.query("What was discussed about the project timeline?")
    #print("Answer:", result["answer"])
    #print("\nRelevant Messages:", result["context"])


    print("What do you want to know?")
    while True:
        query = input("What do you want to know?")
        answer = rag.query(query)
        print(answer["answer"])
        print("\nWhat else can I help you with:")


if __name__ == "__main__":
    #main()
