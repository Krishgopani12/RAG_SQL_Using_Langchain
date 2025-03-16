import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import yaml
from pyprojroot import here


class RAGCallback(BaseCallbackHandler):
    """Callback handler for RAG execution."""
    
    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        print("\nGenerating response...")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM completes a generation."""
        print("\nResponse generated.")


class GenericRAGTool:
    """A generic RAG tool that can work with any document collection."""

    def __init__(self, collection_config: Dict, llm_config: Dict):
        """
        Initialize the RAG tool.
        
        Args:
            collection_config (Dict): Configuration for the document collection
            llm_config (Dict): Configuration for the language model
        """
        self.collection_config = collection_config
        self.llm_config = llm_config
        
        # Initialize components
        self.embeddings = self._create_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.llm = self._create_llm()
        self.qa_chain = self._create_qa_chain()

    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create embeddings model."""
        return OpenAIEmbeddings(
            model=self.collection_config.get('embedding_model', 'text-embedding-3-small'),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _load_vectorstore(self) -> Chroma:
        """Load the vector store for the collection."""
        return Chroma(
            collection_name=self.collection_config['name'],
            embedding_function=self.embeddings,
            persist_directory=str(here(self.collection_config['vectordb_dir']))
        )

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance."""
        return ChatOpenAI(
            model=self.llm_config.get('model', 'gpt-4'),
            temperature=self.llm_config.get('temperature', 0.0),
            max_tokens=self.llm_config.get('max_tokens', 1000)
        )

    def _create_qa_chain(self) -> RetrievalQA:
        """Create the question-answering chain."""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.collection_config.get('k', 3)}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question (str): The question to answer
            
        Returns:
            Dict: Contains the answer and source documents
        """
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }


def create_rag_tool_from_config(config_path: str, collection_name: str) -> GenericRAGTool:
    """
    Create a RAG tool from configuration file.
    
    Args:
        config_path (str): Path to configuration file
        collection_name (str): Name of the collection to use
        
    Returns:
        GenericRAGTool: Configured RAG tool
    """
    with open(here(config_path)) as f:
        config = yaml.safe_load(f)

    # Get collection configuration
    vector_config = config['vector_db_config']
    collection_config = None
    
    for collection in vector_config['collections']:
        if collection['name'] == collection_name:
            collection_config = collection
            # Add additional settings from vector_config
            collection_config['embedding_model'] = vector_config['embedding_model']
            collection_config['chunk_size'] = vector_config['chunk_size']
            collection_config['chunk_overlap'] = vector_config['chunk_overlap']
            break
    
    if not collection_config:
        raise ValueError(f"Collection configuration not found for: {collection_name}")

    return GenericRAGTool(
        collection_config=collection_config,
        llm_config=config['llm_config']
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example usage
    rag_tool = create_rag_tool_from_config(
        config_path="configs/custom_config.yml",
        collection_name="custom-collection-1"
    )
    
    # Example query
    question = "What are the main topics covered in the documents?"
    result = rag_tool.query(question)
    
    print("\nQuestion:", question)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"\nSource {i}:", source) 