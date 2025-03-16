import os
import yaml
from typing import List, Dict
from pyprojroot import here
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import uuid
import base64
from PIL import Image
import io
import glob

class PrepareVectorDB:
    """
    A class to prepare Vector Databases using advanced document processing including
    text, tables, and images from PDF documents.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the PrepareVectorDB with configuration settings.

        Args:
            config_path (str): Path to the configuration file.
        """
        with open(here(config_path)) as cfg:
            self.config = yaml.load(cfg, Loader=yaml.FullLoader)
        
        self.vector_config = self.config['vector_db_config']
        self.chunk_size = self.vector_config['chunk_size']
        self.chunk_overlap = self.vector_config['chunk_overlap']
        self.embedding_model = self.vector_config['embedding_model']
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def process_collection(self, collection: Dict) -> None:
        """
        Process a single document collection and create its vector database.

        Args:
            collection (Dict): Collection configuration containing name, doc_dir, and vectordb_dir.
        """
        collection_name = collection['name']
        doc_dir = collection['doc_dir']
        vectordb_dir = collection['vectordb_dir']

        # Create vectordb directory if it doesn't exist
        if not os.path.exists(here(vectordb_dir)):
            os.makedirs(here(vectordb_dir))
            print(f"Created directory '{vectordb_dir}' for collection '{collection_name}'")

            # Find PDF files in the directory
            pdf_pattern = os.path.join(here(doc_dir), "*.pdf")
            pdf_files = glob.glob(pdf_pattern)
            
            if not pdf_files:
                print(f"No PDF files found in {doc_dir}")
                return
                
            print(f"Found PDF files: {pdf_files}")
            
            for file_path in pdf_files:
                try:
                    # Extract content from PDF
                    chunks = partition_pdf(
                        filename=file_path,
                        infer_table_structure=True,
                        strategy="hi_res",
                        extract_image_block_types=["Image", "Table"],
                        extract_image_block_to_payload=True,
                        chunking_strategy="by_title",
                        max_characters=10000,
                        combine_text_under_n_chars=2000,
                        new_after_n_chars=6000,
                    )

                    # Separate texts, tables and images
                    texts = []
                    tables = []
                    images = []

                    for chunk in chunks:
                        if "CompositeElement" in str(type(chunk)):
                            texts.append(chunk)
                            # Extract tables from composite elements
                            chunk_els = chunk.metadata.orig_elements
                            for el in chunk_els:
                                if "Table" in str(type(el)):
                                    tables.append(el)
                        elif "Table" in str(type(chunk)):
                            tables.append(chunk)

                    # Extract images from composite elements
                    for chunk in chunks:
                        if "CompositeElement" in str(type(chunk)):
                            chunk_els = chunk.metadata.orig_elements
                            for el in chunk_els:
                                if "Image" in str(type(el)):
                                    images.append(el.metadata.image_base64)

                    # Create vector store
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(here(vectordb_dir))
                    )

                    # Create storage for document store
                    store = InMemoryStore()
                    id_key = "doc_id"

                    # Add texts
                    if texts:
                        doc_ids = [str(uuid.uuid4()) for _ in texts]
                        text_docs = [
                            Document(page_content=text.text, metadata={id_key: doc_ids[i]}) 
                            for i, text in enumerate(texts)
                        ]
                        vectorstore.add_documents(text_docs)
                        store.mset(list(zip(doc_ids, texts)))

                    # Add tables
                    if tables:
                        # Create table summaries using LLM
                        prompt_table = """
                        You are an assistant tasked with summarizing tables.
                        Give a concise summary of the table.
                        Respond only with the summary, no additional comment.
                        Do not start your message by saying "Here is a summary" or anything like that.
                        Just give the summary as it is.

                        Table: {element}
                        """
                        table_prompt = ChatPromptTemplate.from_template(prompt_table)
                        model = ChatOpenAI(temperature=0.5, model=self.config['llm_config']['model'])
                        table_chain = {"element": lambda x: x} | table_prompt | model | StrOutputParser()
                        
                        tables_html = [table.metadata.text_as_html for table in tables]
                        table_summaries = table_chain.batch(tables_html, {"max_concurrency": 3})
                        
                        table_ids = [str(uuid.uuid4()) for _ in tables]
                        summary_tables = [
                            Document(page_content=summary, metadata={id_key: table_ids[i]}) 
                            for i, summary in enumerate(table_summaries)
                            if summary
                        ]
                        if summary_tables:
                            vectorstore.add_documents(summary_tables)
                            store.mset(list(zip(table_ids, tables)))

                    # Add images
                    if images:
                        # Create image summaries using LLM
                        prompt_image = """
                        You are an assistant tasked with describing images.
                        Give a concise description of what you see in the image.
                        Respond only with the description, no additional comment.
                        Do not start your message by saying "Here is a description" or anything like that.
                        Just give the description as it is.
                        For context, the image is part of a research paper explaining the transformers architecture.
                        Be specific about graphs, such as bar plots.
                        """
                        messages = [
                            (
                                "user",
                                [
                                    {"type": "text", "text": prompt_image},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                                    },
                                ],
                            )
                        ]
                        image_prompt = ChatPromptTemplate.from_messages(messages)
                        image_chain = image_prompt | ChatOpenAI(model=self.config['llm_config']['model']) | StrOutputParser()
                        image_summaries = image_chain.batch(images)

                        img_ids = [str(uuid.uuid4()) for _ in images]
                        summary_img = [
                            Document(page_content=summary, metadata={id_key: img_ids[i]}) 
                            for i, summary in enumerate(image_summaries)
                            if summary
                        ]
                        if summary_img:
                            vectorstore.add_documents(summary_img)
                            valid_images = [(id_, img) for id_, img in zip(img_ids, images) if img]
                            if valid_images:
                                store.mset(valid_images)

                    vectorstore.persist()
                    print(f"Processed file: {file_path}")
                    print(f"Number of vectors in collection: {vectorstore._collection.count()}\n")
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
        else:
            print(f"Directory '{vectordb_dir}' already exists for collection '{collection_name}'")

    def run(self) -> None:
        """
        Process all document collections defined in the configuration.
        Creates vector databases for each collection if they don't already exist.
        """
        collections = self.vector_config.get('collections', [])
        if not collections:
            print("No collections defined in the configuration")
            return

        for collection in collections:
            print(f"\nProcessing collection: {collection['name']}")
            self.process_collection(collection)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    # Initialize and run the vector database preparation
    prepare_db = PrepareVectorDB(config_path="configs/custom_config.yml")
    prepare_db.run()

