import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from utils.model_loader import ModelLoader
from config.config_loader import load_config
from exceptions.exception import CustomerSupportBotException

load_dotenv()  # Ensure env vars are loaded before anything else

class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        try:
            print("Initializing DataIngestion pipeline...")
            self.model_loader = ModelLoader()
            self._load_env_variables()
            self.csv_path = self._get_csv_path()
            self.product_data = self._load_csv()
            self.config = load_config()
        except Exception as e:
            raise CustomerSupportBotException("Error during initialization", e)

    def _load_env_variables(self):
        try:
            load_dotenv()
            required_vars = ["OPENAI_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
            missing_vars = [var for var in required_vars if os.getenv(var) is None]
            if missing_vars:
                raise EnvironmentError(f"Missing environment variables: {missing_vars}")

            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
            self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        except Exception as e:
            raise CustomerSupportBotException("Error loading environment variables", e)

    def _get_csv_path(self):
        try:
            current_dir = os.getcwd()
            csv_path = os.path.join(current_dir, 'data', 'flipkart_product_review.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            return csv_path
        except Exception as e:
            raise CustomerSupportBotException("Error locating CSV file", e)

    def _load_csv(self):
        try:
            df = pd.read_csv(self.csv_path)
            expected_columns = {'product_title', 'rating', 'summary', 'review'}
            if not expected_columns.issubset(set(df.columns)):
                raise ValueError(f"CSV must contain columns: {expected_columns}")
            return df
        except Exception as e:
            raise CustomerSupportBotException("Error loading or validating CSV", e)

    def transform_data(self):
        try:
            product_list = []
            for _, row in self.product_data.iterrows():
                product_entry = {
                    "product_name": row['product_title'],
                    "product_rating": row['rating'],
                    "product_summary": row['summary'],
                    "product_review": row['review']
                }
                product_list.append(product_entry)

            documents = []
            for entry in product_list:
                metadata = {
                    "product_name": entry["product_name"],
                    "product_rating": entry["product_rating"],
                    "product_summary": entry["product_summary"]
                }
                doc = Document(page_content=entry["product_review"], metadata=metadata)
                documents.append(doc)

            print(f"Transformed {len(documents)} documents.")
            return documents
        except Exception as e:
            raise CustomerSupportBotException("Error transforming data", e)

    def store_in_vector_db(self, documents: List[Document]):
        try:
            collection_name = self.config["astra_db"]["collection_name"]
            vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )
            inserted_ids = vstore.add_documents(documents)
            print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
            return vstore, inserted_ids
        except Exception as e:
            raise CustomerSupportBotException("Error storing clsdocuments in AstraDB", e)

    def run_pipeline(self):
        try:
            documents = self.transform_data()
            vstore, inserted_ids = self.store_in_vector_db(documents)

            query = "Can you tell me the low budget headphone?"
            results = vstore.similarity_search(query)

            print(f"\nSample search results for query: '{query}'")
            for res in results:
                print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")
        except Exception as e:
            raise CustomerSupportBotException("Pipeline execution failed", e)

# Run if this file is executed directly
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        ingestion.run_pipeline()
    except CustomerSupportBotException as e:
        print(f"Custom Exception Caught: {e}")
