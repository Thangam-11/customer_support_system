import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from exceptions.exception import CustomerSupportBotException
from config.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv


class Retriever:
    def __init__(self):
        try:
            self.model_loader = ModelLoader()
            self.config = load_config()
            self._load_env_variables()
            self.vstore = None
            self.retriever = None
        except Exception as e:
            raise CustomerSupportBotException(f"Error during Retriever initialization", e)

    def _load_env_variables(self):
        try:
            load_dotenv()
            required_vars = ["OPENAI_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
            missing_vars = [var for var in required_vars if os.getenv(var) is None]
            if missing_vars:
                raise CustomerSupportBotException(f"Missing environment variables: {missing_vars}")

            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
            self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        except Exception as e:
            raise CustomerSupportBotException(f"Error loading environment variables", e)

    def load_retriever(self):
        try:
            if not self.vstore:
                collection_name = self.config["astra_db"]["collection_name"]

                self.vstore = AstraDBVectorStore(
                    embedding=self.model_loader.load_embeddings(),
                    collection_name=collection_name,
                    api_endpoint=self.db_api_endpoint,
                    token=self.db_application_token,
                    namespace=self.db_keyspace,
                    # Removed create_collection param because it's invalid
                )

            if not self.retriever:
                top_k = self.config.get("retriever", {}).get("top_k", 3)
                self.retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
                print("Retriever loaded successfully.")

            return self.retriever

        except Exception as e:
            # Pass both args to your custom exception, as required
            raise CustomerSupportBotException("Error loading retriever", e)

    def call_retriever(self, query: str) -> List[Document]:
        try:
            retriever = self.load_retriever()
            output = retriever.invoke(query)
            return output
        except Exception as e:
            raise CustomerSupportBotException(f"Error retrieving results for query '{query}'", e)


if __name__ == '__main__':
    try:
        retriever_obj = Retriever()
        user_query = "Can you suggest good budget laptops?"
        results = retriever_obj.call_retriever(user_query)

        for idx, doc in enumerate(results, 1):
            print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")

    except CustomerSupportBotException as cse:
        print(f"Custom Exception: {cse}")
    except Exception as e:
        print(f"Unhandled Exception: {e}")
