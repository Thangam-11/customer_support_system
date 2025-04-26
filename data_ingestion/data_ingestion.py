from langchain_community.vectorstores import AstraDBVectorStore
from dotenv import load_dotenv
import os
from data_ingestion.data_transform import DataConverter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")


class IngestData:
    def __init__(self):
        print("IngestData class initialized...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.data_converter = DataConverter()

    def data_ingestion(self, storage=None):
        vstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )

        if storage is None:
            docs = self.data_converter.data_transformation()
            inserted_ids = vstore.add_documents(docs)
            print(f"Documents inserted with IDs: {inserted_ids}")
            return vstore, inserted_ids
        else:
            return vstore


if __name__ == '__main__':
    ingestion = IngestData()
    ingestion.data_ingestion()
