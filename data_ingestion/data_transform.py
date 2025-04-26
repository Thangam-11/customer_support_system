import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self):
        print("DataConverter initialized...")
        self.product_data = pd.read_csv(r"C:\Users\thang\Desktop\customer_support_project\data\amazon_product_review.csv")
        print(self.product_data.head())

    def data_transformation(self):
        required_columns = list(self.product_data.columns[1:])
        
        product_list = []
        for index, row in self.product_data.iterrows():
            product = {
                "product_name": row['product_title'],
                "product_rating": row['rating'],
                "product_summary": row['summary'],
                "product_review": row['review']
            }
            product_list.append(product)
        
        docs = []
        for entry in product_list:
            metadata = {
                "product_name": entry["product_name"],
                "product_rating": entry["product_rating"],
                "product_summary": entry["product_summary"]
            }
            doc = Document(page_content=entry["product_review"], metadata=metadata)
            docs.append(doc)
        
        return docs

if __name__ == '__main__':
    data_con = DataConverter()
    documents = data_con.data_transformation()
    print(documents[0])  # print one sample Document
