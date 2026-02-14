# Customer Support Chatbot

An AI-powered e-commerce customer support system built with FastAPI, LangChain, and AstraDB. This chatbot provides intelligent product recommendations and handles customer queries based on product reviews and ratings.

## 🚀 Features

- **Intelligent Product Recommendations**: AI-powered responses based on product data
- **Vector Database Integration**: Uses AstraDB for efficient similarity search
- **RESTful API**: FastAPI-based backend with web interface
- **Real-time Chat Interface**: Interactive web-based chat UI
- **Scalable Architecture**: Containerized deployment with Docker
- **CI/CD Pipeline**: Automated deployment to AWS ECR via GitHub Actions

## 🏗️ Architecture
<img width="3593" height="5160" alt="image" src="https://github.com/user-attachments/assets/8af8891d-35a6-4b05-ad68-f42595a1ce71" />

 

<img width="7010" height="3405" alt="image" src="https://github.com/user-attachments/assets/50df2657-15f7-47a5-9bab-727c4979c6ed" />


## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **AI/ML**: LangChain, OpenAI GPT, Google GenAI
- **Database**: AstraDB (Vector Database)
- **Frontend**: HTML/CSS/JavaScript with Jinja2 templates
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud**: AWS ECR

## 📋 Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerization)
- AstraDB account and database setup
- OpenAI API key or Google GenAI API key

## 🔧 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-support-chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
   ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
   ASTRA_DB_KEYSPACE=your_keyspace_name
   ```

5. **Prepare your data**
   
   Place your product review CSV file at:
   ```
   data/flipkart_product_review.csv
   ```
   
   The CSV should contain columns: `product_title`, `rating`, `summary`, `review`

6. **Run data ingestion**
   ```bash
   python data_ingestion/ingestion_pipeline.py
   ```

7. **Start the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

8. **Access the application**
   
   Open your browser and navigate to: `http://localhost:8000`

### Docker Deployment

1. **Build Docker image**
   ```bash
   docker build -t customer-support-bot .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 --env-file .env customer-support-bot
   ```

## 📊 Configuration

The application uses a configuration system that can be customized via `config/config_loader.py`. Key configurations include:

- **AstraDB Settings**: Collection name, connection parameters
- **Embedding Model**: Model selection and parameters  
- **Retriever Settings**: Top-k results for similarity search
- **LLM Settings**: Model selection and generation parameters

## 🔄 Data Ingestion Pipeline

The data ingestion pipeline processes product review data and stores it in AstraDB:

1. **Data Loading**: Reads product review CSV files
2. **Data Transformation**: Converts data into LangChain Document format
3. **Embedding Generation**: Creates vector embeddings using OpenAI/Google models
4. **Vector Storage**: Stores embeddings in AstraDB for similarity search

Run the ingestion pipeline:
```bash
python data_ingestion/ingestion_pipeline.py
```

## 🤖 API Usage

### Chat Endpoint

**POST** `/get`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Parameters**: `msg` (string) - User query
- **Response**: AI-generated response based on product data

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/get" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "msg=Can you recommend budget headphones?"
```

## 🚀 Deployment

### AWS ECR Deployment

The project includes automated CI/CD pipeline using GitHub Actions:

1. **Set up GitHub Secrets**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_SESSION_TOKEN`

2. **Push to main branch** - The workflow automatically:
   - Builds Docker image
   - Pushes to AWS ECR repository: `customer-support-system`

3. **Deploy from ECR** to your preferred AWS service (ECS, EKS, Lambda, etc.)

## 🧪 Testing

Test the configuration setup:
```bash
python test.py
```

Test individual components:
```bash
# Test retriever
python retriever/retrieval.py

# Test data ingestion
python data_ingestion/ingestion_pipeline.py
```

## 📁 Project Structure

```
customer-support-chatbot/
├── .github/
│   └── workflows/
│       ├── aws.yml                 # AWS ECR deployment pipeline
│       └── main.yaml               # Alternative deployment config
├── data/
│   └── flipkart_product_review.csv # Product review dataset
├── data_ingestion/
│   └── ingestion_pipeline.py       # Data processing and ingestion
├── retriever/
│   └── retrieval.py                # Vector similarity search
├── prompt_library/
│   └── prompt.py                   # LLM prompt templates
├── utils/
│   └── model_loader.py             # Model loading utilities
├── config/
│   └── config_loader.py            # Configuration management
├── exceptions/
│   └── exception.py                # Custom exception classes
├── templates/
│   └── chat.html                   # Web chat interface
├── static/                         # CSS, JS, images
├── main.py                         # FastAPI application
├── dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
└── setup.py                       # Package setup
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact: thangamani1128@gmail.com

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice chat interface
- [ ] Advanced analytics dashboard
- [ ] Integration with more e-commerce platforms
- [ ] Sentiment analysis for customer feedback
- [ ] Real-time inventory integration

---

**Built with ❤️ by Thangarasu**
