# Search Your PDF App

This project is a Streamlit-based application that allows users to upload a PDF file, process its content, and perform searches within the document using embeddings for contextual understanding. The app is built using LangChain, Chroma, and Streamlit, and was inspired by the [AIAnytime/Search-Your-PDF-App](https://github.com/AIAnytime/Search-Your-PDF-App/) repository and created with the assistance of ChatGPT.

## Features

- **PDF Upload and Ingestion**: Upload any PDF file to extract, split, and embed the content for contextual retrieval.
- **Embedded Search**: Allows you to search the uploaded PDF's contents using semantic embeddings.
- **Efficient Document Splitting**: Processes large documents by splitting content into chunks for efficient retrieval.
- **Vector Database Storage**: Uses Chroma as a vector store to store and retrieve embedded chunks of the document.

## Tech Stack

- **[LangChain](https://python.langchain.com/)**: For handling text processing, embedding, and creating chains for LLM interactions.
- **[Chroma](https://www.trychroma.com/)**: Vector database to store and retrieve embeddings.
- **[Streamlit](https://streamlit.io/)**: For creating an interactive and user-friendly interface.

## Installation

### Prerequisites

- Python 3.10 or above
- Dependencies (listed in `requirements.txt`)

### Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/deepMB/Search-In-Your-PDF.git
    cd Search-Your-PDF-App
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file**:
    - Include any necessary API keys for your model or vector store if required. For instance:
      ```env
      # Example for environment variables
      GOOGLE_API_Key=your_google_api_key
      ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

5. **Upload a PDF file**:
    - In the Streamlit interface, upload a PDF file and enter a query to search within the document.

## Usage

1. **Upload a PDF**: Use the file uploader in the app interface to upload a PDF document.
2. **Ingest and Process Document**: The app will process and store the document in a Chroma vector database.
3. **Search Within the PDF**: Type in a query related to the content of the PDF, and the app will display relevant results from the document.

## Example Code

The core functions in this project include:

- `ingest_doc(file)`: Loads, splits, and embeds the PDF content.
- `search_query(db, query)`: Retrieves relevant content from the document based on the user's query.

## Credits

This project is inspired by the [AIAnytime/Search-Your-PDF-App](https://github.com/AIAnytime/Search-Your-PDF-App/) and was developed with the assistance of [ChatGPT](https://chat.openai.com/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you encounter issues or have suggestions for improvement, please create an issue or submit a pull request.
