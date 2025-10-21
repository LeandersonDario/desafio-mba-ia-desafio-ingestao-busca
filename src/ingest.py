import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
PGVECTOR_URL = os.getenv("DATABASE_URL")
PGVECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_pdf(path: str):
    """Load a PDF file from the given path."""

    if not os.path.exists(path):
        print(f"❌ Erro: Arquivo PDF não encontrado em {path}")
        print("Por favor, adicione um arquivo PDF chamado 'document.pdf' na raiz do projeto")
        return
    
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f"Documento carregado com {len(documents)} páginas")
    return documents

def split_pdf(documents):
    """Split the text of the documents into smaller chunks."""

    if not documents:
        print("❌ Erro: Documentos não encontrados!")
        print("Por favor, carregue um documento PDF no arquivo .env")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("❌ Erro: Chunks não encontrados!")
        print("Por favor, divida o documento em chunks no arquivo .env")
        return

    print(f"Documento dividido em {len(chunks)} chunks")
    return chunks

def create_embeddings_and_store(chunks):
    """Creates embeddings and stores them in pgvector."""

    if not EMBEDDING_MODEL:
        print("❌ Erro: EMBEDDING_MODEL não encontrada!")
        print("Por favor, configure seu embedding model no arquivo .env")
        return
    
    if not GOOGLE_API_KEY:
        print("❌ Erro: GOOGLE_API_KEY não encontrada!")
        print("Por favor, configure sua API key no arquivo .env")
        return
    
    print("Inicializando embeddings do Google...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )    
    
    print("Criando e populando o vector store...")
    vector_store = PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=PGVECTOR_COLLECTION,
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )

    print(f"✅ Ingestão concluída! {len(chunks)} chunks foram inseridos no banco de dados.")
    return vector_store  
    
def ingest_pdf():
    """Ingest a PDF file, split it into chunks, create embeddings, and store them."""
    
    # Verifica se as variáveis de ambiente estão configuradas
    required_vars = ["PDF_PATH", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Erro: Variáveis de ambiente não configuradas:")
        for var in missing_vars:
            print(f"   - {var}")
        print("Por favor, configure as variáveis no arquivo .env")
        return
    
    try:
        documents = load_pdf(PDF_PATH)
        if not documents:
            return

        chunks = split_pdf(documents)
        if not chunks:
            return
            
        create_embeddings_and_store(chunks)

        print("\n🎉 Ingestão concluída com sucesso!")
        print(f"📊 Total de chunks processados: {len(chunks)}")
        print("💾 Dados armazenados com sucesso no PGVector.")

    except Exception as e:
        print(f"❌ Erro durante a ingestão: {e}")

if __name__ == "__main__":
    ingest_pdf()