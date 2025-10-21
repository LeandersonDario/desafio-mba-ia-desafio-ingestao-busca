import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

PGVECTOR_URL = os.getenv("DATABASE_URL")
PGVECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
def search_prompt(question: str, contexto: str):
    """Gera resposta baseada no contexto usando LLM."""
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question))
    return response.content


# Cache global para reutilização de recursos
_vector_store_cache = None
_embeddings_cache = None

def _get_vector_store(collection_name: str = PGVECTOR_COLLECTION):
    """Retorna uma instância reutilizável do vector store."""
    global _vector_store_cache, _embeddings_cache
    
    if _embeddings_cache is None:
        _embeddings_cache = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    
    if _vector_store_cache is None or _vector_store_cache.collection_name != collection_name:
        _vector_store_cache = PGVector(
            collection_name=collection_name,
            embedding_function=_embeddings_cache,
            connection=PGVECTOR_URL,
            use_jsonb=True,
        )
    
    return _vector_store_cache

def _search_documents(query: str, k: int = 10, collection_name: str = PGVECTOR_COLLECTION, with_scores: bool = False):
    """Função base para busca de documentos com ou sem scores."""
    
    print(f"🔍 Buscando documentos similares para: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        vector_store = _get_vector_store(collection_name)
        
        if with_scores:
            results = vector_store.similarity_search_with_score(query, k=k)
        else:
            results = vector_store.similarity_search(query, k=k)
            
        print(f"✅ Encontrados {len(results)} documentos relevantes")
        return results, None
        
    except Exception as e:
        error_msg = f"Erro na busca: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg

def get_context_from_query(query: str, k: int = 10, collection_name: str = PGVECTOR_COLLECTION):
    """Busca contexto relevante concatenado para a query.
    
    Args:
        query: Pergunta do usuário
        k: Número de documentos a retornar (default: 10)
        collection_name: Nome da coleção no PGVector
        
    Returns:
        tuple: (contexto_concatenado, erro) onde erro é None se sucesso
    """
    
    results, error = _search_documents(query, k, collection_name, with_scores=False)
    
    if error:
        return None, f"Erro: Banco de dados não encontrado. Execute primeiro o script de ingestão."
    
    if not results:
        return "", None
    
    print("📝 Concatenando resultados para formar contexto...")
    context = "\n\n".join(doc.page_content for doc in results)
    
    return context.strip(), None

def search_similar_chunks(query: str, k: int = 10, collection_name: str = PGVECTOR_COLLECTION):
    """Busca chunks similares com scores de distância detalhados.
    
    Args:
        query: Pergunta do usuário
        k: Número de chunks a retornar (default: 10)
        collection_name: Nome da coleção no PGVector
        
    Returns:
        list: Lista de dicts com 'content', 'distance' e 'metadata'
    """
    
    results, error = _search_documents(query, k, collection_name, with_scores=True)
    
    if error or not results:
        return []
    
    print("📊 Formatando resultados com scores de similaridade...")
    similar_chunks = [
        {
            'content': doc.page_content,
            'distance': float(distance),
            'metadata': doc.metadata or {},
            'relevance_score': round(1 - distance, 4)  # Score de relevância (1 = mais relevante)
        }
        for doc, distance in results
    ]
    
    return similar_chunks

def main():
    """Função principal para teste de busca."""
    
    # Verifica se a API key está configurada
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Erro: GOOGLE_API_KEY não encontrada!")
        print("Por favor, configure sua API key no arquivo .env")
        return
    
    # Exemplo de busca
    query = "faturamento"
    print(f"Buscando por: '{query}'")
    
    try:
        results = search_similar_chunks(query, k=10)
        
        if results:
            print(f"\n✅ Encontrados {len(results)} resultados:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Resultado {i} (Distância: {result['distance']:.4f}) ---")
                print(f"Conteúdo: {result['content'][:200]}...")
                if result['metadata']:
                    print(f"Metadata: {result['metadata']}")
        else:
            print("❌ Nenhum resultado encontrado")
            
    except Exception as e:
        print(f"❌ Erro durante a busca: {e}")

if __name__ == "__main__":
    main()

