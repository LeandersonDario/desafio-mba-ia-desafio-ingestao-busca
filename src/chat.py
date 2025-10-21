import os
from dotenv import load_dotenv
from search import search_prompt, get_context_from_query

load_dotenv()

def main():
    """Função principal do chat CLI."""
    
    # Verifica se as variáveis de ambiente estão configuradas
    required_vars = ["GOOGLE_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Erro: Variáveis de ambiente não configuradas:")
        for var in missing_vars:
            print(f"   - {var}")
        print("Por favor, configure as variáveis no arquivo .env")
        return

    print("🤖 Chat iniciado! Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("-" * 50)

    while True:
        query = input("\nFaça sua pergunta: ").strip()
        
        if query.lower() in ['sair', 'exit', 'quit']:
            print("👋 Encerrando o chat. Até logo!")
            break
        
        if not query:
            continue
        
        try:
            print(f"\nPERGUNTA: {query}")
            
            context, error = get_context_from_query(query)
            
            if error:
                print(f"❌ {error}")
                continue
            
            if not context:
                print("RESPOSTA: Não tenho informações necessárias para responder sua pergunta.")
                continue
            
            chain = search_prompt(query, context)
            print(f"RESPOSTA: {chain}")
            
        except Exception as e:
            print(f"❌ Erro ao processar sua pergunta: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()