import os
import time
from dotenv import load_dotenv
from src.ingestion.orchestrator import Orchestrator

# Carrega as variáveis do arquivo .env
load_dotenv()

# Busca o limite de documentos no .env. Se não tiver configurado, processa todos (None).
limite_env = os.getenv("BATCH_SIZE")
limite_pdf = int(limite_env) if limite_env and limite_env.isdigit() else None

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("INICIANDO PROCESSAMENTO MULTIAGENTES")
    print(f"Limite de Lote configurado: {limite_pdf if limite_pdf else 'Todos'}")
    print("=" * 50)

    # Marca o tempo de início para calcular a duração do processamento
    inicio = time.time()

    # Cria o Orchestrator passando o limite puxado dinamicamente do .env
    orq = Orchestrator(limite_docs=limite_pdf)
    
    # Executa a nossa esteira completa (Tarefa 1 + Tarefa 2)
    orq.executar_pipeline()

    # Marca o tempo final
    fim = time.time()

    print("\n" + "=" * 50)
    print("RESUMO DA EXECUÇÃO")
    print("=" * 50)
    print(f"Tempo total de execução: {fim - inicio:.2f} segundos")
    print("=" * 50)