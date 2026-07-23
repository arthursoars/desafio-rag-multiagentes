import os
import time
from dotenv import load_dotenv
from src.ingestion.orchestrator import Orchestrator

# main - Inicializa o pipeline multiagentes com limite de lote configurável via .env
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
    print("==================================================")
    print(f"Tempo total de execução: {fim - inicio:.2f} segundos")
    print(f"Total de PDFs lidos: {getattr(orq, 'total_pdfs_lidos', 0)}")
    
    # EXIBE A LISTA DOS ARQUIVOS QUE FALHARAM TOTALMENTE
    pdfs_falhos = getattr(orq, 'pdfs_com_falha', [])
    print(f"Total de PDFs com falha: {len(pdfs_falhos)}")
    if pdfs_falhos:
        print("Arquivos que falharam na leitura bruta:")
        for falha in pdfs_falhos:
            print(f"  ❌ {falha}")

    if len(pdfs_falhos) == 0:
        print("Todos os PDFs da fila foram processados com sucesso!")
        
    # Busca a variável lá dentro do agente de indexação para mostrar o resultado
    try:
        total_chunks = orq.agente_indexacao.chunks_salvos_sessao
        print(f"Total de chunks indexados no banco: {total_chunks}")
    except AttributeError:
        pass # Ignora caso a variável ainda não tenha sido instanciada no orquestrador
        
    print("==================================================")