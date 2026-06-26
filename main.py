import time
from src.ingestion.orchestrator import Orchestrator

# Marca o tempo de início para calcular duração do processamento
inicio = time.time()

# Cria o Orchestrator e processa os arquivos na pasta 'docs'
orq = Orchestrator()
documentos = orq.processar_lote("docs")

fim = time.time()

print("\n" + "=" * 50)
print("PROCESSAMENTO DE DOCUMENTOS")
print("=" * 50)

# Exibe a lista de arquivos que falharam no processamento (se houver)
if orq.erros:
    print(f"\nErros durante processamento ({len(orq.erros)}):")
    for erro in orq.erros:
        print(f" - {erro}")

# Resumo dos documentos processados com sucesso
print(f"\nDocumentos processados com sucesso ({len(documentos)}):")
for doc in documentos:
    print(f"\nArquivo: {doc.nome}")
    print(f"Páginas: {doc.paginas}")
    print(f"Caracteres: {doc.caracteres:,}")

print("\n" + "=" * 50)
print(f"Total processado: {len(documentos)} documentos")
print(f"Tempo de execução: {fim - inicio:.2f}s")
print("=" * 50)

# Imprime uma visualização inicial de cada documento processado
for i, doc in enumerate(documentos, start=1):
    print(f"\n[{i}] {doc.nome}")
    print(f"Páginas: {doc.paginas} | Caracteres: {doc.caracteres:,}")
    print("-" * 50)
    print(f'Resumo Inicial: \n{doc.texto[:3000]}...')