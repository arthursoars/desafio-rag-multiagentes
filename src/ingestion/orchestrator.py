import os
import glob
from src.ingestion.parser import Parser
from src.processing.chunker import Chunker
from src.processing.cleaner import AgenteLimpeza
from src.processing.contextualizer import AgenteContexto
from src.vectorstore.qdrant_store import AgenteIndexacao

# Orchestrator - Coordena todo pipeline: leitura de PDFs, limpeza, contextualização e indexação
class Orchestrator:
    # O limite de documentos em lote (BATCH_SIZE) para testes
    def __init__(self, pasta_docs="./docs", limite_docs=None):
        self.pasta_docs = pasta_docs
        self.limite_docs = limite_docs
        
        print("Iniciando componentes da Arquitetura Multiagentes...")
        self.parser = Parser()
        self.chunker = Chunker(chunk_size=1000, chunk_overlap=200)
        
        # Inicializa agentes multiagentes
        self.agente_limpeza = AgenteLimpeza()
        self.agente_contexto = AgenteContexto()
        self.agente_indexacao = AgenteIndexacao()

    def executar_pipeline(self):
        arquivos_pdf = glob.glob(os.path.join(self.pasta_docs, "*.pdf"))
        
        # Aplica o limite de lote se configurado (Requisito: Controle de Lote)
        if self.limite_docs:
            arquivos_pdf = arquivos_pdf[:self.limite_docs]
            
        if not arquivos_pdf:
            print(f"Nenhum PDF encontrado na pasta {self.pasta_docs}")
            return

        print(f"Encontrados {len(arquivos_pdf)} documentos. Iniciando a esteira assíncrona...")

        for caminho_pdf in arquivos_pdf:
            nome_arquivo = os.path.basename(caminho_pdf)
            print("\n" + "="*60)
            print(f"📄 Processando Documento: {nome_arquivo}")
            
            # --- TAREFA 1: INGESTÃO BRUTA ---
            documento = self.parser.ler_pdf(caminho_pdf)
            
            if not documento or not documento.texto:
                print(f"⚠️ Aviso: Falha na extração bruta de {nome_arquivo}")
                continue
                
            # --- TAREFA 2: PROCESSAMENTO MULTIAGENTES ---
            print("🔪 Fatiando o documento bruto...")
            chunks_brutos = self.chunker.fatiar_documento(documento)
            print(f"   -> Obtidos {len(chunks_brutos)} chunks para processamento.")
            
            # Fluxo sequencial dos 3 agentes para cada chunk
            for i, chunk_dict in enumerate(chunks_brutos):
                # 🟢 ETAPA DA OTIMIZAÇÃO (POO):
                # Usamos o objeto agente_indexacao para verificar se o ID do chunk já existe no banco
                if self.agente_indexacao.verificar_se_existe(nome_arquivo, i):
                    print(f"⏩ [Chunk {i+1}/{len(chunks_brutos)}] Já indexado no Qdrant. Pulando esteira de IA...")
                    continue  # Abandona o processamento atual e salta para o próximo chunk instantaneamente
                
                # Se não existir, a esteira segue o fluxo normal de processamento
                texto_bruto = chunk_dict["texto"]
                metadados_iniciais = chunk_dict["metadata"]
                
                print(f"\n   [Chunk {i+1}/{len(chunks_brutos)}] Iniciando fluxo de agentes:")
                
                # Agente 1: Limpeza
                texto_limpo = self.agente_limpeza.limpar_chunk(texto_bruto)
                if not texto_limpo:
                    continue
                    
                # Agente 2: Contexto e Metadados Pydantic
                metadados_gerados = self.agente_contexto.gerar_metadados(texto_limpo)

                # Exibe os metadados gerados para conferência (opcional)
                import json
                print(f"      [JSON Estruturado]: {json.dumps(metadados_gerados, ensure_ascii=False, indent=2)}")
                
                # Mescla os metadados gerados pela IA com os metadados de origem (nome do arquivo, index)
                pacote_metadados = {**metadados_iniciais, **metadados_gerados}
                
                # Agente 3: Indexação
                self.agente_indexacao.indexar_pacote(texto_limpo, pacote_metadados)
                

        print("\n" + "="*60)
        print("✅ PIPELINE MULTIAGENTES CONCLUÍDO COM SUCESSO!")
        print("A Base de Conhecimento Vetorial está pronta.")

if __name__ == "__main__":
    # Podemos testar passando limite_docs=1 para rodar apenas o primeiro PDF mais rápido
    orquestrador = Orchestrator(limite_docs=1)
    orquestrador.executar_pipeline()
