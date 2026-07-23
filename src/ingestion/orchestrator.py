import os
import glob
import shutil # NOVA IMPORTAÇÃO: Serve para mover arquivos fisicamente
from src.ingestion.parser import Parser
from src.processing.chunker import Chunker
from src.processing.cleaner import AgenteLimpeza
from src.processing.contextualizer import AgenteContexto
from src.vectorstore.qdrant_store import AgenteIndexacao

# Orchestrator - Coordena todo pipeline: leitura de PDFs, limpeza, contextualização e indexação
class Orchestrator:
    # O orquestrador agora olha para a pasta PENDENTES
    def __init__(self, pasta_base="./docs", limite_docs=None):
        # Configuração das 3 pastas
        self.pasta_base = pasta_base
        self.pasta_pendentes = os.path.join(self.pasta_base, "pendentes")
        self.pasta_processados = os.path.join(self.pasta_base, "processados")
        self.pasta_quarentena = os.path.join(self.pasta_base, "quarentena")
        
        self.limite_docs = limite_docs
        self.total_pdfs_lidos = 0
        self.pdfs_com_falha = []
        
        # Cria as pastas fisicamente no Windows/Mac se elas não existirem
        os.makedirs(self.pasta_pendentes, exist_ok=True)
        os.makedirs(self.pasta_processados, exist_ok=True)
        os.makedirs(self.pasta_quarentena, exist_ok=True)
        
        print("Iniciando componentes da Arquitetura Multiagentes...")
        self.parser = Parser()
        self.chunker = Chunker(chunk_size=1000, chunk_overlap=200)
        self.agente_limpeza = AgenteLimpeza()
        self.agente_contexto = AgenteContexto()
        self.agente_indexacao = AgenteIndexacao()

    def mover_arquivo(self, caminho_origem, pasta_destino):
        """Função auxiliar para mover o PDF com segurança de uma pasta para outra."""
        try:
            nome_arquivo = os.path.basename(caminho_origem)
            caminho_destino = os.path.join(pasta_destino, nome_arquivo)
            
            # Se já existir um arquivo com o mesmo nome no destino, remove para sobrescrever
            if os.path.exists(caminho_destino):
                os.remove(caminho_destino)
                
            shutil.move(caminho_origem, caminho_destino)
            print(f" 📂 Arquivo movido para: {pasta_destino}")
        except Exception as e:
            print(f" ❌ Erro ao tentar mover arquivo {nome_arquivo}: {e}")

    def executar_pipeline(self):
        # A busca agora acontece APENAS na pasta pendentes
        arquivos_pdf = glob.glob(os.path.join(self.pasta_pendentes, "*.pdf"))
        
        if self.limite_docs:
            arquivos_pdf = arquivos_pdf[:self.limite_docs]
            
        self.total_pdfs_lidos = len(arquivos_pdf)
            
        if not arquivos_pdf:
            print(f"Nenhum PDF encontrado na pasta {self.pasta_pendentes}. Todos já foram processados!")
            return

        print(f"Encontrados {len(arquivos_pdf)} documentos pendentes. Iniciando a esteira...")

        for caminho_pdf in arquivos_pdf:
            nome_arquivo = os.path.basename(caminho_pdf)
            print("\n" + "="*60)
            print(f"📄 Processando Documento: {nome_arquivo}")
            
            # Flag para saber se o arquivo quebrou no meio do caminho
            falhou_no_processamento = False
            
            # --- TAREFA 1: INGESTÃO BRUTA ---
            documento = self.parser.ler_pdf(caminho_pdf)
            
            if not documento or not documento.texto:
                print(f"⚠️ Aviso: Falha crítica na extração bruta de {nome_arquivo}")
                self.pdfs_com_falha.append(nome_arquivo) 
                
                # MOVE PARA QUARENTENA
                self.mover_arquivo(caminho_pdf, self.pasta_quarentena)
                continue # Pula para o próximo PDF
                
            # --- TAREFA 2: PROCESSAMENTO MULTIAGENTES ---
            print("🔪 Fatiando o documento bruto...")
            chunks_brutos = self.chunker.fatiar_documento(documento)
            
            # Verificação de segurança: se o chunker falhar e não gerar nada
            if not chunks_brutos:
                print(f"⚠️ Aviso: O fatiador não conseguiu gerar chunks para {nome_arquivo}")
                self.pdfs_com_falha.append(nome_arquivo)
                self.mover_arquivo(caminho_pdf, self.pasta_quarentena)
                continue

            print(f"   -> Obtidos {len(chunks_brutos)} chunks para processamento.")
            
            # Fluxo sequencial dos 3 agentes para cada chunk
            try:
                for i, chunk_dict in enumerate(chunks_brutos):
                    if self.agente_indexacao.verificar_se_existe(nome_arquivo, i):
                        print(f"⏩ [Chunk {i+1}/{len(chunks_brutos)}] Já indexado no Qdrant. Pulando...")
                        continue 
                    
                    texto_bruto = chunk_dict["texto"]
                    metadados_iniciais = chunk_dict["metadata"]
                    
                    texto_limpo = self.agente_limpeza.limpar_chunk(texto_bruto)
                    if not texto_limpo:
                        continue
                        
                    metadados_gerados = self.agente_contexto.gerar_metadados(texto_limpo)
                    pacote_metadados = {**metadados_iniciais, **metadados_gerados}
                    
                    self.agente_indexacao.indexar_pacote(texto_limpo, pacote_metadados)
            
            except Exception as e:
                print(f"❌ Erro grave no processamento multiagentes de {nome_arquivo}: {e}")
                falhou_no_processamento = True

            # --- TAREFA 3: MOVIMENTAÇÃO DE PASTAS ---
            if falhou_no_processamento:
                self.pdfs_com_falha.append(nome_arquivo)
                self.mover_arquivo(caminho_pdf, self.pasta_quarentena)
            else:
                print(f" ✨ Concluído! O processamento do documento '{nome_arquivo}' foi encerrado com sucesso.\n")        
                self.mover_arquivo(caminho_pdf, self.pasta_processados)

        print("\n" + "="*60)
        print("✅ PIPELINE MULTIAGENTES CONCLUÍDO COM SUCESSO!")
        print(f"Resumo: {self.total_pdfs_lidos - len(self.pdfs_com_falha)} com sucesso, {len(self.pdfs_com_falha)} com falha (movidos para quarentena).")