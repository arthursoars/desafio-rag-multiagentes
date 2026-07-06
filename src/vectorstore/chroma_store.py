from langchain_chroma import Chroma
from src.embeddings.embedding_generator import EmbeddingGenerator

class AgenteIndexacao:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Inicializa o Agente de Indexação, conectando ao motor de Embeddings (BGE-M3)
        e preparando o banco vetorial local.
        """
        gerador = EmbeddingGenerator(model_name="bge-m3")
        self.embedding_function = gerador.obter_gerador()
        self.persist_directory = persist_directory

    def indexar_pacote(self, texto_limpo: str, metadados: dict):
        """
        Recebe o pacote final, gera o embedding e salva no ChromaDB com um ID único.
        """
        print("   -> [Agente Indexação] Gerando embedding (BGE-M3) e salvando no ChromaDB...")
        
        # Cria um ID determinístico baseado na origem e na posição do chunk
        nome_arquivo = metadados.get("source", "doc_desconhecido")
        indice_chunk = metadados.get("chunk_index", 0)
        id_unico = f"{nome_arquivo}_chunk_{indice_chunk}"
        
        try:
            Chroma.from_texts(
                texts=[texto_limpo],
                embedding=self.embedding_function,
                metadatas=[metadados],
                ids=[id_unico], # Injeção do ID único e explícito!
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"      [Erro no Agente de Indexação]: Falha ao salvar no banco. {e}")