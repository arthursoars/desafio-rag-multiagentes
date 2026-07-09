import uuid
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.embeddings.embedding_generator import EmbeddingGenerator

# AgenteIndexacao - Armazena chunks vetorizados no Qdrant para recuperação semântica eficiente
class AgenteIndexacao:
    def __init__(self, collection_name="documentos_tecnicos"):
        gerador = EmbeddingGenerator(model_name="bge-m3")
        self.embedding_function = gerador.obter_gerador()
        self.collection_name = collection_name
        self.qdrant_url = "http://qdrant:6333"
        
        # 1. Cria o cliente para gerenciar a coleção
        self.client = QdrantClient(url=self.qdrant_url)
        
        # 2. Se a coleção não existir, cria ela com 1024 dimensões (tamanho do BGE-M3)
        if not self.client.collection_exists(self.collection_name):
            print(f"   -> [Agente Indexação] Criando nova coleção: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
            )

        # 3. Conecta o LangChain ao Qdrant
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function
        )

    def indexar_pacote(self, texto_limpo: str, metadados: dict):
        print("   -> [Agente Indexação] Salvando chunk no Qdrant...")
        nome_arquivo = metadados.get("source", "doc_desconhecido")
        indice_chunk = metadados.get("chunk_index", 0)
        id_unico = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{nome_arquivo}_{indice_chunk}"))
        
        try:
            self.vector_store.add_texts(
                texts=[texto_limpo],
                metadatas=[metadados],
                ids=[id_unico]
            )
            print(f"      ✅ Chunk {indice_chunk} salvo!")
        except Exception as e:
            print(f"      ❌ Erro ao salvar: {e}")

## Técnica para verificar se o chunk já existe no Qdrant antes de salvar, evitando duplicatas.
    def verificar_se_existe(self, nome_arquivo: str, indice_chunk: int) -> bool:
        """
        Método de Consulta (POO): Checa no Qdrant se o ID gerado pelo UUID5 já existe.
        Retorna True se o chunk já foi salvo antes, ou False caso contrário.
        """
        # 1. Gera o mesmo ID único determinístico baseado no arquivo e índice
        id_unico = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{nome_arquivo}_{indice_chunk}"))
        
        try:
            # 2. Pergunta diretamente ao Qdrant se esse ID específico existe na coleção
            resposta = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id_unico]
            )
            # 3. Se a resposta contiver dados, significa que o chunk já está indexado
            return len(resposta) > 0
        except Exception:
            # Se der qualquer erro (banco fora do ar ou tabela vazia), assume que não existe
            return False
   