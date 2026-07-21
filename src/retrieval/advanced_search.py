import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from src.embeddings.embedding_generator import EmbeddingGenerator

# Carrega as variáveis de ambiente
load_dotenv()

# MotorBuscaAvancado - Recupera chunks relevantes usando HyDE e pré-filtros de metadados críticos
class MotorBuscaAvancado:
    def __init__(self, collection_name="documentos_tecnicos", model_llm="qwen2.5vl:7b"):
        """
        Método Construtor (POO): Inicializa as conexões com o banco Qdrant,
        e instancia os modelos conectados via túnel.
        """
        print("\n 🏗️ [Construção] Montando o Motor de Busca Avançado...")
        self.collection_name = collection_name
        self.qdrant_url = "http://qdrant:6333"
        
        # 1. Abre a conexão direta com o cliente do Qdrant
        print(" 🔌 [Conexão] Conectando ao banco vetorial Qdrant...")
        self.client = QdrantClient(url=self.qdrant_url)
        
        # 🏆 ATUALIZADO: Chama a classe de embeddings que agora aponta pro Qwen3
        gerador = EmbeddingGenerator()
        self.embedding_function = gerador.obter_gerador()
        
        # Configura LLM em modo determinístico para HyDE (Apontando pro .env)
        print(" 🔌 [Conexão] Conectando ao modelo LLM para técnica HyDE...")
        self.llm = OllamaLLM(
            model=model_llm,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0
        )
        
        # 4. Conecta o LangChain ao Qdrant para facilitar manipulações futuras
        print(" 🔗 [LangChain] Vinculando Qdrant ao motor de Embeddings...\n")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function
        )

    def _gerar_documento_hipotetico_hyde(self, pergunta_formal: str) -> str:
        """
        Método Interno/Privado (POO): Implementa a técnica HyDE (Hypothetical Document Embeddings).
        """
        print(" -> [HyDE] Projetando uma resposta hipotética ideal para a busca...")
        prompt_hyde = f"""Escreva um parágrafo estritamente técnico e factual que responderia perfeitamente à pergunta abaixo.
Não use introduções. Vá direto ao ponto como se fosse um trecho extraído de um documento oficial.

PERGUNTA: {pergunta_formal}
RESPOSTA HIPOTÉTICA FOCADA:"""
        
        resposta_ficticia = self.llm.invoke(prompt_hyde)
        return resposta_ficticia.strip()

    def recuperar_chunks_elite(self, pergunta_formal: str, top_k=10) -> list:
        """
        Método de Ação (POO): Aplica HyDE, Metadata Pre-filtering e Busca Híbrida.
        """
        texto_busca_hyde = self._gerar_documento_hipotetico_hyde(pergunta_formal)
        
        # Passo 2: Transforma o texto do HyDE em vetores matemáticos
        print(" 🧮 [Matemática] Convertendo a resposta do HyDE em vetores numéricos (Embeddings)...")
        vetor_busca = self.embedding_function.embed_query(texto_busca_hyde)
        
        print(" 🛡️ [Filtro Qdrant] Criando barreira lógica para buscar apenas 'Conteúdos Críticos'...")
        filtro_critico = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.eh_conteudo_critico",
                    match=qdrant_models.MatchValue(value=True) 
                )
            ]
        )
        
        print(f" 🔍 [Qdrant] Disparando busca vetorial no banco (Buscando Top {top_k})...")
        resultados_brutos = self.client.query_points(
            collection_name=self.collection_name,
            query=vetor_busca, 
            query_filter=filtro_critico,
            limit=top_k
        )
        
        return resultados_brutos.points

    def aplicar_reranking(self, pergunta_original: str, pontos_qdrant: list, top_n=3) -> list:
        """
        Método de Ação (POO): Reranker baseado no cruzamento direto das palavras.
        """
        print(f" -> [Reranking] Reclassificando os {len(pontos_qdrant)} chunks recuperados...")
        
        if not pontos_qdrant:
            return []
            
        chunks_avaliados = []
        for ponto in pontos_qdrant:
            texto_chunk = ponto.payload.get("page_content", "")
            metadata_chunk = ponto.payload.get("metadata", {})
            
            palavras_pergunta = set(pergunta_original.lower().split())
            palavras_chunk = set(texto_chunk.lower().split())
            interseccao = len(palavras_pergunta.intersection(palavras_chunk))
            
            score_final = ponto.score + (interseccao * 0.1)
            
            chunks_avaliados.append({
                "texto": texto_chunk,
                "metadata": metadata_chunk,
                "score_relevancia": score_final
            })
            
        chunks_reordenados = sorted(chunks_avaliados, key=lambda x: x["score_relevancia"], reverse=True)
        
        print(f" ⚖️ [Reranking] Sucesso! Chunk campeão reposicionado no topo.")
        return chunks_reordenados[:top_n]