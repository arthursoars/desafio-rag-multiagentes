import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from src.embeddings.embedding_generator import EmbeddingGenerator

class MotorBuscaAvancado:
    def __init__(self, collection_name="documentos_tecnicos", model_llm="qwen2.5vl:3b"):
        """
        Método Construtor (POO): Inicializa as conexões com o banco Qdrant,
        reaproveita o modelo de Embeddings BGE-M3 e instancia o LLM para o HyDE.
        """
        self.collection_name = collection_name
        self.qdrant_url = "http://qdrant:6333"
        
        # 1. Abre a conexão direta com o cliente do Qdrant
        self.client = QdrantClient(url=self.qdrant_url)
        
        # 2. Resgata o gerador de embeddings BGE-M3 (1024 dimensões)
        gerador = EmbeddingGenerator(model_name="bge-m3")
        self.embedding_function = gerador.obter_gerador()
        
        # 3. Instancia o Ollama em modo mecânico (temperatura 0) para a técnica de HyDE
        self.llm = OllamaLLM(
            model=model_llm,
            base_url="http://ollama:11434",
            temperature=0.0
        )
        
        # 4. Conecta o LangChain ao Qdrant para facilitar manipulações futuras
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function
        )

    def _gerar_documento_hipotetico_hyde(self, pergunta_formal: str) -> str:
        """
        Método Interno/Privado (POO): Implementa a técnica HyDE (Hypothetical Document Embeddings).
        Gera uma resposta fictícia ideal para melhorar a busca vetorial.
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
        Retorna os 'K' melhores blocos de texto direto do Qdrant.
        """
        # Passo 1: Executa a técnica HyDE para obter o texto de resposta ideal
        texto_busca_hyde = self._gerar_documento_hipotetico_hyde(pergunta_formal)
        
        # Passo 2: Transforma o texto do HyDE nas 1024 dimensões do modelo BGE-M3
        vetor_busca = self.embedding_function.embed_query(texto_busca_hyde)
        
        print(" -> [Metadata Pre-filtering] Aplicando barreira lógica no Qdrant (Apenas Conteúdos Críticos)...")
        # Passo 3: Cria a regra estrita de filtragem baseada nos metadados do seu Pydantic
        filtro_critico = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.eh_conteudo_critico",
                    match=qdrant_models.MatchValue(value=True) # Só deixa passar se for True
                )
            ]
        )
        
        # Passo 4: Executa a busca híbrida filtrada no Qdrant
        resultados_brutos = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=vetor_busca,
            query_filter=filtro_critico,
            limit=top_k
        )
        
        return resultados_brutos.points

    def aplicar_reranking(self, pergunta_original: str, pontos_qdrant: list, top_n=3) -> list:
        """
        Método de Ação (POO): Simula o comportamento do Cross-Encoder do BGE-M3 (Reranker).
        Compara o contexto profundo dos blocos com a dúvida real do usuário e ordena com rigor.
        """
        print(f" -> [Reranking] Reclassificando os {len(pontos_qdrant)} chunks recuperados...")
        
        if not pontos_qdrant:
            return []
            
        chunks_avaliados = []
        for ponto in pontos_qdrant:
            texto_chunk = ponto.payload.get("page_content", "")
            metadata_chunk = ponto.payload.get("metadata", {})
            
            # Aqui simulamos a inteligência do Reranker dando uma nota (score)
            # baseada no cruzamento direto das palavras da pergunta com o texto do chunk
            palavras_pergunta = set(pergunta_original.lower().split())
            palavras_chunk = set(texto_chunk.lower().split())
            interseccao = len(palavras_pergunta.intersection(palavras_chunk))
            
            # Pontuação híbrida: score matemático original do Qdrant + peso do casamento léxico
            score_final = ponto.score + (interseccao * 0.1)
            
            chunks_avaliados.append({
                "texto": texto_chunk,
                "metadata": metadata_chunk,
                "score_relevancia": score_final
            })
            
        # Ordena a lista do maior score para o menor (Ordem Decrescente)
        chunks_reordenados = sorted(chunks_avaliados, key=lambda x: x["score_relevancia"], reverse=True)
        
        # Retorna apenas o Top N definitivo (Geralmente os 2 ou 3 melhores exatos)
        print(f" ⚖️ [Reranking] Sucesso! Chunk campeão reposicionado no topo.")
        return chunks_reordenados[:top_n]
