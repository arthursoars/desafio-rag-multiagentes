import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

# Carrega as variáveis de ambiente (O nosso túnel)
load_dotenv()

class EmbeddingGenerator:
    # 🏆 ATUALIZADO: Usando o Qwen3-Embedding do Servidor (0.6b para velocidade)
    def __init__(self, model_name="qwen3-embedding:0.6b"):
        """
        Inicializa o gerador de embeddings (vetores).
        """
        self.model_name = model_name
        
        # Agora ele olha para o .env e usa o túnel do Termius
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Embeddings inicializados em: {self.base_url}")

    def obter_gerador(self) -> OllamaEmbeddings:
        """
        Retorna a função de embedding conectada ao servidor remoto de forma limpa e direta.
        """
        print(f"-->[Embeddings] Conectando ao modelo '{self.model_name}' via Servidor...")
        return OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url
        )