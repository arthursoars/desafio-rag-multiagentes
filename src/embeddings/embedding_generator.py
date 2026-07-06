from langchain_ollama import OllamaEmbeddings

class EmbeddingGenerator:
    def __init__(self, model_name="bge-m3"):
        """
        Inicializa o gerador de embeddings apontando para o nosso contêiner do Ollama.
        Usando o bge-m3: O melhor modelo open-source atual para português.
        """
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url="http://ollama:11434" # Aponta para o contêiner do Docker
        )

    def obter_gerador(self):
        """
        Retorna a instância do gerador para ser injetada no Banco Vetorial.
        """
        return self.embeddings