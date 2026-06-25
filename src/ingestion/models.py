from dataclasses import dataclass

# Modelo de dados para representar o documento lido pelo Parser
@dataclass
class Documento:
    nome: str
    caminho: str
    texto: str = ""
    paginas: int = 0

    # Retorna o número de caracteres do texto extraído
    @property
    def caracteres(self) -> int:
        return len(self.texto)