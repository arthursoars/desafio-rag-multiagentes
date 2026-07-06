from dataclasses import dataclass
from pydantic import BaseModel, Field

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
    
# --- NOVAS CLASSES PARA O LANGCHAIN (PYDANTIC) ---

class MetadadosChunk(BaseModel):
    """
    Estrutura obrigatória que o LLM deve preencher para cada chunk.
    Preparada para Chain of Thought e Metadata Pre-filtering.
    """
    tema_principal: str = Field(
        description="O tema central ou assunto principal deste trecho em até 5 palavras."
    )
    resumo: str = Field(
        description="Um resumo conciso de no máximo 2 frases sobre o conteúdo do trecho."
    )
    categoria_documento: str = Field(
        description="Classifique a origem deste texto em uma destas opções: 'Edital', 'Artigo Acadêmico', 'Relatório Financeiro', 'Prova/Avaliação' ou 'Outro'."
    )
    raciocinio_classificacao: str = Field(
        description="Explique passo a passo o seu raciocínio (Chain of Thought) para decidir se o conteúdo é crítico ou não. Detalhe os motivos."
    )
    eh_conteudo_critico: bool = Field(
        description="Retorne True se o trecho contiver regras, prazos, penalidades ou definições críticas. False caso contrário."
    )