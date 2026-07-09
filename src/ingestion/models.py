from dataclasses import dataclass
from pydantic import BaseModel, Field

# Documento - Armazena metadados e conteúdo extraído de PDFs
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
    
# MetadadosChunk - Valida e estrutura metadados como tema, resumo, categoria e criticalidade de chunks
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
        description=(
            "Identifique e extraia em até duas palavras a categoria exata deste documento com base no contexto técnico "
            "(Exemplos comuns: 'Edital', 'Artigo Acadêmico', 'Relatório Financeiro', 'Prova/Avaliação', 'Manual Técnico', 'Contrato Legal'). "
            "Seja preciso e use termos formais adequados para o tipo de arquivo lido."
        )
    )
    raciocinio_classificacao: str = Field(
        description="Explique passo a passo o seu raciocínio (Chain of Thought) para decidir se o conteúdo é crítico ou não. Detalhe os motivos."
    )
    eh_conteudo_critico: bool = Field(
        description="Retorne True se o trecho contiver regras, prazos, penalidades ou definições críticas. False caso contrário."
    )
    
    palavras_chave: list[str] = Field(
        description="Uma lista contendo exatamente 5 palavras-chave ou termos técnicos extraídos deste trecho, ideais para busca."
    )
