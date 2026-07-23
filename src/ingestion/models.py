import os
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# LEITURA DAS REGRAS EXTERNAS (Isolamento de Prompts)
# ---------------------------------------------------------
caminho_config = os.path.join(os.getcwd(), "config", "prompts_metadados.json")
try:
    with open(caminho_config, "r", encoding="utf-8") as arquivo:
        regras_metadados = json.load(arquivo)
except FileNotFoundError:
    print("❌ [ERRO] Arquivo config/prompts_metadados.json não encontrado!")
    regras_metadados = {} # Fallback: se não achar o arquivo, cria um dicionário vazio para não quebrar o código

# ---------------------------------------------------------
# CLASSES DE DADOS
# ---------------------------------------------------------

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
    
# MetadadosChunk - Valida e estrutura metadados de chunks
class MetadadosChunk(BaseModel):
    """
    Estrutura obrigatória que o LLM deve preencher para cada chunk.
    As descrições (prompts) são importadas do arquivo de configuração externo.
    """
    tema_principal: str = Field(
        description=regras_metadados.get("tema_principal", "Tema central do trecho.")
    )
    resumo: str = Field(
        description=regras_metadados.get("resumo", "Resumo conciso do trecho.")
    )
    categoria_documento: str = Field(
        description=regras_metadados.get("categoria_documento", "Categoria do documento.")
    )
    raciocinio_classificacao: str = Field(
        description=regras_metadados.get("raciocinio_classificacao", "Explicação do raciocínio.")
    )
    eh_conteudo_critico: bool = Field(
        description=regras_metadados.get("eh_conteudo_critico", "True se for crítico, False caso contrário.")
    )
    palavras_chave: list[str] = Field(
        description=regras_metadados.get("palavras_chave", "Lista de palavras-chave.")
    )