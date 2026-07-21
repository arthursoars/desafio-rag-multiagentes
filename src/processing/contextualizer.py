import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.ingestion.models import MetadadosChunk
from langchain_core.output_parsers import PydanticOutputParser

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class AgenteContexto:
    # Atualizado definitivamente para o modelo de 7B que o Eric deixou no servidor
    def __init__(self, model_name="qwen2.5vl:7b"):
        """
        Inicializa o Agente de Contexto e Metadados.
        """
        # Aqui precisamos do ChatOllama pois modelos de Chat lidam melhor com JSON
        self.llm = ChatOllama(
            model=model_name,
            # Busca a URL no .env apontando pro servidor (http://187.45.180.38:11434)
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1
        )
        
        # Força o LangChain a usar a nossa classe Pydantic como "molde"
        self.parser = PydanticOutputParser(pydantic_object=MetadadosChunk)
        
        # Adicionamos as "regras de mão de ferro" no prompt para domar o modelo
        self.prompt = PromptTemplate(
            template="""Você é um bibliotecário especialista em classificar documentos técnicos.
            Leia o trecho de texto abaixo e extraia os metadados solicitados.
            
            REGRAS CRÍTICAS DE FORMATAÇÃO (LEIA COM ATENÇÃO):
            1. Você DEVE retornar APENAS um objeto JSON válido, preenchido com as informações reais extraídas do texto.
            2. NÃO copie a estrutura de propriedades (schema) de volta para mim. Preencha os valores!
            3. O campo 'palavras_chave' deve ser uma lista estrita de strings (ex: ["regra", "prazo", "bolsa"]).
            4. O campo 'eh_conteudo_critico' deve ser booleano (true ou false).
            
            TEXTO:
            {texto_limpo}
            
            INSTRUÇÕES DO SCHEMA (PREENCHA ESTES CAMPOS):
            {format_instructions}
            
            Lembre-se: Retorne APENAS o JSON preenchido, sem explicações e sem repetir o schema.
            """,
            input_variables=["texto_limpo"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def gerar_metadados(self, texto_limpo: str) -> dict:
        """
        Lê o texto limpo e gera os metadados estruturados.
        """
        print(f"   -> [Agente Contexto] Gerando metadados via Servidor AceleraI...")
        
        _input = self.prompt.format_prompt(texto_limpo=texto_limpo)
        
        try:
            # O .bind(format="json") força o Ollama a cuspir JSON puro
            resposta_bruta = self.llm.bind(format="json").invoke(_input.to_messages())
            
            # Converte a string JSON de volta para um dicionário Python validado
            metadados_validados = self.parser.parse(resposta_bruta.content)
            return metadados_validados.model_dump()
            
        except Exception as e:
            print(f"      [Erro no Agente de Contexto]: Falha ao estruturar o JSON. {e}")
            return {
                "tema_principal": "Tema Indefinido",
                "resumo": "Falha ao extrair o contexto. Verifique o documento original.",
                "categoria_documento": "Indefinido",
                "raciocinio_classificacao": "Falha técnica na geração (Fallback ativado).",
                "eh_conteudo_critico": False,
                "palavras_chave": ["erro", "indefinido"]
            }