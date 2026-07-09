from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.ingestion.models import MetadadosChunk
from langchain_core.output_parsers import PydanticOutputParser

# AgenteContexto - Extrai metadados estruturados de chunks usando LLM com validação Pydantic
class AgenteContexto:
    def __init__(self, model_name="qwen2.5vl:3b"):
        """
        Inicializa o Agente de Contexto e Metadados.
        """
        # Aqui precisamos do ChatOllama pois modelos de Chat lidam melhor com JSON
        self.llm = ChatOllama(
            model=model_name,
            base_url="http://ollama:11434",
            temperature=0.1
        )
        
        # Força o LangChain a usar a nossa classe Pydantic como "molde"
        self.parser = PydanticOutputParser(pydantic_object=MetadadosChunk)
        
        self.prompt = PromptTemplate(
            template="""Você é um bibliotecário especialista em classificar documentos técnicos.
            Leia o trecho de texto abaixo e extraia os metadados solicitados.
            
            TEXTO:
            {texto_limpo}
            
            INSTRUÇÕES DE FORMATAÇÃO:
            {format_instructions}
            """,
            input_variables=["texto_limpo"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def gerar_metadados(self, texto_limpo: str) -> dict:
        """
        Lê o texto limpo e gera os metadados estruturados.
        """
        print("   -> [Agente Contexto] Gerando metadados estruturados...")
        
        _input = self.prompt.format_prompt(texto_limpo=texto_limpo)
        
        try:
            # O .bind(format="json") força o Ollama a cuspir JSON puro
            resposta_bruta = self.llm.bind(format="json").invoke(_input.to_messages())
            
            # Converte a string JSON de volta para um dicionário Python validadado
            metadados_validados = self.parser.parse(resposta_bruta.content)
            return metadados_validados.model_dump()
            
        except Exception as e:
            print(f"      [Erro no Agente de Contexto]: Falha ao estruturar o JSON. {e}")
            # Fallback seguro caso o LLM alucine
            return {
                "tema_principal": "Tema Indefinido",
                "resumo": "Falha ao gerar resumo.",
                "eh_conteudo_critico": False
            }