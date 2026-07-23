import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class AgenteGerador:
    def __init__(self):
        # 1. Puxando configurações da infraestrutura (Lendo o .env)
        modelo_llm = os.getenv("LLM_MODEL_NAME", "qwen2.5vl:7b")
        url_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        temperatura = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # Inicializando o Qwen no Servidor
        self.llm = ChatOllama(
            model=modelo_llm, 
            base_url=url_base,
            temperature=temperatura 
        )
        
        # 2. Lendo as regras de negócio de fora do código
        caminho_prompt = os.path.join(os.getcwd(), "config", "system_prompt.txt")
        
        try:
            with open(caminho_prompt, "r", encoding="utf-8") as arquivo:
                instrucoes_sistema = arquivo.read()
        except FileNotFoundError:
            print("❌ [ERRO] Arquivo config/system_prompt.txt não encontrado!")
            # Fallback (plano B) caso a pasta config não exista
            instrucoes_sistema = "Você é um assistente técnico. Responda com base no contexto."

        # O System Prompt (A Regra é injetada dinamicamente)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", instrucoes_sistema + "\n\nCONTEXTO DOS DOCUMENTOS:\n{contexto}"),
            ("human", "{pergunta}")
        ])

    def gerar_resposta_final(self, pergunta: str, chunks_recuperados: list) -> str:
        """
        Recebe a pergunta do usuário e os chunks do Qdrant, monta o Mega Prompt e gera a resposta.
        """
        # 1. Extrai apenas o texto útil dos chunks para montar o contexto
        textos_contexto = []
        for chunk in chunks_recuperados:
            # Verifica se é um objeto Document do LangChain ou um dicionário
            if hasattr(chunk, 'page_content'):
                textos_contexto.append(chunk.page_content)
            elif isinstance(chunk, dict) and 'conteudo' in chunk:
                textos_contexto.append(chunk['conteudo'])
                
        contexto_agrupado = "\n\n---\n\n".join(textos_contexto)
        
        # 2. Conecta o Prompt ao LLM (Montagem do Chain)
        chain = self.prompt_template | self.llm
        
        # 3. Executa a inferência e retorna o texto da resposta
        resposta = chain.invoke({
            "contexto": contexto_agrupado,
            "pergunta": pergunta
        })
        
        return resposta.content