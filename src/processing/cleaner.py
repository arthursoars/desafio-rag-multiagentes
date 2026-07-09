from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# AgenteLimpeza - Corrige erros de OCR e formatação em textos extraídos de documentos
class AgenteLimpeza:
    def __init__(self, model_name="qwen2.5vl:3b"):
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://ollama:11434",
            temperature=0.0
        )
        
        self.prompt = PromptTemplate.from_template(
            """Você é um especialista em revisão de textos corrompidos por OCR.
            Sua única tarefa é receber um trecho de texto bruto, corrigir falhas de leitura, 
            caracteres quebrados, hifenizações erradas e problemas de formatação.
            
            REGRAS OBRIGATÓRIAS:
            1. NÃO adicione informações novas.
            2. NÃO resuma o texto.
            3. Retorne APENAS o texto corrigido, sem introduções.
            
            TEXTO BRUTO:
            {chunk_bruto}
            
            TEXTO CORRIGIDO:"""
        )
        
        self.chain = self.prompt | self.llm

    def limpar_chunk(self, chunk_texto: str) -> str:
        if not chunk_texto.strip():
            return ""
            
        print("   -> [Agente Limpeza] Inspecionando e corrigindo chunk...")
        texto_limpo = self.chain.invoke({"chunk_bruto": chunk_texto})
        return texto_limpo.strip()