import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class RoteadorETratadorConsultas:
    # Atualizado definitivamente para o modelo de 7B que o Eric deixou no servidor
    def __init__(self, model_name="qwen2.5vl:7b"):
        """
        Método Construtor (POO): Inicializa o modelo Ollama com temperatura zero
        e prepara os templates de prompt para Roteamento e Reescrita.
        """
        # Atributo da classe que segura a ligação com o modelo de IA
        self.llm = OllamaLLM(
            model=model_name,
            # Busca a URL no .env apontando pro servidor (http://187.45.180.38:11434)
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0
        )
        
        # 1. Prompt do Semantic Routing (Roteamento Semântico) - REGRA INVERTIDA
        self.prompt_roteamento = PromptTemplate.from_template(
            """Você é um roteador de requisições de um sistema de IA corporativo.
Sua única tarefa é classificar a pergunta do usuário em "GERAL" ou "DOCUMENTO".

Regra 1 (GERAL): Use APENAS para saudações curtas (ex: oi, bom dia, olá), despedidas ou perguntas claramente fora de contexto (ex: receita de bolo, piadas).
Regra 2 (DOCUMENTO): Use para TODO O RESTO. Qualquer pergunta técnica, dúvida de editais, termos confusos, busca por regras, ou mesmo frases que não façam muito sentido devem ir para DOCUMENTO. Na dúvida, escolha DOCUMENTO.

EXEMPLOS DE CLASSIFICAÇÃO:
Pergunta: "bom dia, tudo bem?"
Classificação: geral

Pergunta: "quais os documentos para a bolsa?"
Classificação: documento

Pergunta: "como fazer um bolo de chocolate?"
Classificação: geral

PERGUNTA DO USUÁRIO: {pergunta}
CLASSIFICAÇÃO (Apenas uma palavra):"""
        )
        
        # 2. Prompt do Query Rewriting (Reescrita de Consulta)
        self.prompt_reescrita = PromptTemplate.from_template(
            """És um especialista em reescrever consultas para motores de busca vetorial.
A tua tarefa é receber uma pergunta informal do utilizador e transformá-la numa consulta formal, clara e técnica, mantendo a intenção original.

REGRAS CRÍTICAS:
1. Remove gírias, abreviaturas incorretas ou ambiguidades.
2. Usa termos formais que costumam aparecer em artigos científicos, editais, relatórios financeiros ou manuais técnicos.
3. Retorna APENAS a pergunta reescrita final, sem introduções ou comentários.

PERGUNTA ORIGINAL:
{pergunta}

PERGUNTA REESCRITA EM FORMATO FORMAL:"""
        )
        
        # Construção das esteiras usando sintaxe LCEL do LangChain
        self.chain_roteamento = self.prompt_roteamento | self.llm
        self.chain_reescrita = self.prompt_reescrita | self.llm

    def rotear_consulta(self, pergunta_usuario: str) -> str:
        """
        Método de Ação (POO): Valida se a pergunta deve ir para o banco ou ser barrada.
        """
        print(f" -> [Semantic Routing] A avaliar a intenção da pergunta via Servidor...")
        if not pergunta_usuario.strip():
            return "geral"
            
        resposta = self.chain_roteamento.invoke({"pergunta": pergunta_usuario})
        resposta_limpa = resposta.strip().lower()
        
        # Camada de segurança (Se a IA responder "documento", passa. Se não, bloqueia)
        if "documento" in resposta_limpa:
            return "documento"
            
        return "geral"

    def reescrever_consulta(self, pergunta_usuario: str) -> str:
        """
        Método de Ação (POO): Transforma o texto do utilizador em termos formais de documentos.
        """
        print(f" -> [Query Rewriting] A higienizar e formalizar a consulta via Servidor...")
        if not pergunta_usuario.strip():
            return pergunta_usuario
            
        pergunta_formal = self.chain_reescrita.invoke({"pergunta": pergunta_usuario})
        return pergunta_formal.strip()