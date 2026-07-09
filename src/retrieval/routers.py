from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# RoteadorETratadorConsultas - Valida intenção de consultas e normaliza linguagem informal para formal
## Pré -processamento de consultas do usuário: Roteamento Semântico + Reescrita de Consulta
class RoteadorETratadorConsultas:
    def __init__(self, model_name="qwen2.5vl:3b"):
        """
        Método Construtor (POO): Inicializa o modelo Ollama com temperatura zero
        e prepara os templates de prompt para Roteamento e Reescrita.
        """
        # Atributo da classe que segura a conexão com o modelo de IA
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://ollama:11434",
            temperature=0.0
        )
        
        # 1. Prompt do Semantic Routing (Roteamento Semântico)
        self.prompt_roteamento = PromptTemplate.from_template(
            """Você é um guarda de trânsito digital de um sistema RAG especialista em documentos.
Sua única tarefa é avaliar a INTENÇÃO real da pergunta do usuário, ignorando se ela foi escrita de forma informal, com gírias ou erros de digitação.

REGRAS DE CLASSIFICAÇÃO:
1. Responda APENAS 'documento' se o usuário estiver buscando alguma informação contida em arquivos (como regras, prazos, cotas, inscrições, bolsas, critérios, etc.), mesmo que use gírias (ex: "me diz as parada das cota", "como q faz pra se inscrever", "quero o denero da bolsa").
2. Responda APENAS 'geral' se a pergunta for puramente uma saudação (olá, bom dia), gíria vazia ("e aí mano"), piada ou assunto totalmente aleatório que não exige leitura de documentos (ex: receita de bolo, futebol, previsão do tempo).
3. Não escreva explicações, justificativas ou pontuações. Retorne exatamente apenas uma das duas palavras.

PERGUNTA DO USUÁRIO:
{pergunta}

CLASSIFICAÇÃO COMPLETA:"""
        )
        
        # 2. Prompt do Query Rewriting (Reescrita de Consulta)
        self.prompt_reescrita = PromptTemplate.from_template(
            """Você é um especialista em reescrever consultas para motores de busca vetorial.
Sua tarefa é receber uma pergunta informal do usuário e transformá-la em uma consulta formal, clara e técnica, mantendo a intenção original.

REGRAS CRÍTICAS:
1. Remova gírias, abreviações incorretas ou ambiguidades.
2. Use termos formais que costumam aparecer em artigos científicos, editais, relatórios financeiros ou manuais técnicos.
3. Retorne APENAS a pergunta reescrita final, sem introduções ou comentários.

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
        print(f" -> [Semantic Routing] Avaliando a intenção da pergunta...")
        if not pergunta_usuario.strip():
            return "geral"
            
        resposta = self.chain_roteamento.invoke({"pergunta": pergunta_usuario})
        return resposta.strip().lower()

    def reescrever_consulta(self, pergunta_usuario: str) -> str:
        """
        Método de Ação (POO): Transforma o texto do usuário em termos formais de documentos.
        """
        print(f" -> [Query Rewriting] Higienizando e formalizando a consulta...")
        if not pergunta_usuario.strip():
            return pergunta_usuario
            
        pergunta_formal = self.chain_reescrita.invoke({"pergunta": pergunta_usuario})
        return pergunta_formal.strip()
