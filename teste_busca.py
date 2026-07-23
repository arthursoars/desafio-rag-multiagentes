import sys
from src.retrieval.routers import RoteadorETratadorConsultas
from src.retrieval.advanced_search import MotorBuscaAvancado
from src.generation.generator import AgenteGerador  # Importação da Fase 3

# teste_busca - Testa fluxo completo de RAG: roteamento, reescrita, busca HyDE, filtragem e geração
def executar_teste_rag():
    print("==============================================================")
    print("      INICIANDO TESTE INTEGRADO DO MOTOR DE BUSCA (RAG)       ")
    print("==============================================================\n")

    # 1. INSTANCIAÇÃO (POO): Criamos os objetos a partir das classes estruturadas
    try:
        tratador = RoteadorETratadorConsultas()
        buscador = MotorBuscaAvancado()
        gerador = AgenteGerador()  # Instanciando o Gerador de Respostas
    except Exception as e:
        print(f"[ERRO FATAL]: Falha ao conectar aos componentes. Verifique o Docker. {e}")
        sys.exit()

    # Pergunta original do usuário
    pergunta_usuario = "Quais e quantos editais voce tem conhecimento? e quais as datas criticas que devo ficar atento?"
    print(f"[PERGUNTA DO USUÁRIO]: '{pergunta_usuario}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 1: SEMANTIC ROUTING (ROTEAMENTO SEMÂNTICO)
    # ----------------------------------------------------------------------
    rota = tratador.rotear_consulta(pergunta_usuario)
    print(f"[ROTEAMENTO] Rota decidida: {rota.upper()}")

    if rota == "geral":
        print("[BLOQUEIO] Pergunta fora do escopo documental. Fluxo encerrado.")
        sys.exit()
    print("[ROTEAMENTO] Passagem autorizada. Acessando acervo de PDFs...\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 2: QUERY REWRITING (REESCRITA DE CONSULTA)
    # ----------------------------------------------------------------------
    pergunta_formal = tratador.reescrever_consulta(pergunta_usuario)
    print(f"[REESCRITA] Consulta formalizada: '{pergunta_formal}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICAS 3 e 4: HyDE e METADATA PRE-FILTERING (NO QDRANT)
    # ----------------------------------------------------------------------
    try:
        # Puxa os 10 melhores candidatos aplicando o pré-filtro
        pontos_recuperados = buscador.recuperar_chunks_elite(pergunta_formal, top_k=10)
        print(f"[BUSCA VETORIAL] Recuperados {len(pontos_recuperados)} chunks brutos do Qdrant.\n")
    except Exception as e:
        print(f"[ERRO QDRANT]: Falha ao consultar os vetores. {e}")
        sys.exit()

    # ----------------------------------------------------------------------
    # TÉCNICA 5: RERANKING (RECLASSIFICAÇÃO DO BGE-M3)
    # ----------------------------------------------------------------------
    # Aumentado para top_n=5 conforme planejado para maior contexto
    chunks_finais = buscador.aplicar_reranking(pergunta_usuario, pontos_recuperados, top_n=5)

    if not chunks_finais:
        print("\n[AVISO]: Nenhum documento relevante passou pelos filtros de segurança.")
        sys.exit()

    # ----------------------------------------------------------------------
    # EXIBIÇÃO DO RESULTADO COMPILADO (Métricas e Metadados)
    # ----------------------------------------------------------------------
    print("==============================================================")
    print("            CHUNKS SELECIONADOS PARA CONTEXTO                 ")
    print("==============================================================")
    
    # Preparamos uma lista no formato esperado pelo AgenteGerador
    textos_para_gerador = []

    for i, chunk in enumerate(chunks_finais):
        score = chunk.get('score_relevancia', 0)
        metadata = chunk.get('metadata', {})
        texto_chunk = chunk.get('texto', '').strip()
        
        # Salvando o texto na lista que vai para a LLM
        textos_para_gerador.append({"conteudo": texto_chunk})
        
        print(f"\n[Posição #{i+1}] Score: {score:.2f} | Origem: {metadata.get('source', 'Desconhecido')}")
        print(f"Trecho: \"{texto_chunk[:150]}...\"") # Imprimindo só o começo para não poluir a tela
    
    # ----------------------------------------------------------------------
    # FASE 3: GERAÇÃO DA RESPOSTA FINAL (LLM)
    # ----------------------------------------------------------------------
    print("\n==============================================================")
    print("            FASE 3: GERAÇÃO DE RESPOSTA (LLM QWEN)            ")
    print("==============================================================")
    print("[PROCESSANDO] Lendo os contextos e redigindo resposta humana...\n")
    
    resposta_final = gerador.gerar_resposta_final(pergunta_usuario, textos_para_gerador)
    
    print("==============================================================")
    print("                      RESPOSTA DA IA                          ")
    print("==============================================================")
    print(f"\n{resposta_final}\n")
    print("==============================================================")


if __name__ == "__main__":
    executar_teste_rag()