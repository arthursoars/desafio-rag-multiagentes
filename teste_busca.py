import sys
from src.retrieval.routers import RoteadorETratadorConsultas
from src.retrieval.advanced_search import MotorBuscaAvancado

# teste_busca - Testa fluxo completo de RAG: roteamento, reescrita, busca HyDE e filtragem de conteúdo
def executar_teste_rag():
    print("==============================================================")
    print("      🚀 INICIANDO TESTE INTEGRADO DO MOTOR DE BUSCA (RETRIEVAL)")
    print("==============================================================\n")

    # 1. INSTANCIAÇÃO (POO): Criamos os objetos a partir das classes estruturadas
    try:
        tratador = RoteadorETratadorConsultas()
        buscador = MotorBuscaAvancado()
    except Exception as e:
        print(f"❌ [Erro de Inicialização]: Falha ao conectar aos componentes. Verifique o Docker. {e}")
        sys.exit()

    # Sua pergunta original com gírias (vamos testar a resiliência do novo prompt)
    pergunta_usuario = "Como me inscrevo no edital da UFPB?"
    print(f"💬 Pergunta Original do Usuário: '{pergunta_usuario}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 1: SEMANTIC ROUTING (ROTEAMENTO SEMÂNTICO com Chain of Thought)
    # ----------------------------------------------------------------------
    # O método interno agora vai imprimir o LOG DE TRANSPARÊNCIA exigido pelo edital
    rota = tratador.rotear_consulta(pergunta_usuario)
    print(f"   ➔ Rota Decidida pelo Filtro: [{rota.upper()}]")

    if rota == "geral":
        print("   ❌ [Bloqueio de Rota]: Esta pergunta foi considerada fora do escopo documental. Fluxo encerrado.")
        sys.exit()
    print("   ✅ [Rota Liberada]: Passagem autorizada! Buscando no acervo de PDFs...\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 2: QUERY REWRITING (REESCRITA DE CONSULTA)
    # ----------------------------------------------------------------------
    pergunta_formal = tratador.reescrever_consulta(pergunta_usuario)
    print(f"   ➔ Consulta Formalizada pela IA: '{pergunta_formal}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICAS 3 e 4: HyDE e METADATA PRE-FILTERING (DENTRO DO QDRANT)
    # ----------------------------------------------------------------------
    try:
        # Puxa os 10 melhores candidatos aplicando o pré-filtro de conteúdo crítico
        pontos_recuperados = buscador.recuperar_chunks_elite(pergunta_formal, top_k=10)
        print(f"   ➔ Chunks brutos recuperados do Qdrant: {len(pontos_recuperados)} blocos.\n")
    except Exception as e:
        print(f"   ❌ [Erro no Qdrant]: Falha ao consultar os vetores. {e}")
        sys.exit()

    # ----------------------------------------------------------------------
    # TÉCNICA 5: RERANKING (RECLASSIFICAÇÃO DO BGE-M3)
    # ----------------------------------------------------------------------
    # Compara o contexto profundo dos 10 blocos com a dúvida real e escolhe os 3 campeões
    chunks_finais = buscador.aplicar_reranking(pergunta_usuario, pontos_recuperados, top_n=3)

    # ----------------------------------------------------------------------
    # EXIBIÇÃO DO RESULTADO COMPILADO (Métricas e Metadados)
    # ----------------------------------------------------------------------
    print("\n==============================================================")
    print("        🏆 RESULTADO FINAL: TOP 3 CHUNKS SELECIONADOS")
    print("==============================================================")
    
    if not chunks_finais:
        print("\n⚠️ Nenhum documento relevante passou pelos filtros de segurança.")
    
    for i, chunk in enumerate(chunks_finais):
        print(f"\n🥇 Posição #{i+1} (Score Reclassificado: {chunk['score_relevancia']:.2f})")
        print(f"📄 Arquivo de Origem: {chunk['metadata'].get('source', 'Desconhecido')}")
        print(f"🏷️ Categoria: {chunk['metadata'].get('categoria_documento', 'Não identificada')}")
        print(f"💡 Conteúdo do Fragmento:\n   \"{chunk['texto'].strip()}\"")
        print("-" * 60)

if __name__ == "__main__":
    executar_teste_rag()
