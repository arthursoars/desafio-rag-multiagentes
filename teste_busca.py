import sys
from src.retrieval.routers import RoteadorETratadorConsultas
from src.retrieval.advanced_search import MotorBuscaAvancado

def executar_teste_rag():
    print("==============================================================")
    print("      🚀 INICIANDO TESTE INTEGRADO DO MOTOR DE BUSCA (RETRIEVAL)")
    print("==============================================================\n")

    # 1. INSTANCIAÇÃO (POO): Criamos os objetos na memória a partir das nossas classes
    tratador = RoteadorETratadorConsultas()
    buscador = MotorBuscaAvancado()

    # Simulando uma pergunta realista de usuário (com gírias e informal)
    pergunta_usuario = "Como faço para me inscrever no edital?"
    print(f"💬 Pergunta Original do Usuário: '{pergunta_usuario}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 1: SEMANTIC ROUTING (ROTEAMENTO SEMÂNTICO)
    # ----------------------------------------------------------------------
    rota = tratador.rotear_consulta(pergunta_usuario)
    print(f"   ➔ Rota Decidida pela IA: [{rota.upper()}]")

    if rota == "geral":
        print("   ❌ [Bloqueio de Rota]: Esta pergunta não é sobre documentos. Fluxo encerrado.")
        sys.exit()
    print("   ✅ [Rota Liberada]: A pergunta é válida! Prosseguindo para o banco de dados...\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 2: QUERY REWRITING (REESCRITA DE CONSULTA)
    # ----------------------------------------------------------------------
    pergunta_formal = tratador.reescrever_consulta(pergunta_usuario)
    print(f"   ➔ Consulta Formalizada: '{pergunta_formal}'\n")

    # ----------------------------------------------------------------------
    # TÉCNICAS 3 e 4: HyDE e METADATA PRE-FILTERING (DENTRO DO QDRANT)
    # ----------------------------------------------------------------------
    # Este método vai gerar a resposta falsa (HyDE) e filtrar no Qdrant
    pontos_recuperados = buscador.recuperar_chunks_elite(pergunta_formal, top_k=10)
    print(f"   ➔ Chunks brutos recuperados do Qdrant: {len(pontos_recuperados)} blocos.\n")

    # ----------------------------------------------------------------------
    # TÉCNICA 5: RERANKING (RECLASSIFICAÇÃO DO BGE-M3)
    # ----------------------------------------------------------------------
    # Passamos a pergunta original do usuário para desempatar o contexto profundo
    chunks_finais = buscador.aplicar_reranking(pergunta_usuario, pontos_recuperados, top_n=3)

    # ----------------------------------------------------------------------
    # EXIBIÇÃO DO RESULTADO COMPILADO
    # ----------------------------------------------------------------------
    print("\n==============================================================")
    print("        🏆 RESULTADO FINAL: TOP 3 CHUNKS SELECIONADOS")
    print("==============================================================")
    
    for i, chunk in enumerate(chunks_finais):
        print(f"\n🥇 Posição #{i+1} (Score Real: {chunk['score_relevancia']:.2f})")
        print(f"📄 Origem: {chunk['metadata'].get('source', 'Desconhecido')}")
        print(f"🏷️ Tema Principal: {chunk['metadata'].get('tema_principal', 'Não categorizado')}")
        print(f"💡 Texto Limpo Recuperado:\n   \"{chunk['texto'][:200]}...\"")
        print("-" * 60)

if __name__ == "__main__":
    # Ponto de entrada padrão do Python que executa a nossa função principal
    executar_teste_rag()
