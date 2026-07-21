from src.retrieval.routers import RoteadorETratadorConsultas

def rodar_teste():
    print("🚀 Iniciando teste do Roteador via Servidor AceleraI (Qwen 7B)...\n")
    
    # Instancia o nosso roteador (ele já vai ler o .env automaticamente)
    roteador = RoteadorETratadorConsultas()
    
    # Lista de perguntas para testar se o "Bug do Geral" sumiu
    perguntas_teste = [
        "Quais os prazos para entregar os documentos da bolsa?",
        "Me explica sobre aquele edital de auxílio",
        "Como funciona a regra de isenção?",
        "Bom dia, tudo bem com você?",
        "Me dê uma receita de bolo de cenoura."
    ]
    
    for pergunta in perguntas_teste:
        print(f"👤 Usuário: {pergunta}")
        
        # Chama a função que vai lá no servidor do Eric e volta
        classificacao = roteador.rotear_consulta(pergunta)
        
        print(f"🤖 Decisão da IA: {classificacao.upper()}")
        print("-" * 50)
    print("✅ Teste finalizado com sucesso!")
if __name__ == "__main__":
    rodar_teste()