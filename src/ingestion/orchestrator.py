import os
from dotenv import load_dotenv
from src.ingestion.parser import Parser
from src.ingestion.models import Documento

# Orchestrator: coordena o processamento em lote de arquivos PDF.

load_dotenv()
# Carrega variáveis de ambiente do .env (ex.: BATCH_SIZE)

class Orchestrator:
    def __init__(self):
        self.parser = Parser()
        # Lê o tamanho do lote do .env; usa 3 como padrão caso não exista
        self.batch_size = int(os.getenv("BATCH_SIZE", 3))
        self.erros = []  # Guardar erros para exibir depois

    def processar_lote(self, pasta: str) -> list[Documento]:
        # Processa um lote de arquivos na pasta informada e retorna apenas Documentos válidos
        arquivos = [
            os.path.join(pasta, f)
            for f in os.listdir(pasta)
            if f.endswith(".pdf")
        ]
        # aplica limite de lote 
        arquivos = arquivos[:self.batch_size]
        documentos = []
        for caminho in arquivos:
            # chama o parser para ler o PDF; em caso de falha, registra o nome e segue
            doc = self.parser.ler_pdf(caminho)
            if doc is None:
                self.erros.append(os.path.basename(caminho))
                continue
            documentos.append(doc)
        return documentos