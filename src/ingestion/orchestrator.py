import os
from dotenv import load_dotenv
from src.ingestion.parser import Parser
from src.ingestion.models import Documento

load_dotenv()

class Orchestrator:
    def __init__(self):
        self.parser = Parser()
        # Lê a configuração de lote; assume 5 como padrão conforme o edital real
        self.batch_size = int(os.getenv("BATCH_SIZE", 5))
        self.erros = []

    def processar_lote(self, pasta: str) -> list[Documento]:
        # Filtra estritamente arquivos PDF, ignorando screenshots ou imagens soltas
        arquivos = [
            os.path.join(pasta, f)
            for f in os.listdir(pasta)
            if f.lower().endswith(".pdf")
        ]
        
        # Aplica rigorosamente o controle de lote exigido na Tarefa 1
        arquivos = arquivos[:self.batch_size]
        documentos = []
        
        for caminho in arquivos:
            doc = self.parser.ler_pdf(caminho)
            if doc is None:
                self.erros.append(os.path.basename(caminho))
                continue
            documentos.append(doc)
            
        return documentos