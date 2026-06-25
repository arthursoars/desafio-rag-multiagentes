import fitz # ou PyMuPDF
import os 
from src.ingestion.models import Documento 

# Parser: abre e extrai o texto de arquivos PDF usando PyMuPDF (fitz).
class Parser:
    def ler_pdf(self, caminho: str) -> Documento | None:
        try:
            doc = fitz.open(caminho)
            texto = ""
            for pagina in doc:
                texto += pagina.get_text()
            # Retorna um Documento preenchido com metadados e conteúdo
            return Documento(
                nome=os.path.basename(caminho),
                caminho=caminho,
                texto=texto,
                paginas=len(doc)
            )
        # Em caso de falha (ex.: encriptado/corrompido), retorna None
        except Exception:           
            return None