# Divide o texto em blocos de tamanho fixo, com sobreposição opcional.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.models import Documento

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Inicializa o fatiador de textos.
        - chunk_size: Quantidade máxima de caracteres por pedaço.
        - chunk_overlap: Quantos caracteres o pedaço atual vai 'roubar' do final do pedaço anterior (mantém o contexto).
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

    def fatiar_documento(self, documento: Documento) -> list[dict]:
        """
        Recebe o objeto Documento, fatia o texto limpo e devolve uma lista de dicionários 
        contendo o chunk e os metadados (de onde ele veio).
        """
        if not documento or not documento.texto:
            return []
            
        chunks_texto = self.splitter.split_text(documento.texto)
        
        chunks_formatados = []
        for i, texto_chunk in enumerate(chunks_texto):
            chunks_formatados.append({
                "texto": texto_chunk,
                "metadata": {
                    "source": documento.nome,
                    "chunk_index": i
                }
            })
            
        return chunks_formatados