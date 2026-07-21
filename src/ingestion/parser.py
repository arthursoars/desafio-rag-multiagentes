import os
from dotenv import load_dotenv
import fitz
import ollama
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from src.ingestion.models import Documento

# Carrega as variáveis de ambiente
load_dotenv()

# Parser - Extrai texto de PDFs combinando OCR digital rápido com visão computacional para imagens
class Parser:
    def __init__(self):
        # Conecta ao servidor externo via .env
        url_ollama = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=url_ollama)

    def ler_pdf(self, caminho: str) -> Documento | None:
        # Caderneta para anotar quais páginas deram erro no servidor e sofreram "Fallback"
        paginas_com_erro = []
        
        try:
            # 1. Extração rápida do texto digital nativo (Estratégia Fast)
            elementos_fast = partition_pdf(
                filename=caminho,
                strategy="fast",
                languages=["por", "eng"]
            )

            # 2. Agrupa os elementos por número de página
            paginas_fast = {}
            for el in elementos_fast:
                num_pagina = el.metadata.page_number or 1
                paginas_fast.setdefault(num_pagina, []).append(el)

            # 3. Descobre o número real de páginas
            with open(caminho, "rb") as f:
                total_paginas = len(PdfReader(f).pages)

            texto_final = []

            # 4. Avalia cada página individualmente
            for num_pagina in range(1, total_paginas + 1):
                elementos_pagina = paginas_fast.get(num_pagina, [])
                texto_pagina = " ".join(el.text for el in elementos_pagina if el.text).strip()
                
                tem_imagem = self._pagina_tem_imagem(caminho, num_pagina)
                precisa_ia = (not texto_pagina) or tem_imagem

                if precisa_ia:
                    print(f"[{os.path.basename(caminho)}] Página {num_pagina}: acionando visão computacional (Qwen2.5-VL no Servidor)...")
                    
                    # Observação: o modelo de visão processa a página inteira.
                    # Evita-se combinar a extração 'fast' para prevenir duplicação.
                    try:
                        # PLANO A: Tenta usar a IA do Servidor
                        texto_ia = self._processar_pagina_com_ollama(caminho, num_pagina)
                        texto_pagina_final = texto_ia
                    except Exception as e:
                        # PLANO B (FALLBACK): Servidor deu Erro 500? Anota a página e usa o texto digital feio
                        paginas_com_erro.append(num_pagina)
                        texto_pagina_final = texto_pagina
                else:
                    # Página digital pura sem imagens: mantemos a leitura rápida
                    texto_pagina_final = texto_pagina

                if texto_pagina_final:
                    texto_final.append(texto_pagina_final)

            # ESTRATÉGIA DE AVISO: Mostra onde o sistema ficou "imperfeito"
            if paginas_com_erro:
                print(f"\n      ⚠️ RELATÓRIO DE IMPERFEIÇÕES: O documento {os.path.basename(caminho)}")
                print(f"      sofreu falha de Visão (Erro 500) nas páginas: {paginas_com_erro}.")
                print(f"      Foi aplicado o FALLBACK (texto sem IA) nestas páginas para não perder o PDF inteiro.\n")

            return Documento(
                nome=os.path.basename(caminho),
                caminho=caminho,
                texto="\n\n".join(texto_final),
                paginas=total_paginas
            )

        except Exception as e:
            print(f"Erro fatal ao ler {os.path.basename(caminho)}: {e}")
            return None

    def _pagina_tem_imagem(self, caminho: str, num_pagina: int, area_minima: int = 40000) -> bool:
        # Intuito: Otimização; evita-se acionar o modelo de visão computacional para páginas sem imagens relevantes.
        """Verifica via PyMuPDF se a página tem imagens relevantes (ignora logos/ícones pequenos).""" 
        doc_fitz = fitz.open(caminho)
        pagina = doc_fitz[num_pagina - 1]
        imagens = pagina.get_images(full=True)

        for img in imagens:
            xref = img[0]
            pixmap_info = doc_fitz.extract_image(xref)
            area = pixmap_info["width"] * pixmap_info["height"]
            if area >= area_minima:
                doc_fitz.close()
                return True

        doc_fitz.close()
        return False

    def _processar_pagina_com_ollama(self, caminho: str, num_pagina: int) -> str:
        """Motor VLM: Converte página em imagem e extrai o texto usando o servidor remoto."""
        doc_fitz = fitz.open(caminho)
        pagina = doc_fitz[num_pagina - 1]
        pixmap = pagina.get_pixmap(dpi=72)
        imagem_bytes = pixmap.tobytes("png")
        doc_fitz.close()

        # Prompt estrito em português para forçar transcrição literal (Atualizado para o 7b)
        resposta = self.ollama_client.generate(
            model="qwen2.5vl:7b",
            prompt="Transcreva exatamente todo o texto visível nesta imagem. Retorne APENAS a transcrição do texto, sem fazer resumos ou descrições da imagem.",
            images=[imagem_bytes]
        )
        return resposta.get("response", "").strip()