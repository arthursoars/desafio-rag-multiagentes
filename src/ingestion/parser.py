import os
import fitz
import ollama
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from src.ingestion.models import Documento

class Parser:
    def __init__(self):
        # Conecta diretamente ao contêiner do Ollama
        self.ollama_client = ollama.Client(host="http://ollama:11434")

    def ler_pdf(self, caminho: str) -> Documento | None:
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
                    print(f"[{os.path.basename(caminho)}] Página {num_pagina}: acionando visão computacional (Qwen2-VL)...")
                    texto_ia = self._processar_pagina_com_ollama(caminho, num_pagina)
                    
                    # Mescla o texto digital com o que a IA leu da imagem
                    if texto_ia:
                        texto_pagina_final = f"{texto_pagina}\n\n[TEXTO EXTRAÍDO PELA IA]\n{texto_ia}".strip() if texto_pagina else texto_ia
                    else:
                        texto_pagina_final = texto_pagina
                else:
                    texto_pagina_final = texto_pagina

                if texto_pagina_final:
                    texto_final.append(texto_pagina_final)

            return Documento(
                nome=os.path.basename(caminho),
                caminho=caminho,
                texto="\n\n".join(texto_final),
                paginas=total_paginas
            )

        except Exception as e:
            print(f"Erro ao ler {os.path.basename(caminho)}: {e}")
            return None

    def _pagina_tem_imagem(self, caminho: str, num_pagina: int, area_minima: int = 40000) -> bool:
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
        """Motor VLM: Converte página em imagem e extrai o texto com Qwen2.5VL 3B."""
        doc_fitz = fitz.open(caminho)
        pagina = doc_fitz[num_pagina - 1]
        pixmap = pagina.get_pixmap(dpi=72)
        imagem_bytes = pixmap.tobytes("png")
        doc_fitz.close()

        # Prompt estrito em português para forçar transcrição literal
        resposta = self.ollama_client.generate(
            model="qwen2.5vl:3b",
            prompt="Transcreva exatamente todo o texto visível nesta imagem. Retorne APENAS a transcrição do texto, sem fazer resumos ou descrições da imagem.",
            images=[imagem_bytes]
        )
        return resposta.get("response", "").strip()