import os
import re

import pdfplumber

from agentes.base import Agente


def _extraer_plumber(ruta):
    paginas = []
    with pdfplumber.open(ruta) as pdf:
        for p in pdf.pages:
            paginas.append(p.extract_text() or "")
    return paginas


class Lector(Agente):
    nombre = "Lector"
    rol = "Lee el PDF con pdfplumber (extrae el texto de cada página)"

    def ejecutar(self, entrada, feedback=None):
        paginas = _extraer_plumber(entrada)
        texto = "\n".join(t for t in paginas if t)
        vacias = sum(1 for t in paginas if not t)

        lineas = ["Fuente usada: pdfplumber", "", "Lectura por página:"]
        for i, t in enumerate(paginas[:12], 1):
            muestra = t[:200].replace("\n", " ").strip()
            lineas.append(f"Pág. {i} ({len(t)} caracteres): {muestra}..." if t else f"Pág. {i}: sin texto extraíble")
        if len(paginas) > 12:
            lineas.append(f"... y {len(paginas) - 12} páginas más")

        return {
            "salida": {"texto": texto, "paginas": len(paginas), "fuente": "pdfplumber",
                       "texto_plumber": texto, "solapamiento": None},
            "recibio": f"PDF: {os.path.basename(entrada)}",
            "entrego": f"{len(paginas)} páginas, {len(texto)} caracteres de texto (leído con pdfplumber)",
            "razonamiento": "Extraje el texto de cada página con pdfplumber. No funciona con PDFs "
                            "escaneados como imagen (sin OCR).",
            "detalle": "\n".join(lineas),
            "notas": f"{vacias} páginas sin texto extraíble",
        }