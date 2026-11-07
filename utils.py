import base64
import json
import math
import re
import time
from html import unescape

import requests

from config import API_URL, GROQ_API_KEY, MODEL_NAME, VISION_MODEL, TOPE_IMAGENES


def llamar_llm(system_instruction, prompt):
    """Una llamada al LLM. 'prompt' puede ser texto o una lista (texto + imágenes)."""
    partes = prompt if isinstance(prompt, list) else [prompt]
    contenido, hay_imagen = [], False
    for p in partes:
        if isinstance(p, str):
            contenido.append({"type": "text", "text": p})
        else:
            b64 = base64.b64encode(p["data"]).decode()
            contenido.append({"type": "image_url",
                              "image_url": {"url": f"data:{p['mime_type']};base64,{b64}"}})
            hay_imagen = True
    if not hay_imagen:
        contenido = "\n".join(c["text"] for c in contenido)

    payload = {
        "model": VISION_MODEL if hay_imagen else MODEL_NAME,
        "messages": [{"role": "system", "content": system_instruction},
                     {"role": "user", "content": contenido}],
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    for intento in range(4):
        r = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        if r.status_code == 429 and intento < 3:       # límite de uso: espera y reintenta
            try:
                espera = float(r.headers.get("retry-after", 5 * (intento + 1)))
            except ValueError:
                espera = 5 * (intento + 1)
            time.sleep(min(espera, 30))
            continue
        break
    if not r.ok:
        raise RuntimeError(f"Error de la API ({r.status_code}): {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def parsear_json(raw):
    """Extrae el JSON de la respuesta, ignorando ``` o texto extra alrededor."""
    ini, fin = raw.find("{"), raw.rfind("}")
    if ini == -1 or fin == -1:
        raise ValueError("la respuesta no contiene JSON")
    return json.loads(raw[ini:fin + 1])