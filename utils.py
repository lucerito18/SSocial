import base64
import json
import math
import re
import time
from html import unescape

import requests

from config import API_URL, GROQ_API_KEY, MODEL_NAME, VISION_MODEL, TOPE_IMAGENES

REGLA_JSON = (
    "\n\nIMPORTANTE: tu respuesta debe ser JSON estrictamente válido. NO uses comillas dobles dentro "
    "de los textos (usa « » o comillas simples), no pongas comas finales y no uses saltos de línea "
    "dentro de los textos."
)


def _espera_429(texto, intento):
    """Lee del mensaje de Groq cuántos segundos hay que esperar."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", texto)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    return 5 * (intento + 1)


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
        "messages": [{"role": "system", "content": system_instruction + REGLA_JSON},
                     {"role": "user", "content": contenido}],
        "temperature": 0.2,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    for intento in range(7):
        r = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        if r.status_code == 400:
            # el modelo no soporta algún parámetro opcional: se quita y se reintenta
            quitado = False
            for k in ("reasoning_effort", "response_format"):
                if k in payload:
                    payload.pop(k)
                    quitado = True
                    break
            if quitado:
                continue
        if r.status_code in (429, 503) and intento < 6:
            if "tokens per day" in r.text:      # límite diario: esperar no sirve
                break
            espera = _espera_429(r.text, intento)
            if espera > 75:
                break
            time.sleep(espera)
            continue
        break
    if not r.ok:
        raise RuntimeError(f"Error de la API ({r.status_code}): {r.text[:300]}")
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def parsear_json(raw):
    """Extrae el JSON de la respuesta, ignorando ``` o texto extra alrededor."""
    ini, fin = raw.find("{"), raw.rfind("}")
    if ini == -1 or fin == -1:
        raise ValueError("la respuesta no contiene JSON")
    texto = raw[ini:fin + 1]
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        texto = re.sub(r",\s*([}\]])", r"\1", texto)   # quita comas finales
        return json.loads(texto, strict=False)


def num_palabras(texto):
    return len(str(texto).split())


def con_feedback(prompt, feedback):
    """Si el Jefe rechazó un intento anterior, se lo decimos al agente."""
    if not feedback:
        return prompt
    return (
        prompt
        + "\n\nIMPORTANTE: el Jefe rechazó tu intento anterior por estos motivos:\n"
        + feedback
        + "\nCorrígelo en este nuevo intento."
    )


def _max_imagenes(n_slides):
    """Cupo de imágenes según el tamaño de la presentación."""
    return max(4, min(TOPE_IMAGENES, n_slides // 2))


def _chk(lista, regla, ok, detalle="", bloquea=True):
    """Registra una revisión del Jefe. Si 'bloquea' es False, solo avisa (⚠️)."""
    lista.append({"regla": regla, "ok": bool(ok), "detalle": "" if ok else (detalle or regla),
                  "bloquea": bloquea})


def _limpiar_html(s):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _recortar_texto(texto, limite=9000):
    """Si el PDF es largo, conserva inicio, parte del centro y final (ahí suele estar la bibliografía)."""
    if len(texto) <= limite:
        return texto
    inicio, fin = int(limite * 0.4), int(limite * 0.2)
    resto = limite - inicio - fin
    medio_ini = len(texto) // 2 - resto // 2
    return (texto[:inicio]
            + "\n[... texto omitido ...]\n"
            + texto[medio_ini:medio_ini + resto]
            + "\n[... texto omitido ...]\n"
            + texto[-fin:])


def _referencia_en_texto(ref, texto_min):
    """Guardia anti-invención: casi todas las palabras de la referencia deben estar en el PDF."""
    palabras = re.findall(r"\w{4,}", ref.lower())
    if not palabras:
        return False
    return sum(1 for w in palabras if w in texto_min) / len(palabras) >= 0.7


# ---- estimación de si un texto cabe en su caja (sin renderizar) ----
def _altura_in(textos, ancho_in, tam, espacio_pt, sangria_in):
    cpl = max(1.0, (ancho_in - sangria_in) * 144 / tam / 1.1)  # caracteres por línea (aprox., conservador)
    lineas = sum(max(1, math.ceil(len(t) / cpl)) for t in textos)
    return lineas * tam * 1.2 / 72 + len(textos) * espacio_pt / 72


def _mejor_tam(textos, ancho_emu, alto_emu, tams, espacio_pt=0, sangria_in=0.4):
    ancho_in = ancho_emu / 914400 - 0.2
    alto_in = alto_emu / 914400 - 0.1
    for t in tams:
        if _altura_in(textos, ancho_in, t, espacio_pt, sangria_in) <= alto_in * 0.95:
            return t
    return tams[-1]


def _cabe(shape, sangria_in=0.4):
    pars = shape.text_frame.paragraphs
    textos = [p.text for p in pars]
    tam = next((r.font.size.pt for p in pars for r in p.runs if r.font.size), 18)
    esp = pars[0].space_after.pt if pars and pars[0].space_after is not None else 0
    return _altura_in(textos, shape.width / 914400 - 0.2, tam, esp, sangria_in) <= shape.height / 914400 - 0.1


# ---- color y contraste (para validar el diseño propuesto por la IA) ----
def _hex_a_rgb(h):
    h = str(h).strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"color inválido: {h!r}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _a_hex(t):
    return "#%02X%02X%02X" % tuple(t)


def _luminancia(rgb):
    def canal(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (canal(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contraste(a, b):
    l1, l2 = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def _problemas_contraste(pal):
    blanco = (0xFF, 0xFF, 0xFF)
    reglas = [("texto", pal["texto"], pal["fondo"], 4.5), ("título", pal["titulo"], pal["fondo"], 4.5),
              ("texto sobre tinte", pal["texto"], pal["tinte"], 4.5),
              ("título sobre tinte", pal["titulo"], pal["tinte"], 4.5),
              ("blanco sobre portada", blanco, pal["portada"], 4.5),
              ("acento2 sobre portada", pal["acento2"], pal["portada"], 2.0),
              ("blanco sobre acento", blanco, pal["acento"], 2.5)]
    out = []
    for nombre, a, b, minimo in reglas:
        r = _contraste(a, b)
        if r < minimo:
            out.append(f"contraste {nombre} = {r:.1f} (mínimo {minimo})")
    return out