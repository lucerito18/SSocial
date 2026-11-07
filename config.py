import os

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Si Groq cambia los nombres, puedes sobrescribirlos desde Render sin tocar el código
MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "openai/gpt-oss-120b")

MAX_INTENTOS = 3      # cuántas veces el Jefe deja reintentar a un agente
TOPE_IMAGENES = 12    # tope absoluto de imágenes por presentación
TITULO_FUENTES = "Fuentes y créditos"
# Wikimedia pide identificar a quien usa su API. Opcional: define CONTACTO en Render.
USER_AGENT = "PDF2SlidesAgents/1.0 (proyecto de servicio social; uso educativo) " + os.environ.get("CONTACTO", "")