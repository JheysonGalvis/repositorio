from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import spacy
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el modelo de lenguaje de spaCy
nlp = spacy.load("es_core_news_sm")

# Crear la aplicación FastAPI
app = FastAPI()

# Montar la carpeta 'static' para servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Diccionario para almacenar el nombre del usuario
user_data = {"name": None}

# Rutas para servir las páginas HTML
@app.get("/", response_class=HTMLResponse)
async def get_home():
    return open("index.html", "r", encoding="utf-8").read()

@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    return open("chat.html", "r", encoding="utf-8").read()

@app.get("/portal", response_class=HTMLResponse)
async def get_portal():
    return open("portal.html", "r", encoding="utf-8").read()

# Modelo de datos para la consulta del usuario
class Question(BaseModel):
    question: str

# Base de datos de preguntas y respuestas
qa_data = [
    {"category": "Viabilidad por región", "phrase": "¿Qué zonas del país son más efectivas para instalar paneles solares?"},
    {"category": "Época de productividad", "phrase": "¿Cuándo son más productivos los paneles solares?"},
    {"category": "Incentivos", "phrase": "¿Qué incentivos hay para paneles solares?"},
    {"category": "Mantenimiento", "phrase": "¿Qué tipo de mantenimiento requieren los paneles solares?"},
    {"category": "Rentabilidad", "phrase": "¿En cuánto tiempo se recupera la inversión en paneles solares?"},
]

responses = {
    "Viabilidad por región": "Las zonas con mayor irradiación solar, como La Guajira, son ideales para paneles solares.",
    "Época de productividad": "Los meses con menos lluvias y más sol son los más productivos para los paneles solares.",
    "Incentivos": "Existen incentivos fiscales en Colombia para fomentar la instalación de paneles solares.",
    "Mantenimiento": "El mantenimiento consiste principalmente en limpieza y revisión de conexiones eléctricas.",
    "Rentabilidad": "La inversión en paneles solares se recupera en aproximadamente 5 a 10 años, dependiendo del consumo y la ubicación.",
}

# Vectorizar las frases para la búsqueda por similitud
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform([item["phrase"] for item in qa_data])

@app.post("/ask")
async def ask_question(question: Question):
    user_query = question.question.lower()
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, x_train).flatten()
    best_match_index = similarities.argmax()

    if similarities[best_match_index] > 0.2:  # Umbral de similitud
        category = qa_data[best_match_index]["category"]
        return {"response": responses.get(category, "No tengo información sobre eso.")}
    else:
        return {"response": "No encontré una respuesta exacta, pero puedo seguir aprendiendo."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

