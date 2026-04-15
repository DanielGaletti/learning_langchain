from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Literal, TypedDict
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)

prompt_consultor = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor de investimentos especializado em ações brasileiras."),
    ("user", "{query}")
])

prompt_consultor_cripto = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor de investimentos especializado em criptomoedas."),
    ("user", "{query}")
])

cadeia_acoes = prompt_consultor | modelo | StrOutputParser()
cadeia_cripto = prompt_consultor_cripto | modelo | StrOutputParser()

class Rota(TypedDict):
    investimento: Literal["acoes", "cripto"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Você é um roteador de perguntas. Classifique a pergunta como 'acoes' ou 'cripto'. Responda apenas com a classificação."),
    ("user", "{query}")
])

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def response(pergunta: str) -> str:
    rota = roteador.invoke({"query": pergunta})
    if rota["investimento"] == "acoes":
        return cadeia_acoes.invoke({"query": pergunta})
    elif rota["investimento"] == "cripto":
        return cadeia_cripto.invoke({"query": pergunta})
    else:
        return "cripto"
    
print(response("Quais são as melhores ações para investir em 2024?"))