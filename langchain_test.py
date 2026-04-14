from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)
pergunta = "Quais são os melhores restaurantes em São Paulo?"

prompt = PromptTemplate(
    input_variables=["pergunta"],
    template="Você é um assistente de viagem. Responda à pergunta: {pergunta}"
)

cadeia = prompt | modelo
resposta = cadeia.invoke({"pergunta": pergunta})
print(resposta)