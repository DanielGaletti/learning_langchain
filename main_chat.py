import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)

memoria = {}
sessao = "langchain_test"

def historico(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = ChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    "Quero visitar um bairro de são Paulo, onde posso ir?",
    "Quais são os melhores restaurantes em São Paulo?",
]

prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de viagem specializado em São Paulo, se apresente como Mr. Viagem."),
    ("placeholder", "{historico}"),
    ("human", "{pergunta}")
])

cadeia = prompt_sugestao | modelo | StrOutputParser()
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico,
    input_messages_key="pergunta",
    history_messages_key="historico"
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "pergunta": pergunta,
        },
        config={"configurable": {"session_id": sessao}}
    )
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}")
    print("-" * 50)
