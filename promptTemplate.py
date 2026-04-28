# Ajudam a traduzir a entrada e os parâmetros do usuário em instruções para um modelo de linguagem
# Eles permitem que você crie mensagens de sistema, mensagens de usuário e mensagens de assistente, que são usadas para orientar o comportamento do modelo de linguagem.
# Eles também permitem que você use placeholders para inserir dinamicamente informações nas mensagens, o que é útil para criar mensagens personalizadas com base na entrada do usuário ou em dados externos.
# Os prompt templates são uma parte fundamental do processo de criação de aplicações de linguagem natural, pois ajudam a garantir que o modelo de linguagem receba as informações corretas e seja orientado de maneira eficaz para gerar respostas relevantes e precisas.

# PromptTemplate --> Simples, Formatam uma String, Recebem um dicionário de entrada e retornam uma string formatada.
# ChatPromptTemplate --> Formata uma lista de mensagens, Recebem um dicionário de entrada e retornam uma lista de mensagens formatada.
# MessagesPlaceholder --> Usado para inserir dinamicamente uma lista de mensagens em um ChatPromptTemplate, Recebe um dicionário de entrada e retorna uma lista de mensagens formatada.
# MessagePlaceholder é útil para inserir no histórico também!

from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)

print("===== Exemplo 1 =====")
prompt_template = PromptTemplate.from_template("Olá, {nome}! Bem-vindo ao LangChain.")
print(prompt_template.invoke({"nome": "Usuário"}))


print("\n===== Exemplo 2 =====")
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um professor de Geografia dando aula pra os seus alunos em uma classe lotada."),
    ("user", "Qual é a capital do Brasil?")
])  
print(chat_prompt_template.invoke({}))

print("\n===== Exemplo 3 =====")
historico = [
    ("user", "Qual é a capital do Brasil?"),
    ("assistant", "A capital do Brasil é Brasília."),
    ("user", "E qual é a capital da França?")
]
chat_prompt_template_com_historico = ChatPromptTemplate.from_messages([
    ("system", "Você é um professor de Geografia dando aula pra os seus alunos em uma classe lotada."),
    MessagesPlaceholder("historico"),
])

print(chat_prompt_template_com_historico.invoke({"historico": [HumanMessage(content=msg[1]) if msg[0] == "user" else AIMessage(content=msg[1]) for msg in historico]}))

print("\n===== Exemplo 4 =====")
print(" Criando a primeira Chain")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um professor de Geografia dando aula pra os seus alunos")])

chain1 = prompt_template | model

resposta = chain1.invoke({})
print(f"Resposta: {resposta.content}")