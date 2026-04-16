from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)

mensagens = [
    SystemMessage("You are a helpful assistant."),
]

hello = model.invoke(mensagens + [HumanMessage("Hello, how are you?")])

print(f"Model: {hello.content}")

while True:
    user_input = input("You: ")
    mensagens.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    response = model.invoke(mensagens)
    mensagens.append(AIMessage(content=response.content))
    print(f"Model: {response.content}")