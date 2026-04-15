# Created By: Daniel Galetti
# Without AI
#Just a simple example of RunnableLambda, RunnableSequence, RunnableParallel and RunnablePassThrough.
#danielgaletti70@gmail.com

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

def add_one(x):
    return x + 1

runnable = RunnableLambda(add_one)

print(runnable.invoke(5)) # 5 + 1 = 6

# -------------------------------------------------- 
# Agora em uma cadeira sequencial:

print("-" * 50)
def mult_two(x):
    return x * 2

sequence = RunnableLambda(add_one) | RunnableLambda(mult_two)

print("RunnableSequence: " + str(sequence.invoke(5)))  # (5 + 1) * 2 = 12

# -------------------------------------------------- 
# Agora testando o RunnableParalelo:

print("-" * 50)
def mult_three(x):
    return x * 3

sequence = RunnableLambda(add_one) | {
    "mul_two": RunnableLambda(mult_two),
    "mul_three": RunnableLambda(mult_three)
}

print("RunnableParallel: " + str(sequence.invoke(5)))  # {'mul_two': (5 + 1) * 2, 'mul_three': (5 + 1) * 3} = {'mul_two': 12, 'mul_three': 18}

# --------------------------------------------------
# Agora testando o RunnablePassThrough:

print("-" * 50)
chain = RunnablePassthrough() | RunnablePassthrough() | RunnablePassthrough()
print("RunnablePassThrough: Olá, mundo!")  # "Olá, mundo!" passa por cada RunnablePassThrough sem alterações

def entrada_maiuscula(x: str) -> str:
    return x.upper()

chain2 = RunnablePassthrough() | RunnableLambda(entrada_maiuscula) | RunnablePassthrough()
# Se no lugar do RunnableLambda(entrada_maiuscula) colocarmos um RunnablePassthrough, 
# a resposta seria a mesma que a entrada, ou seja, "Olá, mundo!".

resposta = chain2.invoke("Olá, mundo!")
print("Resposta em letra maiúscula: " + resposta)  # "Olá, mundo!" passa por cada Runnable

# --------------------------------------------------
# Operador Assign

print("-" * 50)

runnable = RunnablePassthrough() | RunnablePassthrough.assign(multiplica_3=lambda x: x["num"] * 3)
resposta = runnable.invoke({"num": 5})

print("Resposta do RunnablePassthrough.assign: " + str(resposta))  # {'num': 5, 'multiplica_3': 15}
print("-" * 50)


# Exercício 

runnable1 = RunnablePassthrough()

def count_characters(x):
    return len(x["input"])

runnable2 = RunnableLambda(count_characters)
runnable3 = RunnableLambda(lambda x: "Conseguiu " + x["original"]["input"])

chain = runnable1 | RunnablePassthrough.assign(
    original=RunnablePassthrough(),
    num_caract=runnable2,
) | RunnableParallel({
    "passa_para_frente": RunnablePassthrough(),
    "mensagem": runnable3
})

print(chain.invoke({"input": "Parabéns Você"}))