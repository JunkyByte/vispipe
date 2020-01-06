# Just a generator chain
import numpy as np

data = range(10)
gen1 = (x ** 2 for x in data)
gen2 = (x / 2 for x in gen1)
print(list(gen2))

# Test generator functions

def square_gen(x):
    for value in x:
        yield value ** 2

def halved_gen(x):
    for value in x:
        yield value / 2

data = range(10)
gen1 = square_gen(data)
gen2 = halved_gen(gen1)
print(list(gen2))
