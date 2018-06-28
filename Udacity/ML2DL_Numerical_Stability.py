x = 1000000000

for i in range(1000000):
    x += 1e-6

x -= 1000000000

print(x)