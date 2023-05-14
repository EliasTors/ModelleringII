def f(x):
    return (6*x-3)/(x-1)

h = 0.00001
def Df(x):
    return (f(x+h)-f(x))/h

a = 1.5
while Df(a)<-3:
    a = a+0.001
    
b = f(a) - Df(a)*a
print(b                                                                                                                                                                                                                                                )