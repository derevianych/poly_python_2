import numpy as np
import timeit

# Слияние и разбиение массивов

# Одномерные
x = np.array([1, 2, 3])
y = np.array([4, 5])
z = np.array([6])

xyz = np.concatenate([x, y, z])
print(xyz)

# Двумерные массивы
x = np.array([[1, 2 ,3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
xy1 = np.concatenate([x, y]) # у подставится снизу х
print(xy1)

# xy1 = np.concatenate([x, y], axis = 0)
# print(xy1)

x2 = np.concatenate([x, y], axis = 1) # Подставляем вбок
print(x2)

# Трехмерный массив представляем в виде кубиков, методы для них:
# .vstack - "склейка" снизу
# .hstack - "склейка" сбоку
# .dstack - "склейка" сзади
# 0 измерение - строки, 1 - столбцы, 2 - вглубь кубика

x = np.array([[1, 2 ,3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])

print('vstack:')
print(np.vstack([x, y]))

print('hstack:')
print(np.hstack([x, y]))

print('dstack:')
print(np.dstack([x, y]))


# Разбиение массивов
# При разбиении указываем точки, в которых разбиваем массив

xy = np.vstack([x,y])
print(xy)

print(np.split(xy, [1])) # если не указать, то бьем по 0 измерению

print(np.split(xy, [2])) # Два равных массива

print(np.split(xy, [1], axis=1)) # Разрезаем по вертикали

# По аналогии с .vstack есть .vsplit, .hsplit, .dsplit

print(np.vsplit(xy, [2]))

print(np.hsplit(xy, [2]))

z = np.dstack([x, y])
print(z)

print(np.dsplit(z, [1]))


# Универсальные функции

x = np.arange(1, 10)
print(x)

def f(x):
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = 1.0/x[i]
    return out

# print(timeit.timeit(stmt="f(x)", globals=globals()))
# print(timeit.timeit(stmt="1.0/x", globals=globals()))
# универсальные функции примерно в 10 раз быстрее

# УФ. Арифметика

x = np.arange(5)
print(x)

print(x+1) # add
print(x-1) # substract
print(x*2) 
print(x/2)
print(x//2)
print(-x)
print(x**2) # power
print(x%2)
print(x*2-2)


print(x+1)
print(np.add(x,1)) # результат тот же

x = np.arange(-5,5)
print(x)

print(abs(x)) # py
print(np.abs(x)) #np
print(np.absolute(x)) #np

x = np.array([3 + 4j, 4 - 3j])
print(abs(x))
print(np.abs(x))

# УФ. Тригонометрия
# sin, cos, tan, arcsin, arccos, arctan

# УФ. Показательные и логарифмические
# exp, power, log, log2, log10, expm1

x = [0, 0.0001, 0.001, 0.01, 0.1]
print("exp = ", np.exp(x))
print("exp - 1 = ", np.expm1(x))

# print('log(x) = ', np.log(x)) # запускается с ошибкой из-за 0
print('log(x+1) = ', np.log1p(x))

# УФ. 

x = np.arange(5)
print(x)
y = x * 10
print(y)
y = np.multiply(x, 10)
print(y)

z = np.empty(len(x))
np.multiply(x, 10, out=z)
print(z)

x = np.arange(5)
z = np.zeros(10)
print(x)
print(z)
z[::2] = x * 10
# 0 0 10 0 20 0 30 0 40 0
print(z)

z = np.zeros(10)
np.multiply(x, 10, out=z[::2])
print(z)

# Сводные показатели

x = np.arange(1,5)
print(x)

print(np.add.reduce(x))
print(np.add.accumulate(x))

print(np.multiply.reduce(x))
print(np.multiply.accumulate(x))

print(np.subtract.reduce(x))
print(np.subtract.accumulate(x))

print(np.sum(x))
print(np.cumsum(x))

print(np.prod(x))
print(np.cumprod(x))

x = np.arange(1, 10)
print(np.add.outer(x, x))

print('ура таблуча умножения:')
print(np.multiply.outer(x, x))

# Агрегирование

np.random.seed(1)
s = np.random.random(100)
print(sum(s)) #python
print(np.sum(s)) #numpy

a = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(sum(a))
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))

print(type(a))
# Методы для массивов np
print(a.sum())
print(a.sum(0))
print(a.sum(1))

print(sum(a,2)) # 2 - начальное значение



# минимум/максимум
np.random.seed(1)
s = np.random.random(100)

print(min(s))
print(np.min(s))

print(max(s))
print(np.max(s))


# mean, std, var, median, argmin, argmax, percentile, any, all
# nan*

# Not a Number - NaN

# транслирование (broadcasting)

a = np.array([1, 2 ,3])
b = np.array([5, 5, 5])

print(a+b)
print(a+5)

# Не все массивы можно скрещивать
