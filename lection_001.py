# Библиотека numpy

import sys
import array
import numpy as np
print('Версия NumPy',np.__version__)

# Почему numpy?
# Потому что динамическая типизация в пайтон мешает

# динамическая типизация
x = 1
print(x)
print('Тип переменной ',type(x))

x = 'hello'
print(x)
print('Тип переменной ',type(x))

l = [True, '2', 3.0, 4] # В списке элементы разных типов
print(l)
print('Память занятая списком: ',sys.getsizeof(l),'бит')

l1 = []
print('Тип пустого списка ',type(l1))
print('Память занятая списком: ',sys.getsizeof(l1),'бит')

a1 = array.array('i', [])
print('Массив', a1)
print('Тип массива',type(a1))
print('Память занятая массивом: ',sys.getsizeof(a1))

a2 = array.array('i', [1])
print('Массив', a2)
print('Тип массива',type(a2))
print('Память занятая массивом: ',sys.getsizeof(a2))

# NumPy и Array хранят элементы одного типа данных

# массив в numpy можно создать из списка
l = [1,2,3,4,5,6,7,8,9,0]
a = np.array(l)
print('Массив из списка',a)
print('Тип массива', type(a))

# Почему не массив array, а массив из numpy?
print('list(python)', sys.getsizeof(l))
ap=array.array('i', l)
print('array(python)', sys.getsizeof(ap))
print('array(numpy)', sys.getsizeof(a))
# На маленьких данных и данных одного типа незаметно, но при больших
# данных массив типа numpy.ndarray будет весить меньше питоновского

# Повышающее приведение типов
a = np.array([1.01, 2, 3, 4, 5, "a"])
print(type(a), a)

# Явно задаем тип
a = np.array([1.99, 2, 3, 4, 5], dtype=int)
print(type(a), a)

# Одномерные массивы. range(i, j) -> i, i+1, ..., j-1
a = np.array(range(2, 5))
print(a)

# многомерные массивы
a = np.array([range(i, i+5) for i in [1, 2, 3, 6]])
print(a)

# Создание масива с нуля:

# из 0
print(np.zeros(10, dtype=int))

# из 1
print(np.ones((3, 5), dtype=int))

# предопределенное значение
print(np.full((3,3), 3.1416))

# Линейная последовательность чисел
print(np.arange(0, 20, 2))
print(np.arange(0, 22, 3))

# В интервале с одинаковыми промежутками по количеству элементов
print(np.linspace(0, 2, 5)) # Пять элементов в интервале от 0 до 2
print(np.linspace(0, 1, 11))

# равномерное распределение от 0 до 1
print(np.random.random((2, 4)))

print(np.random.normal(0, 1, (2,4)))

# арвномерное распределение от х до у
print(np.random.randint(0, 5, (2,2)))

# единичная матрица
print(np.eye(5, dtype=int))



# Типы данных в NumPy
a1 = np.zeros(10, dtype=int)
a2 = np.zeros(10, dtype='int16')
a3 = np.zeros(10, dtype=np.int16)
print(a1, type(a1), a1.dtype) #python
print(a2, type(a2), a2.dtype) #np
print(a3, type(a3), a3.dtype) #np

# a1 = np.zeros(10, dtype=int16) - не делаем так
# NameError: name 'int16' is not defined

# Numerical Python = NumPy
# - атрибуты массивов
# - индексация
# - срезы
# - изменение формы
# - объединение и разбиение

# Атрибуты: ndim - число размерностей, shape - размер каждой размерности,
#           size - общий размер массива

np.random.seed(1)

x1 = np.random.randint(10, size = 3)
print(x1)
print(x1.ndim, x1.shape, x1.size)
# 1, (3,), 3

x2 = np.random.randint(10, size = (3,2))
print(x2)
print(x2.ndim, x2.shape, x2.size)

x3 = np.random.randint(10, size = (3,2,2))
print(x3)
print(x3.ndim, x3.shape, x3.size)

# Индексация

# одномерные массивы
a = np.array([1, 2, 3, 4, 5])
print(a[0])

print(a[-2])

a[1] = 20

print(a)

# многомерные массивы
a = np.array([[1,2], [3,4]])
print(a)

print(a[0,0])
# 1
print(a[-1,-2])
# 3

a[1,0] = 100
print(a)

# вставки

a = np.array([1, 2, 3, 4, 5])
print(a.dtype)

a[0] = 3.14
print(a)
print(a.dtype)

a.dtype = float
print(a)
print(a.dtype)



# Срез - подмассив массива [начало:конец:шаг] - [0:<конец>:1]

a = np.array([1, 2, 3, 4, 5])

print(a[:3])

print(a[3:])

print(a[1:4])

print(a[::2])

print(a[1::2])

# шаг < 0 [начало:конец:шаг] -> [конец:начало:шаг]

a = np.array([1, 2, 3, 4, 5])
print(a[::-1])

# Срезы в многомерных массивах
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 , 11, 12]])
print(a)

print(a[:2 ,:3])

print(a[::, ::2])

print(a[::-1, ::-1])

print(a[:, 0]) # Первый столбец

print(a[0, :]) # Первая строка

# Срезы в Python - копии подмассивов, в NumPy - представления (view)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 , 11, 12]])
print(a)

a_2x2 = a[:2, :2]
print(a_2x2)

a_2x2[0, 0] = 999
print(a)

a_2x2 = a[:2, :2].copy() # Больше не влияет на основной массив
print(a_2x2)

a_2x2[0, 0] = 1000 - 7
print(a_2x2)
print(a)

# Форма массива. Изменение

a = np.arange(1, 13)
print(a, a.shape, a.ndim)

print(a[3])
print(a[11])

a1 = a.reshape(1, 12)
print(a1, a1.shape, a1.ndim)

print(a1[:, 3:12:8]) # 4 и 12 сразу
print(a1[0, 3])
print(a1[0, 11])

a2 = a.reshape(2, 6)
print(a2, a2.shape, a2.ndim)

a3 = a.reshape(2, 2, 3)
print(a3, a3.shape, a3.ndim)
print(a3[0,1,2])

a4 = a.reshape(1, 12, 1, 1)
print(a4, a4.shape, a4.ndim)
print(a4[0, 2, 0, 0])

a5 = a.reshape(2,6)
print(a5, a5.shape, a5.ndim)
print(a5[1,4])

a6 = a.reshape((2,6), order='F') # в честь Fortran
print(a6, a6.shape, a6.ndim)
print(a6[1,4])

a = np.arange(1, 13)
print(a, a.shape, a.ndim)

a1 = a.reshape(1, 12)
print(a1, a1.shape, a1.ndim)

a2 = a[np.newaxis, :]
print(a2, a2.shape, a2.ndim)

a3 = a[:, np.newaxis]
print(a3, a3.shape, a3.ndim)
