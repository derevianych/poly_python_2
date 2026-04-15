import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x, y):
    return np.sin(np.sqrt(x**2+y**2))


# x = np.linspace(-6, 6, 30)
# y = np.linspace(-10, 10, 50)

# X, Y = np.meshgrid(x, y)

# Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = plt.axes

# ax.scatter3D(X, Y, Z, c=Z)
# ax.plot_wireframe(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap='viridis')

# angle = np.linspace(0, 2 * np.pi, 50)
# r = np.linspace(0, 6, 30)
# R, Angle = np.meshgrid(r, angle)

# X = R * np.sin(Angle)
# y = R * np.cos(Angle)
# z = f(X, Y)

# ax.plot_surface(X, Y, Z, cmap='viridis')

# angle = np.linspace(0, 1.5 * np.pi, 50)
angle = 1.5 * np.pi * np.random.random(1000)
r = np.linspace(0, 6, 1000)

R, Angle = np.meshgrid(r, angle)

X = R * np.sin(Angle)
Y = R * np.cos(Angle)
Z = f(X, Y)

x = r * np.sin(angle)
y = r * np.cos(angle)
z = f(x,y)

# ax.scatter3D(X, Y, Z, c=Z)
# ax.plot_surface(X, Y, Z, cmap='viridis')
ax.plot_trisurf(x, y, z, cmap='viridis')

# plt.show()






import seaborn as sns

sns.set_style('darkgrid')
#path = r"D:\Richard\Files\Personal files\Polytech\Additional\04th semester\Python\cars\cars.csv"
path = r"D:\Richard\Files\Personal files\Polytech\Additional\4th semester\Python\cars\cars.csv"
cars = pd.read_csv(path)

print(cars.head())

## Числовые данные
## парная диаграмма

## sns.pairplot(cars)

## sns.pairplot(data = cars, hue = 'transmission')

## Тепловая карта

# cars_corr=cars[['year', 'selling_price', 'seats', 'mileage']]

# sns.heatmap(cars_corr.corr(), cmap='viridis', annot = True)

# д. рассеяния

# sns.scatterplot(x='seats', y='mileage', data=cars)
# sns.scatterplot(x='year', y='selling_price', data=cars)
# sns.scatterplot(x='seats', y='mileage', data=cars, hue='fuel')

## Д. рассеяния + лин. регрессия
# sns.regplot(x='seats', y='mileage', data=cars)
# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter')

#sns.relplot(x='seats', y='mileage', data=cars, kind='line', col = 'transmission', col_wrap=2, hue='fuel')
#sns.lmplot(x='seats', y='mileage', data=cars, col = 'transmission', col_wrap=2, hue='fuel')

## Линейный график
#sns.lineplot(x='seats', y='mileage', data=cars, hue='fuel')

# Сводная диаграмма


# sns.jointplot(x='year', y='selling_price', data = cars)

# sns.jointplot(x='year', y='selling_price', data = cars, kind = 'kde')

# sns.jointplot(x='year', y='selling_price', data = cars, kind = 'hex')

# sns.jointplot(x='year', y='selling_price', data = cars, hue='transmission')

### Категории и ччсила

# sns.barplot(x='fuel', y='selling_price', data=cats, estimator=np.mean)

# sns.barplot(x='fuel', y='selling_price', data=cats, estimator=np.mean, hue = 'transmission')

# sns.catplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue = 'transmission', kind='bar', col='seller_type', col_wrap=2)

# sns.pointplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue = 'transmission')

# sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# g = sns.catplot(x='fuel', y='selling_price', data=cars, kind = 'box')

# sns.stripplot(x='fuel', y='selling_price', data=cars, ax=g.ax)

plt.show()




