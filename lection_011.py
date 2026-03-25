import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import geopandas as gpd

x = np.linspace(0, 10, 1000)

# fig, ax = plt.subplots()

# ax.plot(x, np.sin(x), "-k", label='синус')
# ax.plot(x, np.cos(x), "--r", label='косинус')
# ax.axis('equal')

#ax.legend(frameon=True, shadow=True, borderpad=1, loc='lower center', ncol=2)

# y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))

# lines = plt.plot(x, y)
# plt.legend(lines, ['первая', 'вторая', 'третья', 'четвертая'])

# plt.plot(x, y[:, 0], label='Первый')
# plt.plot(x, y[:, 1], label='второй')
# plt.plot(x, y[:, 2])
# plt.legend()

file_path = r"D:\Richard\Files\Personal files\Polytech\Additional\4th semester\Python\california cities\california_cities.csv"
cities = pd.read_csv(file_path)

states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
california = states[states['name'] == 'California']

latd = cities['latd']
longd = cities['longd']
population_total = cities['population_total']
area_total_km2 = cities['area_total_km2']

for geom in california.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        plt.plot(x, y, color='black', lw=1, zorder=1) 
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            y, x = poly.exterior.coords.xy
            plt.plot(x, y, color='black', lw=1, zorder=1)

scatter = plt.scatter(
    longd,
    latd,
    c = np.log10(population_total),
    s = area_total_km2,
    alpha=0.5,
    zorder = 2
)

cbar = plt.colorbar(scatter)



ticks = np.arange(np.floor(np.log10(population_total).min()), np.ceil(np.log10(population_total).max()) + 1)
cbar.set_ticks(ticks)

tick_labels = [f'{int(10**tick):,}' for tick in ticks]
cbar.set_ticklabels(tick_labels)
cbar.set_label('Население', fontsize=12)



plt.scatter([], [], s=100, c='k', alpha=0.5, label='100 $km^2')
plt.scatter([], [], s=300, c='k', alpha=0.5, label='300 $km^2')
plt.scatter([], [], s=500, c='k', alpha=0.5, label='500 $km^2')

plt.legend(frameon=False, labelspacing=2, title='Площадь')

plt.xlabel('Широта (latd)', fontsize=12)
plt.ylabel('Долгота (longd)', fontsize=12)
plt.title('Города Калифорнии: население (цвет) и площадь (размер)', fontsize=14)

plt.axis('equal')

# fig, ax = plt.subplots()

# lines = ax.plot(x, np.sin(x[:, np.newaxis] - np.pi / 2 * np.arange(0, 4)))
# ax.axis('equal')

# ax.legend(lines[:2], ['line A', 'line B'], loc='lower right')

# leg = mpl.legend.Legend(ax, lines[:2], ['line C', 'line D'], loc='upper right')

# ax.add_artist(leg)

# leg2 = mpl.legend.Legend(ax, lines[:2], ['line C', 'line D'], loc='upper left')

# ax.add_artist(leg2)

# y = np.sin(x) * np.cos(x[:, np.newaxis])

# plt.imshow(y, cmap='jet')
# plt.imshow(y, cmap='viridis')
# plt.imshow(y, cmap='RdBu')
# plt.colorbar()

# from sklearn.datasets import load_digits

# digits = load_digits(n_class = 6)
# print(digits)

# fig, ax = plt.subplot(8, 8)

# for i, ax_ in enumerate(ax.flax):
#     ax_.imshow(digits.images[i], cmap = 'binary')
#     ax_.set(xticks=[], yticks=[])

# from sklearn.manifold import Isomap

# iso = Isomap(n_components=2, n_neighbors = 10)
# prj = iso.fit_transform(digits.data)

# plt.scatter(
#     prj[:, 0],
#     prj[:, 1],
#     c = digits.target,
#     cmap = plt.colormaps.get_cmap('jet',6)
# )

# plt.colorbar(ticks = range(6))

plt.show()
