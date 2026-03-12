import numpy as np
import pandas as pd

print(pd.date_range('2026-01-01', periods=4, freq="ME"))

print(pd.date_range('2026-01-01', periods=4, freq="MS"))

print(pd.date_range('2026-01-01', periods=4, freq="QE"))

print(pd.date_range('2026-01-01', periods=4, freq="QS"))

print(pd.date_range('2026-01-01', periods=4, freq="W"))

print(pd.date_range('2026-01-01', periods=4, freq="4W-MON"))

file_path = r'D:\Richard\Files\Personal files\Polytech\Additional\4th semester\Python\index\index.csv'
ind = pd.read_csv(file_path, sep=';')

print(ind)

print(type(ind))
print(ind.dtypes)

index = pd.DatetimeIndex(ind["Date"])

ind.index = index
ind = ind['Close']
print(ind.head())

import matplotlib.pyplot as plt

ind.plot()
plt.ylabel('График торгов')
# plt.show()

file_path = r'D:\Richard\Files\Personal files\Polytech\Additional\4th semester\Python\bycicles\FremontBridge.csv'
df = pd.read_csv(
    file_path,
    index_col="Date",
    parse_dates=True,
    # date_format='%Y-%m=%d %I:%M:%S %p'
    date_format='%m/%d/%Y %I:%M:%S %p'
)
print(df.head())
print(df.dtypes)

print(df.columns)
df.columns = ["Total", "East", "West"]
print(df.head())

print(df.describe())
print(df.dropna().describe())

import matplotlib.pyplot as plt

#df.plot()
#plt.ylabel("Кол-во велосипедистов в час")
# plt.show()

weekly = df.resample("W").sum()
weekly.plot(style=['-', ':', '--',])
plt.ylabel("Кол-во велосипедистов в неделю")
#plt.show()

daily = df.resample("D").sum()
# center = False -> прошлые значения от выбранного
# center = True - прошлые и будущие значения от выбранного
daily.rolling(30, center=True).mean().plot(style=['-', ':', '--',])
plt.ylabel("Среднее месячное кол-во велосипедистов (скользящее окно)")
#plt.show()


daily = df.resample("D").sum()
# center = False -> прошлые значения от выбранного
# center = True - прошлые и будущие значения от выбранного
daily.rolling(30, center=True, win_type="gaussian").mean(std=5).plot(style=['-', ':', '--',])
plt.ylabel("Среднее месячное кол-во велосипедистов (скользящее окно, гауссово распределение)")
# plt.show()

timely = df.groupby(df.index.time).mean()
ticks = 60*60*4*np.arange(6)
print(ticks)
timely.plot()
plt.ylabel('Кол-во великов в зависимости от времени дня')
#plt.show()

weekly = df.groupby(df.index.dayofweek).mean()
weekly.plot()
plt.ylabel('Кол-во великов по дням недели')
#plt.show()

w1 = np.where(df.index.weekday < 5, 'Будни', 'Выходные')
print(w1.shape)
t1 = df.groupby([w1, df.index.time]).mean()

fig,ax = plt.subplots(1, 2)
ax[0].set_ylim(0, 600)
t1.loc['Будни'].plot(ax=ax[0], title = 'Будни')

ax[1].set_ylim(0, 600)
t1.loc['Выходные'].plot(ax=ax[1], title = 'Выходные')
# plt.show()



# MATPLOTLIB


# pip install matplotlib
# pip install pyqt5
# pip install IPython

import matplotlib.pyplot as plt

plt.style.use('classic')

plt.show() # ! ДОЛЖНО БЫТЬ ОДНО, НЕЛЬЗЯ НЕСКОЛЬКО
