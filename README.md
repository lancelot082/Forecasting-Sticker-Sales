# Rohlik-Sales-Forecasting-Challenge
![image](https://github.com/user-attachments/assets/f29dc3f7-1039-4ea7-a20c-0647296fe781)


## Overview
Your Goal: The objective of this challenge is to forecast sticker sales in different countries.

___

Ваша цель: Задача состоит в том, чтобы спрогнозировать продажи наклеек в разных странах.
___

## EDA
### lineplot
<pre>
  ```
# Задаем параметры фигуры
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

# Генерируем графики по годам
for i, year in enumerate(range(2010, 2016)):
    ax = axes[i // 3, i % 3]  # Определяем положение в сетке
    sns.lineplot(
        data=train[train['date'].dt.year == year],
        x=train[train['date'].dt.year == year]['date'].dt.month,
        y='num_sold',
        ax=ax
    )
    ax.set_title(f'Sales in {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Num Sold')

# Очищаем и упорядочиваем
plt.tight_layout()
  ```
</pre>
![image](https://github.com/user-attachments/assets/e9904549-8a5a-46ad-aad3-a120cf10241b)


<pre>
  ```
fit, axes = plt.subplots(1, 3, figsize = (18, 6))

sns.lineplot(data = train, x = train['date'].dt.year, y = train['num_sold'], hue = train['country'], ax = axes[0])
axes[0].set_title('Sold Year-Country')

sns.lineplot(data = train, x = train['date'].dt.year, y = train['num_sold'], hue = train['store'], ax = axes[1])
axes[1].set_title('Sold Year-Store')

sns.lineplot(data = train, x = train['date'].dt.year, y = train['num_sold'], hue = train['product'], ax = axes[2])
axes[2].set_title('Sold Year-Product')
  ```
</pre>
![image](https://github.com/user-attachments/assets/bd87ccc4-eaec-40fa-9eaa-09e6dc68a75a)


### barplot
<pre>
  ```
fig, axes = plt.subplots(3, 1, figsize = (8, 12))

sns.barplot(data = train, x = train['country'], y = train['num_sold'], ax = axes[0])
axes[0].set_title('Sold country')

sns.barplot(data = train, x = train['store'], y = train['num_sold'], ax = axes[1])
axes[1].set_title('Sold store')

sns.barplot(data = train, x = train['product'], y = train['num_sold'], ax = axes[2])
axes[2].set_title('Sold product')

plt.tight_layout()
  ```
</pre>
![image](https://github.com/user-attachments/assets/4bc3de49-f2d4-4528-bd60-a8c477b8f67c)


### seasonal_decompose
<pre>
  ```
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

series = train.copy()

# Преобразуем дату в индекс и выбираем столбец
series.set_index('date', inplace=True)

# Группируем данные по индексу (дате) и агрегируем
series = series.groupby(series.index).sum()

series = series['num_sold'].asfreq('D')  # Устанавливаем дневную частоту

# Задаем размер графика
rcParams['figure.figsize'] = 11, 9

# Применяем seasonal_decompose
decompose = seasonal_decompose(series, model='additive', period=365)
decompose.plot()
plt.show()
  ```
</pre>
![image](https://github.com/user-attachments/assets/3fd03bb7-c0fb-4de0-8f2c-d6d3b9b89aaf)

