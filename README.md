# Forecasting Sticker Sales
![image](https://github.com/user-attachments/assets/f29dc3f7-1039-4ea7-a20c-0647296fe781)

Цель проекта — спрогнозировать продажи (`num_sold`) для различных комбинаций **страны**, **магазина** и **товара** на ежедневной основе. Прогнозируемый период — **2017–2019 гг.**, тренировочные данные даны с 2010 по 2016.

В проекте реализованы **две независимые стратегии прогнозирования**:

1. Градиентный бустинг на признаках времени (XGBoost + Optuna)
2. Классическая модель Exponential Smoothing на агрегированных временных рядах

---

## 1. EDA (исследовательский анализ)

- Исследована сезонность по **годам, месяцам и дням**
- Построены lineplot и barplot по категориям: `country`, `store`, `product`
- Использован `seasonal_decompose` из `statsmodels` для анализа трендов и сезонных колебаний
![image](https://github.com/user-attachments/assets/44785682-734a-45d6-a1dd-236757cff679)
![image](https://github.com/user-attachments/assets/4219cce3-4889-4f37-a8cc-51ac7810e3a1)

Это позволило подтвердить:
- Выраженную **годовую и месячную сезонность**
- Стабильные паттерны спроса по категориям
- Отсутствие выбросов и пропущенных дат в пределах каждой временной серии
![image](https://github.com/user-attachments/assets/c7805a77-5cd4-401c-9879-e710cd8c72fe)
![image](https://github.com/user-attachments/assets/17d18fd1-5974-4628-af76-5e826d539315)

---

## 2. Подход №1 — Модель XGBoost с признаками времени

Машинное обучение используется для предсказания логарифма продаж (`log1p(num_sold)`), а затем значения восстанавливаются в исходную шкалу. Основные этапы:

### Временные признаки

Из временной метки извлечены:
- `year`, `month`, `day`, `day_of_week`, `day_of_year`
- `week_of_year`, `linear_trend`, `squared_trend`
- Периодические признаки через `sin`/`cos`

```python
df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['linear_trend'] = (df[date_column] - df[date_column].min()).dt.days
```

### Препроцессинг

- Категориальные признаки: `country`, `store`, `product` — One-Hot Encoding
- Пропущенные значения отсутствуют
- Целевая переменная логарифмирована: `log1p(num_sold)`

### Оптимизация гиперпараметров

Использован `Optuna` с кросс-валидацией по времени (`TimeSeriesSplit`). Метрика — **MAPE (Mean Absolute Percentage Error)**.

```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=60)
```

### Финальная модель

- Обучена на всех тренировочных данных
- Предсказания на тест: результат переводится обратно из логарифма
- Визуализация важности признаков (`feature_importances_`)

**Плюсы подхода**:
- Гибкость и масштабируемость
- Учитывает нелинейные зависимости
- Возможность легко добавлять дополнительные фичи

---

## 3. Подход №2 — Exponential Smoothing (ETS)

Использована модель `Holt-Winters` из `statsmodels`:

- **Агрегация** данных в матрицу: один временной ряд на каждую комбинацию `(country, store, product)`
- Обработка каждой временной серии отдельно
- Прогноз на 3 года (2017–2019) с годовой сезонностью (365 дней)

```python
model = ExponentialSmoothing(
    np.sqrt(train_data),
    trend='add',
    seasonal='add',
    seasonal_periods=365
)
forecast = model.fit().forecast(steps=steps) ** 2
```

### Постобработка

- Прогноз объединяется с тестом по ключам `date`, `country`, `store`, `product`
- Подготовлен `submission.csv` в требуемом формате

### Валидация на 2016 году

Для оценки качества ETS-модели был проведён прогноз только на 2016:
- MAPE, RMSE, MAE рассчитаны между прогнозом и фактом
- Построен график `fact vs forecast`

**Плюсы подхода**:
- Прозрачность и интерпретируемость
- Учет сезонности напрямую
- Не требует генерации фичей или кодирования категорий

---
### Score
- XGBoost - 0.10661
- ETS - 0.15488
