# Обработка данных с помощью функции из библиотеки SciPy Stats

## Функции для независимых выборок

В библиотеке SciPy Stats есть два основных способа работы с независимыми выборками:

1. **ttst_int для независимых выборок (independent from stats)**:
   - Необходимо передать уже вычисленные статистики:
     - среднее значение для первой группы;
     - среднеквадратичные отклонения для первой группы;
     - количество наблюдений в первой группе;
     - аналогичные значения для второй группы.
   - Можно указать параметр `equal_var` для указания равенства дисперсий выборок (например, `equal_var=False` для неравных дисперсий).

   ![](images/СдАД__LEC_11_PART_05_P/000239s_top_7.jpg)

2. **Т-тест для независимых выборок (без добавления from stats)**:
   - Передаются непосредственно независимые выборки.
   - Указывается, что дисперсии не равны.

```mermaid
classDiagram
    class "ttst_int для независимых выборок" {
        +передача вычисленных статистик
        +указание параметра equal_var
    }
    class "Т-тест для независимых выборок" {
        +передача независимых выборок
        +указание неравных дисперсий
    }
    "ttst_int для независимых выборок" --|> "Т-тест для независимых выборок": альтернатива
```

*Диаграмма показывает два основных способа работы с независимыми выборками в SciPy Stats.*

## Пример расчёта t-критерия Велча

Для расчёта t-критерия Велча с поправкой на неравные дисперсии используется параметр `equal_var=False`. Это позволяет получить значение t-статистики, соответствующее расчёту вручную.

## Доверительный интервал для разницы между средними

Для расчёта доверительного интервала используется 95% интервал. Разница между выборочными средними умножается на критическое значение t и стандартную ошибку.

```mermaid
flowchart LR
    A[Нахождение разницы между выборочными средними] --> B[Умножение на критическое значение t и стандартную ошибку]
    B --> C[Определение нижней и верхней границ доверительного интервала]
```

*Диаграмма иллюстрирует процесс расчёта доверительного интервала для разницы между средними.*

### Пример расчёта

1. Находим разницу между выборочными средними.
2. Умножаем критическое значение t на стандартную ошибку.
3. Определяем нижнюю и верхнюю границы доверительного интервала.

## Вывод

Мы рассчитали t-критерий для двух независимых выборок с поправкой Велча для выборок с неравными дисперсиями. Значение t-статистики попадает в критическую область, что позволяет отвергнуть нулевую гипотезу о разнице средних для двух выборок. Также мы вычислили доверительный интервал для разницы средних, который позволяет с 95% вероятностью утверждать, что разница между средними находится в интервале от 117 до 152.