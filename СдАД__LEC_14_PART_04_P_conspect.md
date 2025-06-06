# Визуализация данных и анализ

## Построение диаграммы рассеяния

Для анализа взаимосвязи между двумя переменными можно использовать **диаграмму рассеяния (точечную диаграмму)**. В данном случае необходимо построить диаграмму для *итогового рейтинга, который дали критики*, и *продаж по Евросоюзу*.

### Инструменты

Для построения диаграммы используется библиотека **matplotlib**. Необходимо передать в функцию *scatter* два аргумента: рейтинг и количество продаж.

```
![](images/СдАД__LEC_14_PART_04_P/000239s_top_7.jpg)
```

### Анализ диаграммы

После построения диаграммы необходимо сделать выводы на основе её анализа.

- **Выбросы**: на диаграмме можно увидеть, что большинство точек группируются в нижней части, но есть одна точка, которая сильно отличается от остальных. Это может быть выброс.
- **Взаимосвязь переменных**: диаграмма не позволяет сделать выводы о причинно-следственных связях между переменными.
- **Линейная взаимосвязь**: на диаграмме нет тенденции к тому, чтобы точки выстраивались в прямую линию, поэтому нельзя сказать, что между переменными существует сильная положительная или отрицательная линейная взаимосвязь.

## Определение типа графика

На основе предложенной картинки необходимо определить, какой из графиков на ней изображён.

### Варианты графиков

```mermaid
classDiagram
    class Графики {
        Диаграмма рассеяния переменных: продажи в Японии и количество пользователей
        Гистограмма переменной по количеству пользователей
        Диаграмма рассеяния переменных: продажи в Северной Америке и Евросоюзе
        Диаграмма рассеяния переменных: рейтинг, который поставили критики, и количество критиков
    }
```

### Анализ

После построения всех трёх диаграмм можно сделать вывод, что наиболее подходящей является диаграмма рассеяния для переменных о продажах в Северной Америке и о продажах в Евросоюзе.

```
![](images/СдАД__LEC_14_PART_04_P/XXXXXXs_top_YYY.jpg)
```
