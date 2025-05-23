# Визуализация данных частотной таблицы

## Гистограмма

### Построение гистограммы для первой группы школьников

Для представления данных в графическом виде можно использовать *гистограмму*. Рассмотрим построение гистограммы на примере распределения частот для первой группы школьников.

У нас есть 6 вариантов оценок. На графике расставляем числа от 1 до 6 и начинаем рисовать столбики гистограммы:

- единичка встречается 4 раза — рисуем столбец высоты 4;
- двойка встречается 6 раз — рисуем столбец высоты 6;
- тройка встречается 3 раза — рисуем столбец высоты 3;
- четвёрка встречается 3 раза — рисуем столбец такой же высоты;
- пятёрка и шестёрка встречаются по два раза — рисуем столбцы высоты 2.

![](images/СдАД__LEC_04_PART_02_T/000239s_top_7.jpg)

```mermaid
barChart
    title Гистограмма для первой группы школьников
    x-axis Оценки
    y-axis Частота
    bar "1" : 4
    bar "2" : 6
    bar "3" : 3
    bar "4" : 3
    bar "5" : 2
    bar "6" : 2
```
*Гистограмма показывает распределение частот оценок в первой группе школьников.*

### Построение гистограммы для второй группы школьников

Теперь построим гистограмму для второй группы:

- единичка встречается 2 раза;
- двойка встречается 3 раза;
- тройка встречается 1 раз;
- четвёрка встречается 7 раз;
- пятёрка встречается 4 раза;
- шестёрка встречается 3 раза.

```mermaid
barChart
    title Гистограмма для второй группы школьников
    x-axis Оценки
    y-axis Частота
    bar "1" : 2
    bar "2" : 3
    bar "3" : 1
    bar "4" : 7
    bar "5" : 4
    bar "6" : 3
```
*Гистограмма для второй группы позволяет сравнить распределение оценок с первой группой.*

## Полигон частотного распределения

Кроме варианта с гистограммой, есть возможность визуализации частот через *полигон частотного распределения*. Он рисуется намного проще. Для того чтобы изобразить полигон, нам нужно просто отметить соответствующие точки для первой группы:

- напротив единички стоит четвёрка;
- напротив двойки — шестёрка;
- напротив тройки — тройка;
- напротив четвёрки — тоже тройка;
- для пятёрки и шестёрки — двойки.

```mermaid
lineChart
    title Полигон частотного распределения для первой группы
    x-axis Оценки
    y-axis Частота
    series "Частота"
    1: 4
    2: 6
    3: 3
    4: 3
    5: 2
    6: 2
```
*Полигон частотного распределения показывает связь между оценками и их частотами.*

## Гистограмма с группировкой

Иногда строят гистограмму по данным с группировкой. Например, можно объединить оценки в более крупные группы:

- один или два балла;
- три или четыре балла;
- пять или шесть баллов.

Для первой группы:

- один или два складываем, у нас получается частота 10;
- три или четыре складываем, частота 6;
- пять или шесть складываем, частота 4.

Для второй группы:

- один или два, частота 5;
- три или четыре (один плюс семь), частота 8;
- пять или шесть складываем, четыре и три, получаем семь.

```mermaid
barChart
    title Гистограмма с группировкой для первой группы
    x-axis Группы оценок
    y-axis Частота
    bar "1-2" : 10
    bar "3-4" : 6
    bar "5-6" : 4
```

```mermaid
barChart
    title Гистограмма с группировкой для второй группы
    x-axis Группы оценок
    y-axis Частота
    bar "1-2" : 5
    bar "3-4" : 8
    bar "5-6" : 7
```
*Гистограмма с группировкой позволяет упростить анализ данных, объединяя близкие значения.*