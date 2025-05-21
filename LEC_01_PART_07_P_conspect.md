# Работа с данными в Pandas

## Открытие данных

Pandas позволяет работать с различными форматами данных. Среди поддерживаемых форматов:

- файлы Excel;
- текстовые файлы с разделителями (CSV, TSV и другие).

### Открытие файлов Excel

Для открытия файлов Excel используется функция `read_excel`. Пример использования:

```python
from pandas import read_excel

df = read_excel('data/coffee_stat.xlsx')
```

### Открытие текстовых файлов с разделителями

Для открытия текстовых файлов с разделителями используется функция `read_csv`. Пример использования:

```python
from pandas import read_csv

df = read_csv('data/students.csv')
```

#### Спецификация разделителя

По умолчанию предполагается, что данные разделены запятой. Однако разделителем может быть и другой символ, например, точка с запятой или табуляция. В этом случае необходимо указать разделитель в аргументе `sep`. Пример использования:

```python
from pandas import read_csv

df = read_csv('data/coffee.csv', sep=';')
```

#### Отсутствие заголовков столбцов

Если в файле нет заголовков столбцов, можно указать это в аргументе `header`. Пример использования:

```python
from pandas import read_csv

df = read_csv('data/coffee.csv', sep=';', header=None)
```

Затем можно задать свои заголовки столбцов с помощью аргумента `names`. Пример использования:

```python
from pandas import read_csv

df = read_csv('data/coffee.csv', sep=';', header=None, names=['type', 'size', 'syrup'])
```

### Открытие текстовых файлов

Даже если файл имеет расширение `txt`, его можно открыть с помощью функции `read_csv`, если данные в нём структурированы. Пример использования:

```python
from pandas import read_csv

df = read_csv('data/forest.txt', sep='\t')
```

## Ограничения

* Pandas не отображает весь датасет полностью, если он слишком большой. Это может быть удобно, если компьютер начинает тормозить при работе с большими объёмами данных.

![](images/LEC_01_PART_07_P/000239s_top_7.jpg)

## Примеры работы с файлами

### Пример 1: открытие файла Excel

```python
from pandas import read_excel

df = read_excel('data/coffee_stat.xlsx')
```

![](images/LEC_01_PART_07_P/000329s_top_4.jpg)

### Пример 2: открытие файла CSV с запятой в качестве разделителя

```python
from pandas import read_csv

df = read_csv('data/students.csv')
```

![](images/LEC_01_PART_07_P/000418s_top_9.jpg)
![](images/LEC_01_PART_07_P/000438s_top_10.jpg)

### Пример 3: открытие файла CSV с нестандартным разделителем

```python
from pandas import read_csv

df = read_csv('data/coffee.csv', sep=';', header=None, names=['type', 'size', 'syrup'])
```

![](images/LEC_01_PART_07_P/000518s_top_1.jpg)
![](images/LEC_01_PART_07_P/000548s_top_7.jpg)

### Пример 4: открытие текстового файла с табуляцией в качестве разделителя

```python
from pandas import read_csv

df = read_csv('data/forest.txt', sep='\t')
```

![](images/LEC_01_PART_07_P/000627s_top_6.jpg)