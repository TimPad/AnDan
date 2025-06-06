# Проверка гипотезы о равенстве среднего числа с помощью одновыборочного критерия Стьюдента

## Постановка задачи

Рассмотрим данные о оценках студентов из некоторой школы: 8, 6, 4, 6, 8, 9, 7, 8. Это баллы за экзамен по английскому языку. Необходимо проверить гипотезу о том, что средний балл для всех учеников данной школы по английскому языку равен 6,5.

*Нулевая гипотеза* ($H_0$): средний балл равен 6,5.  
*Альтернативная гипотеза* ($H_1$): средний балл не равен 6,5.

## Расчёт значения тестовой статистики

Для проверки гипотезы необходимо рассчитать значение тестовой статистики по формуле:

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

где:  
- $\bar{x}$ — среднее значение по выборке;  
- $\mu_0$ — число, с которым сравниваем среднее в гипотезе;  
- $s$ — стандартное отклонение по выборке;  
- $n$ — объём выборки.

### Расчёт среднего значения

Среднее значение по выборке:

$$
\bar{x} = \frac{8 + 6 + 4 + 6 + 8 + 9 + 7 + 8}{8} = 7
$$

### Расчёт стандартного отклонения

Стандартное отклонение:

1. Найдём отклонения значений от среднего:  
   - $8 - 7 = 1$  
   - $6 - 7 = -1$  
   - $4 - 7 = -3$  
   - $6 - 7 = -1$  
   - $8 - 7 = 1$  
   - $9 - 7 = 2$  
   - $7 - 7 = 0$  
   - $8 - 7 = 1$  

2. Возведём отклонения в квадрат:  
   - $1^2 = 1$  
   - $(-1)^2 = 1$  
   - $(-3)^2 = 9$  
   - $(-1)^2 = 1$  
   - $1^2 = 1$  
   - $2^2 = 4$  
   - $0^2 = 0$  
   - $1^2 = 1$  

3. Сумма квадратов отклонений: $1 + 1 + 9 + 1 + 1 + 4 + 0 + 1 = 18$  

4. Стандартное отклонение: $s = \sqrt{\frac{18}{n-1}} \approx 1,64$ (округлено до тысячных)  

### Расчёт наблюдаемого значения тестовой статистики

Наблюдаемое значение тестовой статистики:

$$
t = \frac{7 - 6,5}{1,64 / \sqrt{8}} \approx 0,88
```

```mermaid
flowchart LR
    A[Расчёт среднего значения] --> B[Расчёт отклонений]
    B --> C[Возведение отклонений в квадрат]
    C --> D[Сумма квадратов отклонений]
    D --> E[Расчёт стандартного отклонения]
    E --> F[Расчёт тестовой статистики]
```

*Диаграмма выше иллюстрирует последовательность шагов для расчёта тестовой статистики.*

## Проверка гипотезы

Критические значения для уровня значимости 5% и числа степеней свободы 7 равны $\pm 2,365$.

Так как наблюдаемое значение тестовой статистики (0,88) не попадает в критическую область, то *нулевую гипотезу не отвергаем*.

Вывод: средний балл оценки равен 6,5.