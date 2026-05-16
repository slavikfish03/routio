# Routio

Routio - экспериментальный Python-проект для построения и сравнения маршрутных сетей общественного транспорта на синтетическом графе города.

Проект использует районы, остановки, дороги и OD-матрицу пассажирского спроса, а затем строит маршрутные сети несколькими способами:

- baseline-методом на основе пар районов с высоким спросом и кратчайших путей;
- генетическим алгоритмом, который оптимизирует покрытие спроса, длину маршрутов, непокрытый спрос и пересечения маршрутов;
- опциональным LLM-экспериментом для сравнения маршрутов, предложенных языковой моделью.

На выходе формируются карты, графики и CSV-метрики, которые можно использовать в презентации или отчете.

## Превью

### Синтетический граф города

![Synthetic city graph](outputs/01_city_graph.png)

### Baseline-маршруты

![Baseline routes](outputs/03_baseline_routes_colored.png)

### Маршруты генетического алгоритма

![GA routes](outputs/04_ga_routes_colored.png)

### Приближение центральной части

![GA center zoom](outputs/07_ga_center_zoom_colored.png)

### Сравнение методов

![Method comparison](outputs/11_method_comparison.png)

## Структура проекта

```text
routio/
  main.py                 # основная точка запуска эксперимента
  data_input.py           # районы, остановки, дороги, OD-матрица, конфиг
  graph_utils.py          # построение и проверка графа
  metrics.py              # метрики маршрутной сети и компоненты objective-функции
  baseline.py             # построение baseline-маршрутов
  genetic_algorithm.py    # оптимизация маршрутной сети генетическим алгоритмом
  visualization.py        # построение карт и графиков
  experiment.py           # pipeline сравнения baseline и GA
  grid_search.py          # перебор параметров генетического алгоритма
  llm_experiment.py       # опциональный эксперимент с LLM-маршрутами
  outputs/                # выбранные визуальные результаты
```

## Установка

Нужен Python 3.12 или новее.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install networkx matplotlib numpy
```

На macOS/Linux виртуальное окружение активируется так:

```bash
source .venv/bin/activate
```

## Запуск

Запустить основной эксперимент baseline vs GA:

```bash
python main.py
```

Запустить перебор параметров генетического алгоритма:

```bash
python grid_search.py
```

Запустить опциональное сравнение с LLM-маршрутами:

```bash
python llm_experiment.py
```

Все сгенерированные артефакты сохраняются в папку `outputs/`.

## Основные результаты

- `outputs/01_city_graph.png` - синтетический граф города с районами и остановками.
- `outputs/02_od_matrix_heatmap.png` - тепловая карта OD-матрицы спроса.
- `outputs/03_baseline_routes_colored.png` - маршрутная сеть baseline-метода.
- `outputs/04_ga_routes_colored.png` - маршрутная сеть, найденная генетическим алгоритмом.
- `outputs/05_baseline_edge_load.png` - загрузка ребер для baseline.
- `outputs/06_ga_edge_load.png` - загрузка ребер для GA.
- `outputs/07_ga_center_zoom_colored.png` - приближение центральной части для GA-маршрутов.
- `outputs/08_baseline_center_zoom_colored.png` - приближение центральной части для baseline.
- `outputs/09_ga_loss_history.png` - история изменения loss генетического алгоритма.
- `outputs/10_ga_components_history.png` - история компонентов objective-функции.
- `outputs/11_method_comparison.png` - сравнение метрик baseline и GA.

## Как загрузить проект на GitHub

1. Создать новый пустой репозиторий на GitHub.
2. Инициализировать Git в папке проекта:

```bash
git init
git add README.md .gitignore *.py outputs/01_city_graph.png outputs/02_od_matrix_heatmap.png outputs/03_baseline_routes_colored.png outputs/04_ga_routes_colored.png outputs/05_baseline_edge_load.png outputs/06_ga_edge_load.png outputs/07_ga_center_zoom_colored.png outputs/08_baseline_center_zoom_colored.png outputs/09_ga_loss_history.png outputs/10_ga_components_history.png outputs/11_method_comparison.png
git commit -m "Initial commit"
```

3. Подключить GitHub-репозиторий:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

4. Открыть свой профиль GitHub.
5. Нажать "Customize your pins".
6. Выбрать этот репозиторий и сохранить.

## Что лучше вставлять в README: картинки или GIF

Для этого проекта лучше оставить статичные PNG-картинки. Они быстрее загружаются и позволяют нормально рассмотреть маршруты, граф города и сравнение методов.

GIF стоит добавлять только как короткую демонстрацию последовательности, например:

```text
граф города -> baseline -> GA -> приближение центра
```

Рекомендуемый вариант:

- оставить в README 3-5 основных PNG;
- при желании добавить один короткий `outputs/demo.gif` в начало README;
- не коммитить все отдельные картинки из `outputs/routes/`, потому что они перегружают репозиторий.

Если позже будет добавлен GIF, его можно положить в `outputs/demo.gif` и вставить в README так:

```md
![Routio demo](outputs/demo.gif)
```

