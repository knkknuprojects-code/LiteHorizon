# Horizon

Horizon — інструмент **сканування горизонту**: він виявляє та відстежує нові сигнали,
тренди й зрушення в **будь-якій галузі** (технології, політика, ринки, наука,
геополітика тощо), а не в одній темі. Тексти аналізує модель Google **Gemma** через
Hugging Face `transformers`. Модель працює локально, а для масового інференсу та
донавчання використовується GPU-кластер.

## Зміст

- [Склад пакета](#склад-пакета)
- [Як це працює](#як-це-працює)
- [Встановлення](#встановлення)
- [Структура репозиторію](#структура-репозиторію)
- [Модулі](#модулі)
- [Команди `horizon`](#команди-horizon)
- [Модулі донавчання (`python -m`)](#модулі-донавчання-python--m)
- [Допоміжні скрипти](#допоміжні-скрипти)
- [Змінні середовища](#змінні-середовища)
- [Розробка](#розробка)

## Склад пакета

Пакет містить лише вихідний код, тести, скрипти та конфігурацію.
**Даних у ньому немає**: вихідних книг WoS/Scopus і патентного реєстру, зібраних
корпусів, збережених аналізів, кешів ембедингів, звітів запусків, навчених адаптерів.
Також не входять віртуальне оточення, git-історія, налаштування IDE та файл із ключем API.

Порожні теки `collection/`, `patents/`, `datasets/` і `reports/` залишено для
структури; у кожній є `README.md`, де описано, що туди покласти і якою командою
побудувати набір.

Швидка перевірка після розпакування:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # тести не потребують ні моделі, ні даних
horizon scan examples/sample.txt --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 256
```

## Як це працює

```
джерело ──► gather ──► corpus/ ──► scan ──► data/analyses.jsonl ──► trends / report ──► interpret
(RSS, arXiv,        (нормалізовані      (Gemma: зведення,       (ембединги +           (прочитання
 URL, WoS/Scopus,    тексти)             сигнали, класифікація,  кластеризація,         моделі, окремо
 патенти)                                сутності, ключові слова) швидкість росту)      від вимірювань)
```

1. **Збирання** (`gather`, `collection-build`, `patents-build`) — тексти з джерел
   нормалізуються й зберігаються в `corpus/` або як JSONL-шарди `records/<year>.jsonl`.
2. **Сканування** (`scan`, `gather --scan`) — модель перетворює кожен текст на
   структурований, валідований `Analysis`. Результат кешується за хешем вмісту та
   версією промпту, тому повторний запуск не завантажує модель.
3. **Тренди** (`trends`) — аналізи вбудовуються багатомовною моделлю ембедингів,
   кластеризуються, і кожен кластер оцінюється за свіжістю та швидкістю росту.
4. **Звіт** (`report`, `fields`) — звіт по корпусу (лише вимірювання) обома мовами,
   з рисунками, PDF та DOCX.
5. **Інтерпретація** (`interpret`) — прочитання звіту локальною моделлю, винесене в
   окремий документ і позначене як судження моделі, а не як результат вимірювання.
6. **Донавчання** (`export`, `approve`, `python -m horizon.training.*`) — LoRA/QLoRA на
   накопичених аналізах; вчителем для дистиляції слугує Gemini API.

## Встановлення

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # transformers, torch, accelerate, sentence-transformers + pytest, black
```

Додаткові набори залежностей (extras):

| Extra | Що встановлює | Навіщо |
|---|---|---|
| `dev` | `pytest`, `black` | тести й форматування |
| `train` | `trl<0.13`, `peft`, `datasets` | LoRA-донавчання (у тому числі на CPU) |
| `train-gpu` | `bitsandbytes` | QLoRA — лише CUDA, лише кластер |
| `infer-gpu` | `bitsandbytes` | 4/8-бітний інференс — лише CUDA |
| `ingest` | `feedparser`, `openpyxl` | RSS/arXiv-адаптери та читання `.xlsx` для `collection-build` / `patents-build` |
| `url` | `trafilatura` | якісніше виділення основного тексту для `gather url` (необов'язково) |
| `teacher` | `google-genai` | Gemini API як модель-вчитель |

```bash
pip install -e ".[dev,ingest,train]"
```

**Gemma закрита (gated):** запросіть доступ на сторінці моделі в Hugging Face і
виконайте `huggingface-cli login`. До отримання доступу користуйтеся відкритою
моделлю-замінником: `--model Qwen/Qwen2.5-0.5B-Instruct`.

**Gemini-вчитель:** ідентифікатор моделі `gemini-*` (наприклад, `--model gemini-3.5-flash`)
перемикає на хмарний бекенд. Ключ зберігається в `.horizon.env` у корені репозиторію
(файл у `.gitignore`, `chmod 600`, **ніколи не комітьте його**):

```bash
set -a; . ./.horizon.env; set +a
```

Це **єдине** місце, де Horizon звертається до зовнішнього сервісу.

**GPU-машина (Gemma 4):** Gemma 4 потребує `transformers>=5.5`, тому працює лише на
кластері. Там встановлюють залежності з `requirements-gpu.txt`, а пакет — через
`pip install -e . --no-deps`.
Обмеження в `pyproject.toml` (`torch` 2.2.2, `transformers<5`, `numpy<2`) задані під
Intel Mac, на якому ведеться розробка.

## Структура репозиторію

```
Horizon/
├── src/horizon/            # пакет (src-layout)
│   ├── llm/                # інтерфейс LLMClient і бекенди (transformers, Gemini)
│   ├── ingest/             # адаптери джерел і запис корпусу (етап 5)
│   ├── patents/            # патентний реєстр (етап 9)
│   └── training/           # експорт, донавчання, оцінювання (етап 4)
├── tests/                  # швидкі тести з FakeClient — модель не потрібна
├── scripts/                # допоміжні Python- і shell-скрипти (звіти, кластер, бенчмарки)
├── reports/                # звіти запусків (порожня; en + uk поруч, запуск по корпусу — окрема тека)
├── examples/               # приклади вхідних текстів (у тому числі іспанською)
├── datasets/               # корені наборів даних за темами (порожня; створюються командами)
├── collection/             # сюди — вихідні .xlsx-книги WoS + Scopus (не входять у пакет)
├── patents/                # сюди — вихідні .xlsx-книги патентного реєстру (не входять у пакет)
├── k8s/                    # маніфести Kubernetes (PVC, prep-job, scan-job, notebook)
├── feeds.toml, feeds-tech.toml   # конфігурації стрічок для `horizon gather feeds`
├── Dockerfile, requirements-gpu*.txt   # оточення для GPU/кластера
└── pyproject.toml          # залежності, extras, точка входу `horizon`
```

**Корінь набору даних** (`--data-dir`, або `$HORIZON_DATA_DIR`, або поточна тека) має
однакову структуру для кожної теми:

```
<data-dir>/
├── data/analyses.jsonl     # сховище аналізів (кеш + персистентність)
├── data/embeddings.jsonl   # кеш ембедингів
├── corpus/                 # зібрані тексти + manifest.jsonl
├── raw/                    # сирий вміст до нормалізації
└── records/<year>.jsonl    # побудовані шарди (collection-build / patents-build) + build-report.json
```

## Модулі

### Ядро — `src/horizon/`

| Модуль | Призначення |
|---|---|
| `config.py` | `Settings` (модель, пристрій, параметри завантаження, кластеризація) і `GenerationConfig`. Модель закріплена тут; перевизначається змінними `HORIZON_*`. |
| `analysis.py` | Модель даних структурованого аналізу: dataclass-и та перелічення. |
| `prompts.py` | Версійовані промпти (сканер, combined/staged-витягування, рубрики). |
| `scanner.py` | `scan_text()` — мінімальний вільний аналіз одного тексту (етап 1). |
| `extract.py` | `analyze_text()` / `analyze_batch()`: текст → валідований `Analysis`; розбір JSON, валідація, повторна спроба з виправленням. |
| `store.py` | `JsonlStore` — локальне сховище аналізів (кеш за хешем вмісту + версією промпту). |
| `sources.py` | Допоміжне сховище `content_hash` → вихідний текст. |
| `corpus.py` | Зіставлення аналізів із записами, з яких їх отримано. |
| `embeddings.py` | Інтерфейс `Embedder`, багатомовний бекенд і кеш ембедингів. |
| `clustering.py` | Бекенди кластеризації: `greedy` (без залежностей) і `agglomerative` (sklearn). |
| `trends.py` | Виявлення трендів: кластеризація аналізів і оцінка росту кожного кластера. |
| `fields.py` | Рівень таксономії: розвиток галузей і їхня взаємодія за класифікацією WoS/Scopus. |
| `lenses.py` | Індикатори — різні способи прочитати ті самі теми. |
| `metadata_lenses.py` | Індикатори, що читають метадані записів (цитування тощо), а не сам тренд. |
| `composition.py` | Розкладає кожну тему за значеннями поля запису (наприклад, `applicant_type`) і оцінює тренд кожної частини окремо. |
| `report.py` | Складання звіту по корпусу: спершу факти (`facts.json`), потім текст. |
| `figures.py` | Рисунки звіту: що малюється, де розміщується і якою мовою. |
| `render.py` | Рендер markdown звіту в PDF і DOCX. |
| `interpret.py` | Рівень коментаря: прочитання звіту моделлю, перевірка чисел, яких немає у фактах, і зламаних генерацій. |
| `cli.py` | Точка входу `horizon` з усіма підкомандами. |

### `src/horizon/llm/` — модель за єдиним інтерфейсом

| Модуль | Призначення |
|---|---|
| `base.py` | Абстрактний клас `LLMClient` і `Message`. Решта коду залежить лише від нього. |
| `transformers_client.py` | Локальний бекенд Gemma. `torch`/`transformers` імпортуються ліниво, модель завантажується при першому `generate()`. `system` вбудовується в хід користувача. |
| `gemini_client.py` | Бекенд Gemini API — модель-вчитель для дистиляції. |

### `src/horizon/ingest/` — збирання корпусу

| Модуль | Призначення |
|---|---|
| `base.py` | Інтерфейс `Source` і `RawItem`, який він повертає. |
| `normalize.py` | Мінімальна нормалізація тексту без залежностей. |
| `feedparse.py` | Спільні допоміжні функції для адаптерів на `feedparser`. |
| `rss.py` | RSS/Atom: один `RawItem` на запис стрічки. |
| `feeds.py` | Завантаження TOML-конфігурації стрічок для `gather feeds`. |
| `arxiv.py` | arXiv: свіжі статті за пошуковим запитом. |
| `url.py` | Звичайні URL: завантаження сторінки й виділення основного тексту. |
| `local.py` | Локальна тека з підготовленими текстовими файлами. |
| `corpus.py` | Запис зібраного в `corpus/` і повторне відтворення з нього. |
| `collection.py` | Побудова нормалізованого набору з книг WoS + Scopus у `collection/`. |
| `wos.py` | Читання побудованого набору WoS/Scopus як джерела для `gather`. |

### `src/horizon/patents/` — патентний реєстр (етап 9)

| Модуль | Призначення |
|---|---|
| `workbooks.py` | Розбір патентних книг у канонічний запис, який уже читає рівень аналізу. |
| `completeness.py` | За якими когортами і на якій часовій осі (пріоритет/подання/видача) реєстр можна вимірювати. |
| `source.py` | Читання побудованого патентного набору як джерела для `gather`. |
| `vocabulary.py` | Патентна лексика звіту — накладка на бібліографічні підписи. |

### `src/horizon/training/` — донавчання (етап 4)

| Модуль | Призначення |
|---|---|
| `export.py` | Експорт аналізів у SFT-дані (prompt, completion). |
| `stats.py` | Аудит готовності навчального набору. |
| `train.py` | LoRA SFT — один шлях коду для трьох рівнів (замінник / мала Gemma на CPU / велика Gemma на A100). |
| `merge.py` | Злиття LoRA-адаптера з базовою моделлю для автономного запуску. |
| `evaluate.py` | Оцінювання структурованого виходу моделі відносно еталонних аналізів. |
| `compare.py` | Порівняння двох сканувань одного корпусу (замінник проти вчителя). |

## Команди `horizon`

```
horizon {scan,trends,export,dataset-stats,corpus-stats,patents-build,collection-build,
         report,interpret,fields,compare,approve,gather} ...
```

Повну довідку з поточними значеннями за замовчуванням виводить `horizon <команда> --help`.

### Спільні аргументи

**Усі підкоманди** приймають:

| Аргумент | Опис |
|---|---|
| `--data-dir DATA_DIR` | Корінь набору даних із `data/`, `corpus/`, `raw/` (за замовчуванням `$HORIZON_DATA_DIR` або поточна тека). Для тем — окремі корені, напр. `datasets/tech`. |

**Команди, що завантажують модель** (`scan`, `interpret`, `gather`), додатково приймають:

| Аргумент | Опис |
|---|---|
| `--device DEVICE` | Пристрій інференсу: `cpu` \| `cuda` \| `cuda:N` \| `mps` (за замовчуванням — автовизначення). |
| `--dtype {auto,bfloat16,float16,float32}` | Тип ваг (за замовчуванням `bfloat16` на CUDA, інакше `float32`). На картах до Ampere — `float16`. |
| `--load-4bit` | Завантаження в 4 біти (nf4) через bitsandbytes — лише CUDA; потрібен `.[infer-gpu]`. |
| `--load-8bit` | Завантаження в 8 біт — лише CUDA. Проміжний варіант: 12B у bf16 займає 22,3 GiB на 24-гігабайтній карті, int8 — близько 11 GiB із запасом на батчі. |
| `--device-map DEVICE_MAP` | Розміщення accelerate, напр. `"auto"` — розподіл моделі між GPU/CPU. |
| `--max-memory MAX_MEMORY` | Бюджет пам'яті на пристрій для `--device-map`, напр. `"0=38GiB,cpu=64GiB"`. |
| `--attn {sdpa,flash_attention_2,eager}` | Ядро уваги (за замовчуванням — як у transformers; недоступне замінюється з попередженням). |
| `--batch-size BATCH_SIZE` | Діалогів на один виклик моделі (за замовчуванням 1). На GPU — головний важіль пропускної здатності, на CPU нічого не дає. |
| `--max-new-tokens N` | Обмеження довжини генерації. |
| `--temperature T` | Температура семплювання. |
| `--max-input-tokens N` | Обрізати промпт до N токенів зліва (за замовчуванням без обмеження). |

### `horizon scan` — аналіз одного тексту

Перетворює один текстовий файл на структурований аналіз (зведення, сигнали,
класифікація, сутності, ключові слова) і зберігає його в `data/analyses.jsonl`.

| Аргумент | Опис |
|---|---|
| `text_file` | Шлях до текстового файлу в UTF-8 (обов'язковий). |
| `--model MODEL_ID` | Перевизначити модель. |
| `--strategy {combined,staged}` | `combined` — один структурований виклик (за замовчуванням), `staged` — окремий виклик на кожен аспект. |
| `--date ITEM_DATE` | Дата матеріалу (`YYYY-MM-DD`) для швидкості тренду; за замовчуванням — час сканування. |
| `--json` | Вивести сирий JSON. |
| `--no-cache` | Ігнорувати кеш. |
| `--no-save` | Не зберігати результат. |
| + спільні аргументи моделі | див. вище |

```bash
horizon scan examples/sample.txt
horizon scan examples/sample.txt --json --strategy staged --date 2026-06-30
horizon scan examples/sample.txt --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 256
```

### `horizon trends` — тренди зі збережених аналізів

Вбудовує збережені аналізи, кластеризує їх і ранжує за ростом. Модель Gemma не потрібна.

| Аргумент | Опис |
|---|---|
| `--json` | Вивести сирий JSON. |
| `--cluster-method {greedy,agglomerative}` | Бекенд кластеризації (за замовчуванням із конфігу: `greedy`). |
| `--threshold THRESHOLD` | Поріг косинусної подібності (за замовчуванням із конфігу: `0.62`). |
| `--min-size MIN_SIZE` | Відкинути кластери, менші за це значення. |
| `--now NOW` | Дата відліку (`YYYY`, `YYYY-MM-DD` або ISO). За замовчуванням — найновіша `item_date` у сховищі. |
| `--window WINDOW` | Вікно швидкості: `7d` (дні) або `2y` (календарні роки; групування за когортами року публікації). За замовчуванням `7d`. |
| `--recency-scale DAYS` | Масштаб згасання свіжості: `recency = 1/(1 + days_since/scale)`. Для корпусу, датованого лише роком, — `365`. |
| `--top TOP` | Показати лише N найкращих трендів. |
| `--records PATH` | Побудовані записи для нормалізації швидкості — щоб кластер, який лише встигає за ростом корпусу, не виглядав зростаючим. Корінь набору або тека `records/`. |
| `--baseline-field NAME` | Разом із `--records`: нормалізувати відносно однієї предметної області (напр. `Physics and Astronomy`), а не всього корпусу. |

```bash
horizon trends --min-size 2 --top 20
horizon trends --data-dir datasets/wos-scopus --window 2y --recency-scale 365 --records datasets/wos-scopus
```

### `horizon gather` — збирання текстів із джерела

Збирає тексти в `corpus/` (і за бажанням одразу сканує їх).

| Аргумент | Опис |
|---|---|
| `source` | Джерело: `rss` (`--url`), `local` (`--path`), `feeds` (`--config`), `arxiv` (`--query`), `url` (`--url` / `--url-file`), `corpus` (`--path`; повторне відтворення зібраного корпусу, напр. для пересканування), `wos` (побудований набір `collection/`), `patents` (побудований набір `patents/`). |
| `--url URL` | URL стрічки для `rss` або сторінки для `url`. |
| `--path PATH` | Тека для `local` / `corpus`. |
| `--query QUERY` | Пошуковий запит arXiv. |
| `--max-results N` | Максимум результатів arXiv (за замовчуванням 50). |
| `--url-file FILE` | Файл зі списком URL (по одному в рядку, коментарі `#` дозволені). |
| `--delay SECONDS` | Пауза між завантаженнями сторінок для `url` (за замовчуванням 1.0). |
| `--config CONFIG` | TOML-файл стрічок для `feeds` (за замовчуванням `feeds.toml`). |
| `--records RECORDS` | Корінь набору з шардами `records/` для `wos`/`patents` (за замовчуванням корінь `--data-dir`). |
| `--years YEARS` | Роки для `wos`/`patents`: `2019`, `2015-2018` або `2019,2023` (за замовчуванням усі побудовані). |
| `--limit N` | Лише перші N записів (швидко, впорядковано за роком). |
| `--sample N` | N випадкових записів, відтворювано за `--seed` (репрезентативний пробний прогін). |
| `--seed SEED` | Зерно для `--sample` (за замовчуванням 0). |
| `--out OUT` | Тека корпусу для текстів + `manifest.jsonl` (за замовчуванням `<data-dir>/corpus`). |
| `--no-raw` | Не зберігати сирий вміст у `raw/`. |
| `--scan` | Одразу сканувати кожен елемент у сховище з датою публікації (потрібна модель; ідемпотентно — вже проскановане пропускається). |
| `--model MODEL_ID` | Модель для `--scan`; `gemini-*` — сканування вчителем через API. |
| `--progress-every N` | Разом із `--scan`: прогрес у stderr кожні N елементів (0 — вимкнути). |
| `--no-corpus` | Разом із `--scan`: не писати записи в `corpus/` (уникає мільйонів дрібних файлів; для `wos` вони лише дублюють шарди). |
| `--rescan` | Разом із `--scan`: перезаписати вже проскановані елементи (напр. прохід вчителя поверх міток замінника). |
| + спільні аргументи моделі | див. вище |

```bash
horizon gather rss --url https://example.org/feed.xml
horizon gather feeds --config feeds-tech.toml --data-dir datasets/tech
horizon gather arxiv --query "quantum error correction" --max-results 100
horizon gather url --url-file urls.txt --delay 2
horizon gather wos --data-dir datasets/wos-scopus --years 2019 --sample 200 --scan --no-corpus
horizon gather corpus --path datasets/tech/corpus --scan --model gemini-3.5-flash --rescan
```

### `horizon collection-build` — побудова набору WoS + Scopus

Розбирає книги `Дедублікація_WoS_Scopus_<YYYY>.xlsx` у шарди `records/<year>.jsonl`
з дедуплікацією. Потрібен extra `ingest`.

| Аргумент | Опис |
|---|---|
| `--path PATH` | Тека з книгами (за замовчуванням `collection/`). |
| `--out OUT` | Корінь набору для `records/<year>.jsonl` + `build-report.json` (за замовчуванням корінь `--data-dir`). |
| `--years YEARS` | Будувати лише ці роки: `2019`, `2015-2018` або `2019,2023`. Обмежує пам'ять, але дедуплікація охоплює лише побудовані роки. |

### `horizon patents-build` — побудова патентного набору

Розбирає книгу реєстру разом з аркушами WO та IP5 у шарди за когортами.

| Аргумент | Опис |
|---|---|
| `--path PATH` | Тека з книгою реєстру та аркушами WO/IP5 (за замовчуванням `patents/`). |
| `--out OUT` | Корінь набору (за замовчуванням `datasets/patents`). |
| `--axis {filing,grant,priority}` | Яка з трьох дат патенту стає роком когорти: `priority` (за замовчуванням — коли винахід заявлено), `filing` або `grant`. Це рішення про вимірювання: тренд за роком видачі описує роботу відомства, за роком пріоритету — коли відбувався винахід. |
| `--coverage COVERAGE` | Частка майбутніх видач когорти, яка вже має бути видана, щоб когорта вважалася придатною для звіту (за замовчуванням 0.85). Свіжі когорти записуються, але виключаються з діапазону звіту. |

### `horizon corpus-stats` — зведення зібраного корпусу

Кількість і мовний склад за джерелами, без моделі.

| Аргумент | Опис |
|---|---|
| `--corpus CORPUS` | Тека корпусу (за замовчуванням `<data-dir>/corpus`). |
| `--json` | Вивести сирий JSON. |

### `horizon fields` — розвиток галузей і їхня взаємодія

Працює за таксономією, яку вже присвоїли WoS і Scopus. Не потребує ні моделі, ні кластеризації.

| Аргумент | Опис |
|---|---|
| `--records PATH` | Побудовані записи: корінь набору або тека `records/` (обов'язковий). |
| `--key {subject_areas,wos_categories}` | Яку таксономію рахувати (за замовчуванням `subject_areas`). |
| `--years SPEC` | Когорти, напр. `2015-2025`. Неповний останній рік краще виключити: він занижує всі похідні показники. |
| `--min-total N` | Відкинути позиції з меншою кількістю записів. |
| `--top N` | Лише N перших рядків кожної таблиці. |
| `--json` | Вивести сирий JSON. |

### `horizon report` — звіт по корпусу

Будує звіт: розвиток галузей, міжгалузева взаємодія, тренди за галузями й по всьому
корпусу — обома мовами, з рисунками, PDF і DOCX, а також `<slug>-facts.json`, з якого
отримано кожну таблицю й рисунок.

| Аргумент | Опис |
|---|---|
| `--records PATH` | Побудовані записи: корінь набору або тека `records/`. |
| `--from-facts PATH` | Перерендерити попередній запуск із його `facts.json` замість нового вимірювання. Діють лише прапорці, що визначають виведення. |
| `--out OUT` | Тека для запису. |
| `--slug SLUG` | Основа імені файлів. |
| `--title TITLE` | Заголовок звіту. |
| `--date YYYY-MM-DD` | Дата в іменах файлів (за замовчуванням сьогодні). Запуск ідентифікується датою збирання даних, а не перебудови. |
| `--years SPEC` | Когорти, напр. `2015-2025`. |
| `--fit-years SPEC` | Когорти, за якими рахуються частки й нахили, якщо вужчі за `--years` (неповні хвостові когорти залишаються видимими в підрахунках, але не спотворюють похідні). |
| `--min-field-total N` | Мінімум записів для галузі. |
| `--min-pair-total N` | Мінімум записів для пари галузей. |
| `--top-fields N` | Кількість галузей. |
| `--top-pairs N` | Кількість пар галузей. |
| `--top-trends N` | Тем на розділ у `facts.json` (за замовчуванням 50) — більше, ніж друкують таблиці, бо з JSON будуються рисунки. |
| `--table-rows N` | Рядків у таблиці тем (за замовчуванням 10). |
| `--indicator-rows N` | Рядків у таблиці індикатора (за замовчуванням 20). |
| `--large-topic-size N` | Публікацій, потрібних темі для ранжування за спадом (за замовчуванням 200). |
| `--min-size N` | Відкинути менші кластери. |
| `--lens-min-size N` | Членів кластера, потрібних для ранжування індикатором (за замовчуванням 20). |
| `--threshold T` | Поріг косинусної подібності. |
| `--cluster-method {greedy,agglomerative}` | Бекенд кластеризації. |
| `--window SPEC` | Вікно швидкості, напр. `2y`. |
| `--anchor DATE` | Дата відліку для оцінки. |
| `--block-by {field,none}` | Кластеризувати окремо в кожній предметній області (за замовчуванням) чи по всій сукупності разом (`none` — для невеликих корпусів або записів із кількома галузями). |
| `--compose-by FIELD` | Розкласти кожну тему за значеннями поля запису (напр. `applicant_type`) — додає розділ 6. |
| `--vocabulary {bibliographic,patents}` | Лексика заголовків і колонок: «публікації» / WoS і Scopus чи «патенти» / реєстр. Арифметика однакова. |
| `--page-break-areas` | Кожна предметна область розділу 3 — з нової сторінки. |
| `--no-figures` | Без рисунків і посилань на них. |
| `--no-trends` | Лише рівні 1 і 2 — без ембедингів і кластеризації (секунди замість хвилин). |

```bash
horizon report --records datasets/wos-scopus --out reports/wos-scopus-2026-08-28 \
  --slug wos-scopus --years 2015-2025 --window 2y
horizon report --from-facts reports/<run>/<slug>-facts.json --out reports/<run> --page-break-areas
horizon report --records datasets/patents --vocabulary patents --block-by none --compose-by applicant_type
```

### `horizon interpret` — прочитання звіту моделлю

Коментар науковця-наукометриста, написаний локальною моделлю. Одна модель за запуск
пише власний файл `<slug>-reading-<model>.json`; кілька прочитань потім зводить
`scripts/build_interpretation.py`.

| Аргумент | Опис |
|---|---|
| `--facts PATH` | `facts.json` звіту (обов'язковий). |
| `--model MODEL_ID` | Перевизначити модель. |
| `--out OUT` | Тека для запису. |
| `--slug SLUG` | Основа імені файлів. |
| `--lang {en,uk}` | Мова тексту. |
| `--domain {bibliographic,patents}` | Як промпт описує корпус: анотації публікацій WoS/Scopus чи анотації патентів реєстру. Помилку тут не видно у виході. |
| `--min-size N` | Публікацій, потрібних темі для власного прочитання (за замовчуванням 200). |
| `--limit N` | Зупинитися після N тем. |
| `--pass {all,corpus,topics,areas}` | Які прочитання генерувати (за замовчуванням `all`). Кожне зберігається окремо й відновлюється незалежно. |
| + спільні аргументи моделі | див. вище |

```bash
horizon interpret --facts reports/<run>/<slug>-facts.json --out reports/<run> \
  --slug <slug> --lang uk --model google/gemma-3-4b-it
```

### `horizon compare` — порівняння двох сканувань

Порівнює два сканування одного корпусу (напр. замінник проти вчителя).

| Аргумент | Опис |
|---|---|
| `--baseline SUBSTR` | Підрядок `model_id` базової сторони (за замовчуванням — наявна не-Gemini модель). |
| `--teacher SUBSTR` | Підрядок `model_id` вчителя (за замовчуванням — наявна модель Gemini). |
| `--json` | Вивести сирий JSON. |

### `horizon approve` — схвалення аналізів для навчання

| Аргумент | Опис |
|---|---|
| `hashes ...` | Значення `content_hash` для схвалення (дозволено префікс). |
| `--all` | Схвалити всі придатні для навчання записи. |
| `--set-source SRC` | Також встановити походження (напр. `human`, `teacher`). |

### `horizon export` — експорт SFT-даних

Експортує аналізи як пари (prompt, completion): `train.jsonl`, `eval.jsonl`, `manifest.json`.

| Аргумент | Опис |
|---|---|
| `--out OUT` | Тека виводу. |
| `--smoke N` | Лише перші N прикладів (крихітний набір для перевірки циклу). |
| `--eval-fraction F` | Частка для eval-розбиття за хешем вмісту (за замовчуванням 0.2). |
| `--no-filter` | Залишити вироджені/заглушкові аналізи (за замовчуванням відкидаються). |
| `--approved-only` | Лише записи, схвалені через `horizon approve`. |
| `--source SRC` | Лише записи з цим походженням (напр. `teacher`, `human`). |

### `horizon dataset-stats` — готовність навчального набору

| Аргумент | Опис |
|---|---|
| `--target N` | Цільова кількість готових прикладів, відносно якої рахується нестача (за замовчуванням 500). |
| `--json` | Вивести сирий JSON. |

## Модулі донавчання (`python -m`)

Потребують extra `train` (QLoRA — ще й `train-gpu`, лише на кластері).

### `python -m horizon.training.train` — LoRA SFT

| Аргумент | Опис |
|---|---|
| `--base BASE_MODEL` | Ідентифікатор базової моделі (обов'язковий). |
| `--data DATA_DIR` | Тека експортованого набору (`train.jsonl` / `eval.jsonl`). |
| `--out OUTPUT_DIR` | Тека для адаптера (обов'язковий). |
| `--lora-r`, `--lora-alpha` | Ранг і альфа LoRA. |
| `--max-seq-length` | Максимальна довжина послідовності. |
| `--lr` | Швидкість навчання. |
| `--epochs` / `--max-steps` | Кількість епох або кроків. |
| `--batch-size` | Батч на пристрій. |
| `--grad-accum` | Кроки накопичення градієнта. |
| `--seed` | Зерно. |
| `--device` | Примусовий пристрій (за замовчуванням — автовизначення). |
| `--qlora` / `--no-qlora` | Увімкнути/вимкнути QLoRA (4 біти, лише CUDA). |

### `python -m horizon.training.merge` — злиття адаптера

| Аргумент | Опис |
|---|---|
| `--base BASE_MODEL` | Базова модель (обов'язковий). |
| `--adapter ADAPTER_PATH` | Тека адаптера (обов'язковий). |
| `--out OUT_DIR` | Тека злитої моделі (обов'язковий). |
| `--device` | Пристрій завантаження (за замовчуванням `cpu`). |

### `python -m horizon.training.evaluate` — база проти донавченої

| Аргумент | Опис |
|---|---|
| `--base BASE_MODEL` | Базова модель (обов'язковий). |
| `--adapter ADAPTER_PATH` | LoRA-адаптер для донавченої сторони (поверх `--base`). |
| `--tuned-model ID` | Або автономна донавчена/злита модель замість `--adapter`. |
| `--data DATA_DIR` | Тека з `eval.jsonl`. |
| `--device` | Примусовий пристрій. |
| `--max-new-tokens`, `--temperature` | Параметри декодування (температура 0 — жадібне, детерміноване). |
| `--json` | Вивести сирий JSON. |

```bash
horizon export --out out/sft --approved-only
python -m horizon.training.train --base Qwen/Qwen2.5-0.5B-Instruct --data out/sft --out out/adapter --max-steps 20
python -m horizon.training.evaluate --base Qwen/Qwen2.5-0.5B-Instruct --adapter out/adapter --data out/sft
python -m horizon.training.merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter out/adapter --out out/merged
```

Донавчену модель можна запускати через `HORIZON_MODEL_ID=<base>` + `HORIZON_ADAPTER=<тека адаптера>`.

## Допоміжні скрипти

Запускаються з кореня репозиторію (`python scripts/<name>.py --help`).

### Звіти та коментарі

| Скрипт | Призначення та аргументи |
|---|---|
| `build_interpretation.py facts readings...` | Зводить прочитання кількох моделей у два документи коментаря (`-interpretation-ua.md`, `-annotated-ua.md`). `--report` (markdown звіту), `--out`, `--min-size`, `--render` (також PDF і DOCX). |
| `build_figures.py facts` | Перемальовує рисунки звіту з `facts.json`. `--out`, `--slug`. |
| `build_deck.py facts` | Будує презентацію `.pptx` з `facts.json`. `--lang {en,uk}`, `--out`, `--title`, `--reference-doc` (шаблон `.pptx`), `--keep-md`. |
| `build_workbook.py facts [readings...]` | Експортує теми, їхні показники та прочитання моделей у таблицю. `--out`, `--min-size`. |
| `compare_corpora.py facts_a facts_b` | Порівнює два корпуси за спільними роками в частках, без абсолютних одиниць. Обов'язкові: `--out`, `--slug`, `--date` і для кожної сторони (`-a`/`-b`) `--name-*`, `--name-*-uk`, `--unit-*`, `--unit-*-uk` (три форми через `\|`: `публікація\|публікації\|публікацій`), `--taxonomy-*`, `--taxonomy-*-uk`; необов'язкові: `--coverage-*` (`build-report.json`), `--top`, `--min-labels`, `--no-figures`, `--reading PATH` (повторюваний), `--render`. |
| `compare_interpret.py facts` | Прочитання порівняння двох корпусів однією локальною моделлю. `--out`, `--slug`, `--model` (обов'язкові), `--lang {uk,en}`, `--pass {all,comparison,taxonomies,matches}`, `--match-floor`, `--max-new-tokens`, `--device`. |

### Вимірювання та вибір параметрів

| Скрипт | Призначення та аргументи |
|---|---|
| `bench_models.py` | Бенчмарк моделей витягування: секунди на валідований аналіз, пропускна здатність, пам'ять. `--device`, `--models`, `--preset {fallback,gemma4,fit}`, `--quant {none,4bit}...`, `--batch N...`, `--max-new-tokens`, `--records`, `--items`, `--seed`, `--out`, `--probe` (VRAM і які варіанти Gemma 4 вміщаються), `--legacy-short`, `--legacy-long`. |
| `threshold_sweep.py root` | Вибір порогу кластеризації вимірюванням. `--thresholds T...`, `--sample` (за замовчуванням 20000), `--min-size` (за замовчуванням 20), `--models SUBSTR SUBSTR`, `--seed`, `--json`. |
| `cluster_agreement.py root` | Чи дають аналізи двох моделей ті самі кластери трендів. `--models SUBSTR SUBSTR` (за замовчуванням `E4B 12B`), `--threshold`, `--cluster-method`. |
| `merge_shards.py out roots...` | Зливає корені шардів в одне сховище, відмовляючи, якщо це не один набір. `--dry-run`, `--allow-mixed-model`, `--allow-overlap N`, `--expect N`. |

### GPU-машина та кластер

| Скрипт | Призначення |
|---|---|
| `survey_box.sh [-o FILE]` | Огляд машини перед встановленням. |
| `quick_survey.sh [-o FILE]` | Мінімальний огляд без залежностей — для вставки в SSH-сесію. |
| `bootstrap_box.sh [--dry-run]` | Створює `.venv` і встановлює оточення інференсу на CUDA-машині. |
| `push_to_box.sh USER@HOST [--dry-run]` | Надсилає робоче дерево на GPU-машину (`BOX_DEST` — інша тека призначення). |
| `run_scan.sh YEAR [MODEL] [BATCH] [none\|4bit]` | Сканує один шард року в окремий корінь набору (за замовчуванням `google/gemma-4-E4B-it`, батч 8, без квантизації). Повторний запуск продовжує роботу. Змінні: `HORIZON_SOURCE` (`wos`/`patents`), `HORIZON_RECORDS`, `HORIZON_SCRATCH`. |
| `sweep.sh [N]` | Перевіряє карту й проганяє варіанти Gemma 4, що на неї вміщаються. |
| `compare_models.sh [--dry-run]` | Сканує той самий зріз кількома моделями й попарно порівнює результати. |

## Змінні середовища

| Змінна | Значення за замовчуванням | Опис |
|---|---|---|
| `HORIZON_DATA_DIR` | поточна тека | Корінь набору даних (те саме, що `--data-dir`). |
| `HORIZON_MODEL_ID` | `google/gemma-2-2b-it` | Модель для інференсу. |
| `HORIZON_ADAPTER` | — | Тека LoRA-адаптера поверх `HORIZON_MODEL_ID`. |
| `HORIZON_DEVICE` | автовизначення | `cuda` / `mps` / `cpu`. |
| `HORIZON_STRATEGY` | `combined` | `combined` або `staged`. |
| `HORIZON_DTYPE` | — | Тип ваг (`--dtype`). |
| `HORIZON_LOAD_4BIT`, `HORIZON_LOAD_8BIT` | вимкнено | `1/true/yes/on` вмикає квантизацію. |
| `HORIZON_DEVICE_MAP`, `HORIZON_MAX_MEMORY` | — | Розміщення accelerate. |
| `HORIZON_ATTN` | — | Ядро уваги. |
| `HORIZON_BATCH_SIZE` | `1` | Розмір батчу генерації. |
| `HORIZON_MAX_INPUT_TOKENS` | без обмеження | Обрізання промпту. |
| `HORIZON_EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Модель ембедингів. |
| `HORIZON_CLUSTER_METHOD` | `greedy` | `greedy` або `agglomerative`. |
| `HORIZON_CLUSTER_THRESHOLD` | `0.62` | Поріг косинусної подібності. |
| `HORIZON_VELOCITY_WINDOW` | `7d` | Вікно швидкості (`HORIZON_VELOCITY_WINDOW_DAYS` — застарілий псевдонім у днях). |
| `HORIZON_RECENCY_SCALE_DAYS` | `1` | Масштаб згасання свіжості. |
| `HORIZON_ATTENTION_WEIGHT` | `0.5` | Вага чинника уваги (цитування тощо) в оцінці тренду: множник у межах `[1-w, 1+w]`; має бути `< 1.0`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Ключ для моделей `gemini-*` (зберігається в `.horizon.env`). |

## Розробка

```bash
pytest                                                    # тести без ML-стеку (модель підмінена FakeClient)
pytest tests/test_llm.py::test_to_chat_folds_system_into_user   # один тест
black .                                                   # форматування
```

Правила:

- Новий код працює через `LLMClient`; `torch` і `transformers` не імпортуються на рівні
  модуля — лише ліниво всередині методів.
- Звіти лежать у `reports/` (англійська та українська версії поруч, суфікс `-ua`);
  у git комітяться лише `.md`, а PDF, DOCX, PPTX, XLSX, `-facts.json` і `-reading-*.json`
  залишаються локально.
