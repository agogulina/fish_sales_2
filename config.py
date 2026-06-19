
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.environ.get("PROMO_DATA_DIR", os.path.join(BASE_DIR, "data"))
OUT_DIR    = os.environ.get("PROMO_OUT_DIR",  os.path.join(BASE_DIR, "outputs"))

SALES_CSV       = os.path.join(DATA_DIR, "sales.csv")          # все продажи (помесячно)
PROMO_SALES_CSV = os.path.join(DATA_DIR, "promo_sales1.csv")   # промо-факт (узкий канал)
PROMO_FINAL_CSV = os.path.join(DATA_DIR, "promo_final.csv")    # недельные скидки/план
LENTA_PRICE_XLSX = os.path.join(DATA_DIR, "lenta_prices.xlsx") # прайс розничной сети (для Слоя 3)
PRICE_TABLE_XLSX = os.path.join(OUT_DIR, "prices_sku_matching.xlsx")

COL_SKU    = "Головное СКЮ Артикул"
COL_NAME   = "Головное СКЮ Наименование"
COL_CLIENT = "УПП__Группа клиентов"
COL_DATE   = "Дата"
COL_QTY    = "Кол_шт"
COL_TYPE   = "Головное СКЮ ТИП"
COL_CAT    = "Категория"
COL_GROUP  = "Группа new"
COL_WSTART = "Дата начала недели"
COL_WNUM   = "Неделя"
COL_PPLAN  = "_промо план шт"
COL_PDISC  = "_промо %скидки"
OPT_CATS   = [COL_TYPE, COL_CAT, COL_GROUP, COL_NAME]

E5_MODEL       = "intfloat/multilingual-e5-base"   # bi-encoder (плотный поиск)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"         # cross-encoder (переранжирование)
LLM_MODEL      = "Qwen/Qwen2.5-7B-Instruct"        # извлечение атрибутов 

RANDOM_STATE = 42
K            = 10            # отсечка для метрик ранжирования (P@K, R@K, nDCG@K)
CAND_POOL    = 120           # размер пула кандидатов для реранкера
EMB_DIM      = 16            # размерность PCA эмбеддингов как признаков (Слой 2)

# Слой 2/3
GRID         = [0, 5, 10, 15, 20, 25, 30]   # сетка глубины скидки, %
DML_FOLDS    = 5                            # число фолдов кросс-фиттинга
TEST_MONTHS  = 3                            # последние месяцы под тест/конформную оценку

# Слой 3 (оптимизатор)
COST_SHARE   = 0.6     # ДОЛЯ СЕБЕСТОИМОСТИ в цене входа (маржа 40%; ПРЕДПОЛОЖЕНИЕ, заменить реальной)
BUDGET_SHARE = 0.05    # запасной бюджет скидок = доля базовой выручки (если не равняем на историю)
SLOTS        = 200     # максимум SKU в промо
KMAX         = 3       # максимум субститутов из одной группы в промо
ROBUST       = False   # True -> оптимизация по нижней границе доверительного интервала эффекта
