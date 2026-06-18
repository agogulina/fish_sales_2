"""
Единая точка входа: запускает весь конвейер или отдельный слой.

Примеры:
    python -m promo_copilot all            # все слои по порядку
    python -m promo_copilot pricing        # только ценовая таблица
    python -m promo_copilot layer2         # только Слой 2
    python __main__.py all                 # то же из корня проекта

Шаги: pricing -> layer1 -> cannibalization -> layer2 -> layer3
"""
import sys, os, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# имя шага -> файл скрипта
STAGES = {
    "pricing":         "run_pricing.py",
    "layer1":          "run_layer1.py",
    "cannibalization": "run_cannibalization.py",
    "layer2":          "run_layer2.py",
    "layer3":          "run_layer3.py",
}
ORDER = ["pricing", "layer1", "cannibalization", "layer2", "layer3"]


def run_stage(name: str):
    script = os.path.join(ROOT, "scripts", STAGES[name])
    print("\n" + "=" * 70)
    print(f">>> ШАГ: {name}  ({STAGES[name]})")
    print("=" * 70)
    try:
        runpy.run_path(script, run_name="__main__")
    except FileNotFoundError as e:
        print(f"[пропуск {name}] не найдены входные данные: {e}")
    except Exception as e:                              # noqa
        print(f"[ошибка на шаге {name}] {type(e).__name__}: {e}")


def main(argv):
    arg = (argv[1] if len(argv) > 1 else "all").lower()
    if arg in ("all", "pipeline"):
        for s in ORDER:
            run_stage(s)
    elif arg in STAGES:
        run_stage(arg)
    else:
        print("Использование: python -m promo_copilot [all|" + "|".join(ORDER) + "]")
        print("По умолчанию (без аргумента) выполняется весь конвейер.")
        sys.exit(1)
    print("\nГотово.")


if __name__ == "__main__":
    main(sys.argv)
