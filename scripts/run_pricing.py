
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pricing import build_price_table
from config import PRICE_TABLE_XLSX

if __name__ == "__main__":
    df, summary = build_price_table()
    print(summary.to_string(index=False))
    print("Сохранено:", PRICE_TABLE_XLSX, "| строк:", len(df))
