
from typing import Tuple
import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID"
]

def load_and_validate_excel(file_like) -> pd.DataFrame:
    df = pd.read_excel(file_like, engine="openpyxl")
    # Validar columnas mínimas
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    # Limpieza básica
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # Normalizar fecha a solo fecha (sin hora)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.date
    return df

def build_product_demand(df: pd.DataFrame) -> pd.DataFrame:
    # Serie por producto y fecha (cantidades y precio medio)
    product_demand = df.groupby(["InvoiceDate", "StockCode"]).agg(
        Quantity=("Quantity", "sum"),
        UnitPrice=("UnitPrice", "mean")
    ).reset_index()
    return product_demand

def build_customer_product(df: pd.DataFrame) -> pd.DataFrame:
    # Serie por cliente, producto y fecha
    customer_product = df.groupby(["CustomerID", "InvoiceDate", "StockCode"]).agg(
        Quantity=("Quantity", "sum"),
        UnitPrice=("UnitPrice", "mean")
    ).reset_index()
    return customer_product
