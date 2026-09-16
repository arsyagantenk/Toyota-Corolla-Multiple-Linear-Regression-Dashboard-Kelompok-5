from pathlib import Path
import numpy as np
import pandas as pd

# Canonical schema follows the NEW Excel dataset.
COLUMN_ALIASES = {
    # older dataset labels -> new canonical labels (kept only for compatibility)
    "Age_08_04": "Age",
    "KM": "Kilometers",
    "Fuel_Type": "Fuel Type",
    "HP": "Horse Power",
    "Met_Color": "Metallic",
    "Quarterly_Tax": "Quart Tax",
    "Weight (kg)": "Weight",
    "Price": "Price",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return renamed

def generate_default_dataset(n=1436, seed=42):
    rng=np.random.default_rng(seed)
    age=rng.integers(1,80,n)
    km=(age*1200+rng.normal(0,8000,n)).clip(500,200000)
    fuel=rng.choice(["Petrol","Diesel","CNG"],n,p=[.7,.25,.05])
    hp=rng.choice([86,90,97,110,116],n)
    metallic=rng.integers(0,2,n)
    automatic=rng.integers(0,2,n)
    cc=rng.choice([1300,1400,1600,1800],n)
    doors=rng.choice([2,3,4,5],n)
    tax=rng.choice([19,55,69,85,100,135,210],n)
    weight=rng.integers(1000,1350,n)
    price=(20000-95*age-.03*km+35*hp+8*weight+2*cc+500*automatic+rng.normal(0,800,n)).clip(4000,35000)
    return pd.DataFrame({"Id":np.arange(1,n+1),"Model":["TOYOTA Corolla"]*n,"Price":price,"Age":age,"Kilometers":km,"Fuel Type":fuel,"Horse Power":hp,"Metallic":metallic,"Automatic":automatic,"CC":cc,"Doors":doors,"Quart Tax":tax,"Weight":weight})

def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        name=getattr(uploaded_file,"name","").lower()
        if name.endswith(".csv"):
            return normalize_columns(pd.read_csv(uploaded_file))
        if name.endswith((".xlsx",".xls")):
            return normalize_columns(pd.read_excel(uploaded_file,sheet_name=0))
    bundled=Path(__file__).resolve().parent/"Toyota Corolla.xlsx"
    if bundled.exists():
        return normalize_columns(pd.read_excel(bundled,sheet_name=0))
    return generate_default_dataset()
