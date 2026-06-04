"""
Online Retail Dataset — Data Cleaning Pipeline
================================================
Input : online_retail.csv  (raw UCI Online Retail dataset)
Output: clean_retail.csv   (cleaned, feature-enriched file)

Steps covered
-------------
1.  Load raw data
2.  Inspect & profile
3.  Remove cancelled orders
4.  Drop rows with missing CustomerID / Description
5.  Remove non-positive Quantity and UnitPrice
6.  Fix data types (InvoiceDate → datetime, CustomerID → int)
7.  Remove duplicates
8.  Strip whitespace from string columns
9.  Engineer new features (Revenue, date parts, IsUK flag)
10. Final summary report
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
# ──────────────────────────────────────────────────────────────────────────────
RAW_PATH   = r"C:\Users\SUJAL A M\Downloads\online_retail.csv\online_retail.csv"
CLEAN_PATH = r"C:\Users\SUJAL A M\Downloads\online_retail.csv\online_retaila.csv"

print("=" * 60)
print("STEP 1 — Loading raw data")
print("=" * 60)

df = pd.read_csv(RAW_PATH, encoding="unicode_escape")

print(f"  Rows loaded   : {len(df):,}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"  Memory usage  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — INSPECT & PROFILE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Initial profile")
print("=" * 60)

print("\n[Null counts]")
print(df.isnull().sum().to_string())

print("\n[Data types]")
print(df.dtypes.to_string())

print("\n[Numeric summary]")
print(df[["Quantity", "UnitPrice"]].describe().to_string())

print(f"\n  Unique InvoiceNo  : {df['InvoiceNo'].nunique():,}")
print(f"  Unique StockCode  : {df['StockCode'].nunique():,}")
print(f"  Unique Customers  : {df['CustomerID'].nunique():,}")
print(f"  Unique Countries  : {df['Country'].nunique():,}")

raw_count = len(df)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — REMOVE CANCELLED ORDERS
#           InvoiceNo starting with 'C' are credit notes / cancellations
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Remove cancelled orders (InvoiceNo starts with 'C')")
print("=" * 60)

df["InvoiceNo"] = df["InvoiceNo"].astype(str)
cancelled_mask  = df["InvoiceNo"].str.startswith("C")

print(f"  Cancelled rows removed : {cancelled_mask.sum():,}")
df = df[~cancelled_mask].copy()
print(f"  Rows remaining         : {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — DROP MISSING CustomerID AND Description
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Drop rows with missing CustomerID or Description")
print("=" * 60)

before = len(df)
df = df.dropna(subset=["CustomerID", "Description"]).copy()
dropped = before - len(df)
print(f"  Rows dropped   : {dropped:,}")
print(f"  Rows remaining : {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — REMOVE NON-POSITIVE Quantity AND UnitPrice
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Remove non-positive Quantity and UnitPrice")
print("=" * 60)

before = len(df)
neg_qty   = (df["Quantity"]  <= 0).sum()
neg_price = (df["UnitPrice"] <= 0).sum()
print(f"  Rows with Quantity  <= 0 : {neg_qty:,}")
print(f"  Rows with UnitPrice <= 0 : {neg_price:,}")

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
print(f"  Rows removed   : {before - len(df):,}")
print(f"  Rows remaining : {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — FIX DATA TYPES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Fix data types")
print("=" * 60)

# InvoiceDate → datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=False)
print(f"  InvoiceDate range : {df['InvoiceDate'].min().date()}  →  {df['InvoiceDate'].max().date()}")

# CustomerID → integer (safe now, nulls already removed)
df["CustomerID"] = df["CustomerID"].astype(int)
print(f"  CustomerID dtype  : {df['CustomerID'].dtype}")

# UnitPrice & Quantity already numeric; confirm
print(f"  Quantity  dtype   : {df['Quantity'].dtype}")
print(f"  UnitPrice dtype   : {df['UnitPrice'].dtype}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — REMOVE DUPLICATES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Remove duplicate rows")
print("=" * 60)

before = len(df)
df = df.drop_duplicates().copy()
print(f"  Duplicates removed : {before - len(df):,}")
print(f"  Rows remaining     : {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — STRIP WHITESPACE FROM STRING COLUMNS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Strip whitespace from string columns")
print("=" * 60)

str_cols = ["InvoiceNo", "StockCode", "Description", "Country"]
for col in str_cols:
    df[col] = df[col].str.strip()
    print(f"  Stripped : {col}")

# Title-case Description for readability
df["Description"] = df["Description"].str.title()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Feature engineering")
print("=" * 60)

# Line-level revenue
df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
print("  + Revenue = Quantity × UnitPrice")

# Date parts
df["Year"]        = df["InvoiceDate"].dt.year
df["Month"]       = df["InvoiceDate"].dt.month
df["MonthStr"]    = df["InvoiceDate"].dt.strftime("%Y-%m")
df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
df["Hour"]        = df["InvoiceDate"].dt.hour
df["Quarter"]     = df["InvoiceDate"].dt.quarter
print("  + Year, Month, MonthStr, YearMonth, DayOfWeek, Hour, Quarter")

# UK flag
df["IsUK"] = (df["Country"] == "United Kingdom").astype(int)
print("  + IsUK flag (1 = UK, 0 = International)")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — SAVE & FINAL REPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Save clean file & final report")
print("=" * 60)

df.to_csv(CLEAN_PATH, index=False)

removed = raw_count - len(df)
print(f"\n  Raw rows            : {raw_count:,}")
print(f"  Clean rows          : {len(df):,}")
print(f"  Rows removed total  : {removed:,}  ({removed/raw_count*100:.1f}%)")
print(f"\n  Total Revenue       : £{df['Revenue'].sum():,.2f}")
print(f"  Unique Customers    : {df['CustomerID'].nunique():,}")
print(f"  Unique Products     : {df['StockCode'].nunique():,}")
print(f"  Unique Countries    : {df['Country'].nunique():,}")
print(f"  Avg Order Value     : £{df.groupby('InvoiceNo')['Revenue'].sum().mean():,.2f}")
print(f"\n  Saved to            : {CLEAN_PATH}")
print("\n" + "=" * 60)
print("  CLEANING COMPLETE")
print("=" * 60)
