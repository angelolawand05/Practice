import os
import pandas as pd
import numpy as np
from scipy import stats

# ---------------- CONFIG ----------------
FILE_PATH = r"C:\Users\angel\Documents\coding python\epigeneticexcel.csv"
ALPHA = 0.05
REMOVE_OUTLIERS = True
P_MIN = 1e-6
# ----------------------------------------

def fmt_p(p):
    if p is None or np.isnan(p):
        return "NA"
    return f"<{P_MIN:.6f}" if p < P_MIN else f"{p:.6f}"


# ============================================================
# 1️⃣ Load dataset
# ============================================================
def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    df = df.replace(["", " ", "NA", "N/A", "nan", "NaN"], np.nan)
    print(f"\n📂 Loaded dataset: {len(df)} rows × {len(df.columns)} columns\n")

    summary = pd.DataFrame({
        "Column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns]
    })
    print(summary.to_string(index=False))
    print()
    return df


# ============================================================
# 2️⃣ Detect columns
# ============================================================
def detect_columns(df):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if not any(x in c.lower() for x in ["id", "sample"])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 10]
    print(f"📊 Numeric columns: {num_cols}")
    print(f"📁 Categorical columns (potential groups): {cat_cols}\n")
    return num_cols, cat_cols


# ============================================================
# 3️⃣ Pick the grouping column
# ============================================================
def choose_grouping_column(cat_cols):
    if not cat_cols:
        return None
    # Prefer Diagnosis-like columns if available
    for name in cat_cols:
        if any(k in name.lower() for k in ["diagnosis", "group", "condition", "status", "class"]):
            print(f"✅ Using '{name}' as grouping column.\n")
            return name
    # Otherwise, pick the first categorical column
    print(f"✅ Using '{cat_cols[0]}' as grouping column.\n")
    return cat_cols[0]


# ============================================================
# 4️⃣ Outlier detection
# ============================================================
def detect_and_remove_outliers(df, col, group_col):
    info, cleaned_parts = {}, []
    grouped = df[[group_col, col]].dropna().groupby(group_col)

    for g, sub in grouped:
        Q1, Q3 = np.percentile(sub[col], [25, 75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (sub[col] < lower) | (sub[col] > upper)
        outs = sub.loc[mask, col]
        info[str(g)] = {
            "method": "IQR (1.5 × IQR rule)",
            "n_outliers": len(outs),
            "values": list(map(float, outs.values)),
            "bounds": (float(lower), float(upper))
        }
        cleaned_parts.append(sub.loc[~mask] if REMOVE_OUTLIERS else sub)
    cleaned = pd.concat(cleaned_parts)
    return cleaned, info


# ============================================================
# 5️⃣ Normality test
# ============================================================
def shapiro_per_group(df, col, group_col):
    results = {}
    for g, sub in df[[group_col, col]].dropna().groupby(group_col):
        x = sub[col].dropna()
        if len(x) < 3:
            results[str(g)] = {"n": len(x), "W": None, "p": None, "normal": None}
            continue
        W, p = stats.shapiro(x)
        results[str(g)] = {"n": len(x), "W": float(W), "p": float(p), "normal": p >= ALPHA}
    return results


# ============================================================
# 6️⃣ Statistical testing
# ============================================================
def run_tests(df, num_cols, group_col):
    results = []
    for feature in num_cols:
        data = df[[feature, group_col]].dropna()
        if data[group_col].nunique() < 2:
            continue

        print(f"\n=== {feature} ===")
        cleaned, outliers = detect_and_remove_outliers(data, feature, group_col)
        print("Outlier detection: IQR (1.5 × IQR rule)")
        for g, i in outliers.items():
            print(f"  {g}: n_outliers={i['n_outliers']}, values={i['values']}, "
                  f"bounds=({i['bounds'][0]:.3f}..{i['bounds'][1]:.3f})")
        print("✅ Outliers removed.")

        normality = shapiro_per_group(cleaned, feature, group_col)
        print("\nShapiro–Wilk test results:")
        for g, r in normality.items():
            if r["W"] is not None:
                print(f"  {g}: n={r['n']}, W={r['W']:.4f}, p={fmt_p(r['p'])}, "
                      f"{'Normal' if r['normal'] else 'Not normal'}")

        test_name, stat, p_value = "Descriptive only", None, None
        groups = list(cleaned[group_col].unique())
        n_groups = len(groups)

        if n_groups == 2:
            x = cleaned[cleaned[group_col] == groups[0]][feature].values
            y = cleaned[cleaned[group_col] == groups[1]][feature].values
            both_normal = all(r["normal"] for r in normality.values() if r["normal"] is not None)
            if both_normal:
                test_name = "Welch’s t-test"
                stat, p_value = stats.ttest_ind(x, y, equal_var=False)
            else:
                test_name = "Mann–Whitney U"
                stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        elif n_groups > 2:
            arrays = [cleaned[cleaned[group_col] == g][feature].values for g in groups]
            all_normal = all(r["normal"] for r in normality.values() if r["normal"] is not None)
            if all_normal:
                test_name = "One-way ANOVA"
                stat, p_value = stats.f_oneway(*arrays)
            else:
                test_name = "Kruskal–Wallis"
                stat, p_value = stats.kruskal(*arrays)

        print(f"\nChosen test: {test_name}")
        if p_value is not None:
            print(f"Statistic: {stat:.4f}, p-value: {fmt_p(p_value)} → "
                  f"{'SIGNIFICANT' if p_value < ALPHA else 'Not significant'}")

        results.append({
            "feature": feature,
            "test": test_name,
            "p_value": fmt_p(p_value) if p_value is not None else "NA",
            "significant": p_value is not None and p_value < ALPHA
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_dataset(FILE_PATH)
    num_cols, cat_cols = detect_columns(df)
    group_col = choose_grouping_column(cat_cols)
    results = run_tests(df, num_cols, group_col)

    print("\n--- Summary of Tests ---")
    print(results.to_string(index=False))
