import pandas as pd, zipfile, re, html, numpy as np

zip_path = r"C:\Users\HP\Downloads\archive.zip"
out_csv = r"C:\Users\HP\Downloads\fake_job_postings_clean_10000.csv"

# ---------- 1) Load from ZIP ----------
with zipfile.ZipFile(zip_path) as z:
    csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f)

n0 = len(df)

# ---------- 2) Cleaning helpers ----------
TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    return s

def norm_ws(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]

# Ensure expected columns exist
for c in ["job_id", "location", "department", "salary_range", "employment_type",
          "required_experience", "required_education", "industry", "function",
          "telecommuting", "has_company_logo", "has_questions", "fraudulent"]:
    if c not in df.columns:
        df[c] = np.nan

# ---------- 3) De-dup ----------
df = df.drop_duplicates(subset=["job_id"], keep="first")
df = df.drop_duplicates(keep="first")
n1 = len(df)

# ---------- 4) Clean text fields ----------
for c in TEXT_COLS:
    df[c] = df[c].fillna("").astype(str).map(strip_html).map(norm_ws)

# Build unified document text
df["TEXT"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
).map(norm_ws)

# Remove rows with empty/very short text
df = df[df["TEXT"].str.len() >= 50].copy()
n2 = len(df)

# ---------- 5) Keep only relevant columns ----------
keep_cols = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent", "TEXT"
]
df = df[keep_cols]

# ---------- 6) Sample 10,000 rows ----------
n_target = 10000
df_10k = df.sample(n=n_target, random_state=42) if len(df) >= n_target else df.copy()

# ---------- 7) Save ----------
df_10k.to_csv(out_csv, index=False)

print({
    "rows_original": n0,
    "rows_after_dedup": n1,
    "rows_after_text_filter(len>=50)": n2,
    "rows_saved": len(df_10k),
    "output": out_csv
})
