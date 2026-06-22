from pathlib import Path

import pandas as pd
from gurobipy import GRB, Model, quicksum

# =========================================================
# CONSTANTS
# =========================================================
# Folder where the CSV files are stored
BASE_DIR = Path("dados")

# CSV files used by the model
CSV_FILES = {
    "bebidas": "bebidas.csv",
    "carnes_vermelhas": "carnes_vermelhas.csv",
    "frangos": "frangos.csv",
    "linguicas": "linguicas.csv",
    "salada_maionese": "salada_maionese.csv",
    "temperos": "temperos.csv",
    "vinagrete": "vinagrete.csv",
}

# Number of guests
P = int(input("Número de pessoas: "))

# Consumption constants
SOLIDS_PER_PERSON_G = 400   # Grams
DRINKS_PER_PERSON_L = 2     # Liters 

# Meat -> charcoal / skewers relations
CARVAO_KG_PER_MEAT_KG = 1 / 1.5   # 1 kg charcoal for each 1.5 kg meat
ESPETOS_PER_MEAT_KG = 10          # 1 skewer per 100 g = 10 skewers per kg

# Fixed items kept inside the script
FIXED_ITEMS = pd.DataFrame([
    {"categoria": "descartaveis", "tipo": "Prato Descartável",          "marca": "mercado", "preco": 2.99,  "peso": 100, "unidade": "un"},
    {"categoria": "descartaveis", "tipo": "Copo Descartável",           "marca": "mercado", "preco": 5.49,  "peso": 100, "unidade": "un"},
    {"categoria": "descartaveis", "tipo": "Garfo/Colher Descartável",   "marca": "mercado", "preco": 2.15,  "peso": 10,  "unidade": "un"},
    {"categoria": "carvao",       "tipo": "Carvão",                     "marca": "mercado", "preco": 17.60, "peso": 2.5,  "unidade": "kg"},
    {"categoria": "acendedor",    "tipo": "Acendedor",                  "marca": "mercado", "preco": 10.40, "peso": 1,    "unidade": "un"},
    {"categoria": "espetos",      "tipo": "Espeto",                     "marca": "mercado", "preco": 4.70,  "peso": 100,  "unidade": "un"},
])

BIG_M = 1000  # Simple big-M for the salad "different item" constraint

# =========================================================
# HELPERS
# =========================================================
def load_csv(category: str, filename: str) -> pd.DataFrame:
    """Read one CSV into a cleaned DataFrame."""
    df = pd.read_csv(BASE_DIR / filename)

    # Normalize column names for easier handling
    df.columns = [c.strip().lower() for c in df.columns]

    # Add category and clean text columns
    df["categoria"] = category
    df["tipo"] = df["tipo"].astype(str).str.strip()

    # Empty brand becomes 'mercado'
    df["marca"] = (
        df["marca"]
        .fillna("mercado")
        .replace("", "mercado")
        .astype(str)
        .str.strip()
    )

    # Numeric columns
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")

    # Remove broken rows
    df = df.dropna(subset=["preco", "peso"]).copy()
    df["unidade"] = df["unidade"].astype(str).str.strip()

    return df


def ask_brand_preference(category: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ask the user for a preferred brand.

    For most categories:
        - keep only rows from that brand
        - if no brand is chosen, keep everything

    For salada_maionese:
        - the brand preference is applied only to Maionese rows
        - all non-maionese salad items remain available
    """
    brands = sorted(df["marca"].dropna().unique().tolist())

    print(f"\nCategoria: {category}")
    print("Marcas disponíveis:", ", ".join(brands))

    pref = input("Marca preferida (Enter = nenhuma): ").strip()
    if not pref:
        return df

    # Special case: brand preference only for Maionese inside salad
    if category == "salada_maionese":
        mayo_mask = df["tipo"].str.contains("maionese", case=False, na=False)
        mayo_filtered = df[mayo_mask & (df["marca"].str.lower() == pref.lower())]

        # If the user typed a brand that does not exist for mayo, keep all salad options
        if mayo_filtered.empty:
            print("Marca não encontrada para Maionese. Usando todas as opções da salada.")
            return df

        # Keep all non-mayo items + only the preferred mayo rows
        return pd.concat([df[~mayo_mask], mayo_filtered], ignore_index=True)

    # Normal behavior for all the other categories
    filtered = df[df["marca"].str.lower() == pref.lower()]
    if filtered.empty:
        print("Marca não encontrada. Usando todas as opções.")
        return df

    return filtered.copy()


def quantity_in_base(row: pd.Series) -> float:
    """
    Convert the row quantity to the base unit used in the constraints:
    - kg -> grams
    - g  -> grams
    - L  -> liters
    - un -> units
    """
    unit = str(row["unidade"]).lower()
    value = float(row["peso"])

    if unit == "kg":
        return value * 1000.0
    if unit == "g":
        return value
    if unit == "l":
        return value
    if unit == "un":
        return value

    return value


def format_total(row: pd.Series, x_value: float) -> str:
    """Pretty-print the total amount bought for one chosen item."""
    unit = str(row["unidade"]).lower()
    total = quantity_in_base(row) * x_value

    if unit == "kg":
        return f"{total / 1000:.2f} kg"
    if unit == "g":
        return f"{total:.0f} g"
    if unit == "l":
        return f"{total:.2f} litros"
    if unit == "un":
        return f"{total:.0f} unidades"

    return f"{total:.2f} {row['unidade']}"


def idxs(df: pd.DataFrame, category: str, contains: str | None = None) -> list[int]:
    """Get row indexes for a category, optionally filtering by a text fragment in 'tipo'."""
    mask = df["categoria"].eq(category)
    if contains is not None:
        mask &= df["tipo"].str.contains(contains, case=False, na=False)
    return df.index[mask].tolist()

def add_min_distinct_items(model, items_df, xvars, category: str, minimum: int, name: str):
    """Require at least `minimum` different rows selected in one category."""
    idx_list = items_df.index[items_df["categoria"].eq(category)].tolist()

    y = model.addVars(idx_list, vtype=GRB.BINARY, name=f"sel_{name}")

    for i in idx_list:
        model.addConstr(xvars[i] <= BIG_M * y[i], name=f"ub_{name}_{i}")
        model.addConstr(xvars[i] >= y[i], name=f"lb_{name}_{i}")

    model.addConstr(quicksum(y[i] for i in idx_list) >= minimum, name=f"min_{name}")

# =========================================================
# LOAD DATA
# =========================================================
# Pandas reads each CSV as a table (DataFrame), then we merge everything later.
frames = []
for category, filename in CSV_FILES.items():
    df = load_csv(category, filename)
    df = ask_brand_preference(category, df)
    frames.append(df)

# Add the fixed items defined directly in the script
all_items = pd.concat(frames + [FIXED_ITEMS], ignore_index=True).reset_index(drop=True)

# =========================================================
# MODEL
# =========================================================
# Gurobi creates and solves the optimization model.
m = Model("churrasco")

# One integer decision variable for each row in the final table
x = m.addVars(all_items.index.tolist(), vtype=GRB.INTEGER, lb=0, name="x")

# Objective: minimize total cost
m.setObjective(
    quicksum(all_items.loc[i, "preco"] * x[i] for i in all_items.index),
    GRB.MINIMIZE
)

# =========================================================
# CATEGORY GROUPS
# =========================================================
food_categories = {
    "carnes_vermelhas",
    "frangos",
    "linguicas",
    #"salada_maionese",
   #"temperos",
    #"vinagrete",
}

meat_categories = {
    "carnes_vermelhas",
    "frangos",
    "linguicas",
}

# =========================================================
# DEMAND CONSTRAINTS
# =========================================================
# Food demand in grams
m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] in food_categories
    ) >= SOLIDS_PER_PERSON_G * P,
    name="food_demand"
)

# Minimum composition for some food categories
m.addConstr(
    quicksum(
        x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "linguicas"
    ) >= 1,
    name="min_linguicas"
)

m.addConstr(
    quicksum(
        x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "frangos"
    ) >= 1,
    name="min_frangos"
)

m.addConstr(
    quicksum(
        x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "carnes_vermelhas"
    ) >= 1,
    name="min_carnes_vermelhas"
)

add_min_distinct_items(m, all_items, x, "vinagrete", 3, "vinagrete")
add_min_distinct_items(m, all_items, x, "temperos", 2, "temperos")

# Drinks demand in liters
m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "bebidas"
    ) >= DRINKS_PER_PERSON_L * P,
    name="drink_demand"
)

# Disposables: at least one package per person of each required item type
m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "descartaveis"
        and "prato" in all_items.loc[i, "tipo"].lower()
    ) >= P,
    name="plates_demand"
)

m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "descartaveis"
        and "copo" in all_items.loc[i, "tipo"].lower()
    ) >= P,
    name="cups_demand"
)

m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "descartaveis"
        and ("garfo" in all_items.loc[i, "tipo"].lower() or "colher" in all_items.loc[i, "tipo"].lower())
    ) >= P,
    name="cutlery_demand"
)

# =========================================================
# SALAD RESTRICTIONS
# =========================================================
# Salad must contain at least one Maionese
salad_idx = idxs(all_items, "salada_maionese")
mayo_idx = idxs(all_items, "salada_maionese", "maionese")
non_mayo_idx = [i for i in salad_idx if i not in mayo_idx]

m.addConstr(
    quicksum(x[i] for i in mayo_idx) >= 1,
    name="salad_mayo"
)

# Salad must contain at least two different non-mayo items
# We use binary variables to mark whether a type is selected.
salad_types = sorted(all_items.loc[non_mayo_idx, "tipo"].unique().tolist())
y = m.addVars(salad_types, vtype=GRB.BINARY, name="salad_type")

for t in salad_types:
    t_idx = [i for i in non_mayo_idx if all_items.loc[i, "tipo"] == t]
    m.addConstr(quicksum(x[i] for i in t_idx) <= BIG_M * y[t], name=f"salad_use_{t}")
    m.addConstr(quicksum(x[i] for i in t_idx) >= y[t], name=f"salad_pick_{t}")

m.addConstr(quicksum(y[t] for t in salad_types) >= 2, name="salad_two_types")

# =========================================================
# CHARCOAL, SKEWERS AND LIGHTER
# =========================================================
meat_total_kg = quicksum(
    quantity_in_base(all_items.loc[i]) / 1000.0 * x[i]
    for i in all_items.index
    if all_items.loc[i, "categoria"] in meat_categories
)

m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "carvao"
    ) >= CARVAO_KG_PER_MEAT_KG * meat_total_kg,
    name="charcoal_demand"
)

m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "espetos"
    ) >= ESPETOS_PER_MEAT_KG * meat_total_kg,
    name="skewers_demand"
)

m.addConstr(
    quicksum(
        quantity_in_base(all_items.loc[i]) * x[i]
        for i in all_items.index
        if all_items.loc[i, "categoria"] == "acendedor"
    ) >= 1,
    name="lighter_demand"
)

# =========================================================
# SOLVE
# =========================================================
m.optimize()

# =========================================================
# OUTPUT
# =========================================================
if m.status == GRB.OPTIMAL:
    print("\nSolução ótima encontrada:\n")

    print(f"CATEGORIA{'':7s} | TIPO{'':24s} | MARCA{'':7s} | QTD{'':6s} | UNIDADE{'':21s} | PRECO ")
    for i in all_items.index:
        val = x[i].X
        if val > 1e-6:
            row = all_items.loc[i]
            total_amount = format_total(row, val)

            print(
                f"{row['categoria']:16s} | "
                f"{row['tipo'][:28]:28s} | "
                f"{row['marca']:12s} | "
                f"qtd = {int(round(val)):3d} | "
                f"total = {total_amount:20s} | "
                f"{int(round(val)):1d} * R$ {row['preco']:.2f} = R$ {((int(round(val))) * row['preco']):.2f} "
            )

    print(f"\nCusto total: R$ {m.objVal:.2f}")
else:
    print("Não foi encontrada solução ótima.")
