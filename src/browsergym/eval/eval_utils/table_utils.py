import pandas as pd
def table_exact_match(df1: pd.DataFrame, df2: pd.DataFrame, ignore_case: bool = False) -> bool:
    if df1.shape != df2.shape:
        return False

    if ignore_case:
        df1 = df1.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)
        df2 = df2.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)

    return df1.equals(df2)

def table_column_check(df: pd.DataFrame, required_columns: list) -> bool:
    return all(col in df.columns for col in required_columns)

def table_number_tolerance(df: pd.DataFrame, gold_dict: dict, nutrient_cols: list, rel_tol: float = 0.05) -> bool:
    for _, row in df.iterrows():
        ingredient = str(row["Ingredient"]).lower().strip()

        if ingredient not in gold_dict:
            print(f"Ingredient '{ingredient}' not in gold data.")
            return False

        for nutrient in nutrient_cols:
            if nutrient not in row or nutrient not in gold_dict[ingredient]:
                print(f"Missing {nutrient} for {ingredient}")
                return False

            try:
                actual = float(row[nutrient])
                expected = float(gold_dict[ingredient][nutrient])
                if not np.isclose(actual, expected, rtol=rel_tol):
                    print(f"{ingredient} → {nutrient} mismatch: got {actual}, expected {expected}")
                    return False
            except Exception as e:
                print(f"Error comparing {ingredient} → {nutrient}: {e}")
                return False

    return True

def table_partial_match(df: pd.DataFrame, gold_df: pd.DataFrame, key_column: str = "Ingredient") -> bool:
    df_keys = set(df[key_column].str.lower().str.strip())
    gold_keys = set(gold_df[key_column].str.lower().str.strip())

    missing = gold_keys - df_keys
    if missing:
        print(f"Missing values: {missing}")
        return False
    return True