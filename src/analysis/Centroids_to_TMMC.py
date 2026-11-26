#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import csv

def read_table(path: Path) -> pd.DataFrame:
    """
    Reads .xlsx or .csv into a DataFrame.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def pick_first_nonnull(series: pd.Series) -> Optional[Any]:
    """Return the first non-null value in a Series, or None."""
    try:
        return next(v for v in series if pd.notna(v))
    except StopIteration:
        return None

# --- Normalize column names (strip spaces, unify case) ---
def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def build_inchikey_dict(
    centroids_df: pd.DataFrame,
    chemprop_df: pd.DataFrame,
    med_df: pd.DataFrame,
    chemical_protein_df: pd.DataFrame = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build the dictionary keyed by InChIKey:
      {
        "InChIKey1": {
           "id_cluster": ...,
           "SMILES": ...,
           "ID": ...,
           "COMPOUNDS": [ ... ]
        },
        ...
      }
    """

    centroids_df = normcols(centroids_df)
    chemprop_df = normcols(chemprop_df)
    med_df = normcols(med_df)

    # --- Validate & map expected columns ---
    # Centroids: InChIKey + id_cluster(s)
    print(centroids_df.columns)
    if "INCHIKEY" not in centroids_df.columns:
        raise KeyError("centroids file must contain an 'InChIKey' column")

    id_cluster_col = None
    for cand in ("id_cluster", "id_clusters", "cluster_id"):
        if cand in centroids_df.columns:
            id_cluster_col = cand
            break
    if id_cluster_col is None:
        raise KeyError(
            "centroids file must contain an 'id_cluster' (or 'id_clusters'/'cluster_id') column"
        )

    # Chemical properties: must have InChIKey and ID; SMILES may be spelled SMILES or SMILIEs
    if "INCHIKEY" not in chemprop_df.columns:
        raise KeyError("chemical_property file must contain an 'InChIKey' column")
    if "ID" not in chemprop_df.columns:
        raise KeyError("chemical_property file must contain an 'ID' column")

    smiles_col = None
    for cand in ("SMILES", "SMILIEs", "SMILIE", "Smiles"):
        if cand in chemprop_df.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        raise KeyError("chemical_property file must contain a 'SMILES' (or 'SMILIEs') column")

    # Medicinal compounds: ID + COMPOUND
    id_col_med = None
    for cand in ("ID", "Id", "id"):
        if cand in med_df.columns:
            id_col_med = cand
            break
    if id_col_med is None:
        raise KeyError("medicinal_compound file must contain an 'ID' column")

    compound_col = None
    for cand in ("COMPOUND", "Compound", "compound"):
        if cand in med_df.columns:
            compound_col = cand
            break
    if compound_col is None:
        raise KeyError("medicinal_compound file must contain a 'COMPOUND' column")

    # --- Build base dict from centroids (InChIKey -> {"id_cluster": ...}) ---
    # If duplicates exist, we pick the first non-null id_cluster per InChIKey.
    base = (
        centroids_df.groupby("INCHIKEY")[id_cluster_col]
        .apply(pick_first_nonnull)
        .rename("id_cluster")
        .to_frame()
        .reset_index()
    )

    # --- Attach chemical properties by InChIKey (SMILES + ID) ---
    # If there are multiple chemprop rows per InChIKey, we take the first non-null in each column.
    chem_agg = (
        chemprop_df.groupby("INCHIKEY")
        .agg({
            "ID": pick_first_nonnull,
            smiles_col: pick_first_nonnull,
        })
        .rename(columns={smiles_col: "SMILES"})
        .reset_index()
    )

    merged = pd.merge(base, chem_agg, on="INCHIKEY", how="left")

    # --- Prepare medicinal mapping: ID -> list of COMPOUNDs ---
    med_map: Dict[Any, List[Any]] = (
        med_df[[id_col_med, compound_col]]
        .dropna(subset=[id_col_med])
        .groupby(id_col_med)[compound_col]
        .apply(lambda s: [v for v in s.tolist() if pd.notna(v)])
        .to_dict()
    )

    # --- Prepare protein mapping: ID -> list of PREFERRED_NAMEs ---
    protein_map: Dict[Any, List[Any]] = {}
    if chemical_protein_df is not None:
        # Normalize columns
        chemical_protein_df = chemical_protein_df.copy()
        chemical_protein_df.columns = [c.strip() for c in chemical_protein_df.columns]
        id_col_prot = None
        for cand in ("ID", "Id", "id"):
            if cand in chemical_protein_df.columns:
                id_col_prot = cand
                break
        preferred_name_col = None
        for cand in ("PREFERRED_NAME", "Preferred_Name", "preferred_name"):
            if cand in chemical_protein_df.columns:
                preferred_name_col = cand
                break
        if id_col_prot and preferred_name_col:
            protein_map = (
                chemical_protein_df[[id_col_prot, preferred_name_col]]
                .dropna(subset=[id_col_prot])
                .groupby(id_col_prot)[preferred_name_col]
                .apply(lambda s: [v for v in s.tolist() if pd.notna(v)])
                .to_dict()
            )

    # --- Final assembly ---
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in merged.iterrows():
        ik = row["INCHIKEY"]
        entry: Dict[str, Any] = {
            "id_cluster": row.get("id_cluster"),
            "SMILES": row.get("SMILES"),
            "ID": row.get("ID"),
        }

        compounds: List[Any] = []
        id_val = entry.get("ID")
        if pd.notna(id_val) and id_val in med_map:
            compounds = med_map[id_val]
        entry["COMPOUNDS"] = compounds

        # Add Proteins key
        proteins: List[Any] = []
        if pd.notna(id_val) and id_val in protein_map:
            proteins = protein_map[id_val]
        entry["Proteins"] = proteins

        result[str(ik)] = entry

    return result


def centroids_to_TMMC(centroids, output, datafolder):

    ## USAGE EXAMPLE: python3 .\Centroids_to_TMMC.py --centroids .\data\centroids_with_inchikey.csv --output .\data --datafolder .\data\

    # Set file paths as strings here for easy modification

    centroids_path = centroids
    
    output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_json_path = str(output_folder / "inchikey_dict.json")
    output_csv_path = str(output_folder / "inchikey_dict.csv")

    datafolder = Path(datafolder)
    chemical_path = str(datafolder / "chemical_property.xlsx")
    medicinal_path = str(datafolder / "medicinal_compound.xlsx")
    chemical_protein_path = str(datafolder / "chemical_protein.xlsx")

    centroids_df = read_table(Path(centroids_path))
    chemprop_df = read_table(Path(chemical_path))
    med_df = read_table(Path(medicinal_path))
    chemical_protein_df = read_table(Path(chemical_protein_path)) 

    result = build_inchikey_dict(centroids_df, chemprop_df, med_df, chemical_protein_df)


    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Prepare rows: flatten lists as semicolon-separated strings
    rows = []
    for ik, entry in result.items():
        row = {
            "INCHIKEY": ik,
            "id_cluster": entry.get("id_cluster"),
            "SMILES": entry.get("SMILES"),
            "ID": entry.get("ID"),
            "COMPOUNDS": ";".join(map(str, entry.get("COMPOUNDS", []))) if entry.get("COMPOUNDS") else "",
            "Proteins": ";".join(map(str, entry.get("Proteins", []))) if entry.get("Proteins") else "",
        }
        rows.append(row)

    # Write CSV
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["INCHIKEY", "id_cluster", "SMILES", "ID", "COMPOUNDS", "Proteins"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(result):,} entries to {output_json_path} and {output_csv_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build InChIKey-indexed dictionary and table from TCM data files.")
    parser.add_argument('--centroids', required=True, help='Path to centroids_with_inchikey.csv or .xlsx')
    parser.add_argument('--output', required=True, help='Output folder for JSON and CSV files')
    parser.add_argument('--datafolder', required=True, help='Path to folder containing chemical_property.xlsx, medicinal_compound.xlsx, chemical_protein.xlsx')
    args = parser.parse_args()

    centroids_to_TMMC(args.centroids, args.output, args.datafolder)

