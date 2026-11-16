"""Tahap 1: membersihkan data mentah Pima Diabetes."""
import os

import numpy as np
import pandas as pd
import yaml

# Kolom yang tidak mungkin bernilai 0 pada manusia hidup -> 0 berarti data kosong
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def preprocess(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # 1. Hapus baris duplikat (kalau ada)
    n_before = len(df)
    df = df.drop_duplicates()

    # 2. Ubah 0 yang mustahil menjadi NaN (diisi nanti di train, setelah split)
    zero_counts = (df[ZERO_AS_MISSING] == 0).sum()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    # 3. Simpan DENGAN header supaya train.py bisa memanggil kolom "Outcome"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Baris awal: {n_before}, setelah hapus duplikat: {len(df)}")
    print("Jumlah nilai 0 yang diubah menjadi NaN:")
    print(zero_counts.to_string())
    print(f"Data bersih disimpan ke {output_path}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]
    preprocess(params["input"], params["output"])