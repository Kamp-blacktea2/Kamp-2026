
from pathlib import Path
import pandas as pd

# 현재 py 파일이 있는 폴더
HERE = Path(__file__).resolve().parent

# 같은 폴더의 Excel 불러오기
df = pd.read_excel(HERE / "Welding Data Set_01.xlsx")

# 파생변수 4개 생성
df["Resistance"] = df["weld Voltage(v)"] / df["weld current(kA)"]

df["Power"] = df["weld Voltage(v)"] * df["weld current(kA)"]

df["Energy"] = (
    df["weld Voltage(v)"]
    * df["weld current(kA)"]
    * df["weld time(ms)"]
)

df["I2t"] = (
    df["weld current(kA)"] ** 2
    * df["weld time(ms)"]
)

# 원본 4개 + 파생변수 4개
df_8 = df[[
    "weld force(bar)",
    "weld current(kA)",
    "weld Voltage(v)",
    "weld time(ms)",
    "Resistance",
    "Power",
    "Energy",
    "I2t"
]]

# 확인
print(df_8.head())
print("\n행, 열 개수:", df_8.shape)

# 같은 폴더에 저장
df_8.to_csv(HERE / "welding_8features.csv", index=False)

print("\nwelding_8features.csv 저장 완료")

import matplotlib.pyplot as plt

# 8개 변수 각각 히스토그램
for column in df_8.columns:
    plt.figure(figsize=(8, 5))

    plt.hist(df_8[column].dropna(), bins=50)

    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()