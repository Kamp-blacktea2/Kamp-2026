# ============================================================
# KAMP - 행별 Current 분포 K-Means
#
# 목적:
# 1. 모든 생산 행의 Current를 K-Means로 군집화
# 2. Low / Middle / High Current 군집 확인
# 3. 날짜별 각 군집 비율 계산
# 4. 날짜별 Current MAD와 비교
# 5. 실제 불량률과 비교
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# 1. 파일 위치
# ============================================================

BASE = Path(
    r"C:\Users\sj789\OneDrive\바탕 화면\KAMP 데이터분석"
)

BOOK = (
    BASE
    / "data"
    / "Welding Data Set_01.xlsx"
)

if not BOOK.exists():
    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다:\n{BOOK}"
    )


# ============================================================
# 2. 데이터 불러오기
# ============================================================

raw = pd.read_excel(
    BOOK,
    sheet_name="Raw data"
)

result = pd.read_excel(
    BOOK,
    sheet_name="result"
)

CURRENT = "weld current(kA)"
TIME = "working time"


print("Raw data 크기:", raw.shape)
print("Result 크기:", result.shape)


# ============================================================
# 3. 날짜 생성
# ============================================================

raw["date"] = pd.to_datetime(
    raw[TIME],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

result["date"] = pd.to_datetime(
    result[TIME],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


raw[CURRENT] = pd.to_numeric(
    raw[CURRENT],
    errors="coerce"
)

result["defect"] = pd.to_numeric(
    result["defect"],
    errors="coerce"
)


raw = raw.dropna(
    subset=["date", CURRENT]
).copy()


print("분석 행 수:", len(raw))


# ============================================================
# 4. 행별 Current를 K-Means 입력으로 사용
# ============================================================

X = raw[
    [CURRENT]
].copy()


# ============================================================
# 5. Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 6. K-Means
#
# K = 3
#
# Low Current
# Middle Current
# High Current
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

raw["cluster"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 7. Cluster 중심값 원래 kA 단위로 복원
# ============================================================

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
).flatten()


center_table = pd.DataFrame({
    "cluster": range(3),
    "current_center": centers
})


center_table = center_table.sort_values(
    "current_center"
)


print("\n===== Cluster 중심 =====")
print(center_table.round(5))


# ============================================================
# 8. Low / Middle / High 이름 자동 지정
# ============================================================

ordered_clusters = (
    center_table["cluster"]
    .tolist()
)


cluster_names = {
    ordered_clusters[0]: "Low Current",
    ordered_clusters[1]: "Middle Current",
    ordered_clusters[2]: "High Current"
}


raw["current_group"] = (
    raw["cluster"]
    .map(cluster_names)
)


# ============================================================
# 9. Cluster별 실제 Current 분포
# ============================================================

cluster_summary = (
    raw
    .groupby("current_group")[CURRENT]
    .agg(
        count="size",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    )
)


print("\n")
print("=" * 70)
print("Cluster별 Current 분포")
print("=" * 70)

print(
    cluster_summary.round(5)
)


# ============================================================
# 10. Silhouette Score
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    raw["cluster"]
)


print(
    "\nSilhouette Score:",
    round(silhouette, 4)
)


# ============================================================
# 11. 전체 Current 분포 + K-Means 군집
# ============================================================

plt.figure(
    figsize=(11, 6)
)


for group in [
    "Low Current",
    "Middle Current",
    "High Current"
]:

    values = raw.loc[
        raw["current_group"] == group,
        CURRENT
    ]

    plt.hist(
        values,
        bins=40,
        alpha=0.5,
        label=group
    )


plt.xlabel(
    "Weld Current (kA)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Weld Current Distribution - K-Means Clusters"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 12. MAD 함수
# ============================================================

def mad(x):

    x = np.asarray(x)

    median = np.median(x)

    return np.median(
        np.abs(x - median)
    )


# ============================================================
# 13. 날짜별 Current MAD
# ============================================================

daily = (
    raw
    .groupby("date")
    .agg(

        production_count=(
            CURRENT,
            "size"
        ),

        current_mean=(
            CURRENT,
            "mean"
        ),

        current_MAD=(
            CURRENT,
            mad
        )
    )
)


# ============================================================
# 14. 날짜별 K-Means Cluster 비율
# ============================================================

cluster_ratio = (
    pd.crosstab(
        raw["date"],
        raw["current_group"],
        normalize="index"
    )
    * 100
)


# 없는 군집이 있더라도 오류 방지
for col in [
    "Low Current",
    "Middle Current",
    "High Current"
]:

    if col not in cluster_ratio.columns:
        cluster_ratio[col] = 0


cluster_ratio = cluster_ratio[
    [
        "Low Current",
        "Middle Current",
        "High Current"
    ]
]


daily = daily.join(
    cluster_ratio
)


# ============================================================
# 15. 날짜별 실제 불량률
# ============================================================

defect_daily = (
    result
    .dropna(
        subset=["date", "defect"]
    )
    .groupby("date")["defect"]
    .sum(min_count=1)
)


daily["defect_count"] = (
    defect_daily
)


daily["defect_rate"] = (

    daily["defect_count"]

    /

    daily["production_count"]

    * 100
)


# ============================================================
# 16. 날짜별 최종 결과
# ============================================================

print("\n")
print("=" * 70)
print("날짜별 Current Cluster 분포")
print("=" * 70)


print(
    daily[
        [
            "production_count",

            "current_mean",

            "current_MAD",

            "Low Current",

            "Middle Current",

            "High Current",

            "defect_rate"
        ]
    ].round(4)
)


# ============================================================
# 17. 품질정보 존재 날짜
# ============================================================

quality = daily.dropna(
    subset=["defect_rate"]
).copy()


# ============================================================
# 18. 불량률과 관계
# ============================================================

print("\n")
print("=" * 70)
print("각 지표 ↔ 실제 불량률")
print("=" * 70)


columns = [

    "current_MAD",

    "Low Current",

    "Middle Current",

    "High Current"

]


for col in columns:

    pearson = (
        quality[col]
        .corr(
            quality["defect_rate"],
            method="pearson"
        )
    )

    spearman = (
        quality[col]
        .corr(
            quality["defect_rate"],
            method="spearman"
        )
    )

    print(
        f"\n{col}"
    )

    print(
        f" Pearson  = {pearson:.4f}"
    )

    print(
        f" Spearman = {spearman:.4f}"
    )


# ============================================================
# 19. MAD와 Cluster 비율 관계
# ============================================================

print("\n")
print("=" * 70)
print("Current MAD ↔ Cluster 비율")
print("=" * 70)


for col in [

    "Low Current",

    "Middle Current",

    "High Current"

]:

    corr = (
        daily["current_MAD"]
        .corr(
            daily[col]
        )
    )

    print(
        f"MAD vs {col}: "
        f"{corr:.4f}"
    )


# ============================================================
# 20. 날짜별 Cluster 구성비 그래프
# ============================================================

plot_ratio = daily[
    [
        "Low Current",
        "Middle Current",
        "High Current"
    ]
].copy()


plot_ratio.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Ratio (%)"
)

plt.title(
    "Daily Current Cluster Distribution"
)

plt.xticks(
    rotation=45
)

plt.legend(
    title="Current Cluster"
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. MAD vs Low Current 비율
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    quality["current_MAD"],
    quality["Low Current"],
    s=100
)


for date, row in quality.iterrows():

    plt.annotate(
        date[5:],
        (
            row["current_MAD"],
            row["Low Current"]
        )
    )


plt.xlabel(
    "Current MAD"
)

plt.ylabel(
    "Low Current Cluster Ratio (%)"
)

plt.title(
    "Current MAD vs Low Current Ratio"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. Low Current 비율 vs 불량률
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    quality["Low Current"],
    quality["defect_rate"],
    s=100
)


for date, row in quality.iterrows():

    plt.annotate(
        date[5:],
        (
            row["Low Current"],
            row["defect_rate"]
        )
    )


plt.xlabel(
    "Low Current Cluster Ratio (%)"
)

plt.ylabel(
    "Actual Defect Rate (%)"
)

plt.title(
    "Low Current Ratio vs Defect Rate"
)

plt.tight_layout()

plt.show()


# ============================================================
# 23. 결과 저장
# ============================================================

OUTPUT = BASE / "results"

OUTPUT.mkdir(
    exist_ok=True
)


raw.to_csv(
    OUTPUT / "kmeans_current_row_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)


daily.to_csv(
    OUTPUT / "kmeans_current_daily_summary.csv",
    encoding="utf-8-sig"
)


print("\n분석 완료")

print(
    OUTPUT / "kmeans_current_row_clusters.csv"
)

print(
    OUTPUT / "kmeans_current_daily_summary.csv"
)