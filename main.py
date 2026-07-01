import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from datetime import datetime

from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
from tslearn.metrics import dtw

# Input files
ref_path = r"path/to/reference_timeseries.tif"
cluster_stack_path = r"path/to/cluster_timeseries.tif"
cluster_map_path = r"path/to/cluster_map.tif"

# Parameters
MAX_DAY_DIFFERENCE = 3
MIN_PIXELS_PER_CLUSTER = 50


def read_dates(src):
    """Read acquisition dates from raster band descriptions."""

    dates = []

    for d in src.descriptions:
        d = re.search(r"\d{8}", d).group()
        dates.append(datetime.strptime(d, "%Y%m%d"))

    return np.array(dates)


def match_dates(ref_dates, target_dates, max_difference=3):
    """Match acquisition dates between two time series."""

    ref_idx = []
    target_idx = []

    used_ref = set()

    for j, td in enumerate(target_dates):

        diffs = np.array(
            [abs((td - rd).days) for rd in ref_dates]
        )

        i = np.argmin(diffs)

        if diffs[i] <= max_difference and i not in used_ref:
            ref_idx.append(i)
            target_idx.append(j)
            used_ref.add(i)

    return np.array(ref_idx), np.array(target_idx)


def mean_curve(stack):
    """Compute the mean temporal profile."""

    bands = stack.shape[0]

    X = stack.reshape(bands, -1).T
    X = X[np.all(np.isfinite(X), axis=1)]

    return np.nanmean(X, axis=0)


print("Loading reference...")

with rasterio.open(ref_path) as src:
    ref_stack = src.read().astype(np.float32)
    ref_dates = read_dates(src)

print(f"Reference images: {len(ref_dates)}")

print("=" * 60)
print("STEP 1/7 : Loading reference stack")
print("=" * 60)

print("Loading AOI stack...")

with rasterio.open(cluster_stack_path) as src:
    aoi_stack = src.read().astype(np.float32)
    aoi_dates = read_dates(src)

print(f"AOI images: {len(aoi_dates)}")

print("=" * 60)
print("STEP 2/7 : Matching acquisition dates")
print("=" * 60)

ref_index, aoi_index = match_dates(
    ref_dates,
    aoi_dates,
    MAX_DAY_DIFFERENCE
)

print("\nMatched dates\n")

for r, a in zip(ref_index, aoi_index):
    print(
        ref_dates[r].strftime("%Y-%m-%d"),
        "<-->",
        aoi_dates[a].strftime("%Y-%m-%d"),
        "|",
        abs((ref_dates[r] - aoi_dates[a]).days),
        "days"
    )

ref_stack = ref_stack[ref_index]
aoi_stack = aoi_stack[aoi_index]
matched_dates = aoi_dates[aoi_index]

print(f"\nImages used: {len(matched_dates)}")

with rasterio.open(cluster_map_path) as src:
    cluster_map = src.read(1)

clusters = np.unique(cluster_map)
clusters = clusters[clusters > 0]

print(f"Clusters found: {len(clusters)}")

print("\nExtracting reference curve...")

ref_curve = mean_curve(ref_stack)

print(f"Reference curve length: {len(ref_curve)}")

print("=" * 60)
print("STEP 3/7 : Reshaping AOI stack")
print("=" * 60)

bands, rows, cols = aoi_stack.shape

print(f"Shape = {bands} × {rows} × {cols}")

X = aoi_stack.reshape(bands, -1).T

valid_pixels = np.all(np.isfinite(X), axis=1)

X = X[valid_pixels]
cluster_labels = cluster_map.flatten()[valid_pixels]

print(f"Valid pixels: {len(X):,}")

cluster_curves = {}
cluster_sizes = {}

print("=" * 60)
print("STEP 4/7 : Computing mean curves")
print("=" * 60)

nclusters = len(clusters)

for i, cid in enumerate(clusters, start=1):

    print(f"Cluster {cid:2d} ({i}/{nclusters})")

    pixels = X[cluster_labels == cid]

    if len(pixels) < MIN_PIXELS_PER_CLUSTER:
        continue

    cluster_curves[cid] = np.mean(pixels, axis=0)
    cluster_sizes[cid] = len(pixels)

print(f"Clusters retained: {len(cluster_curves)}")

plt.figure(figsize=(12, 6))

plt.plot(
    matched_dates,
    ref_curve,
    color="black",
    linewidth=3,
    label="Reference"
)

for cid, curve in cluster_curves.items():

    plt.plot(
        matched_dates,
        curve,
        linewidth=1,
        alpha=0.45
    )

plt.grid(True)
plt.xlabel("Date")
plt.ylabel("NDVI")
plt.title("Reference vs Cluster Mean Curves")
plt.legend()
plt.tight_layout()
plt.show()

print("=" * 60)
print("STEP 5/7 : Computing similarity metrics")
print("=" * 60)

results = []

for cid, curve in cluster_curves.items():

    corr = pearsonr(ref_curve, curve)[0]
    rmse = np.sqrt(np.mean((ref_curve - curve) ** 2))
    euc = euclidean(ref_curve, curve)
    cos_sim = 1.0 - cosine(ref_curve, curve)
    dtw_dist = dtw(ref_curve, curve)

    results.append([
        cid,
        cluster_sizes[cid],
        corr,
        rmse,
        euc,
        cos_sim,
        dtw_dist
    ])

df = pd.DataFrame(
    results,
    columns=[
        "Cluster",
        "Pixels",
        "Pearson",
        "RMSE",
        "Euclidean",
        "Cosine",
        "DTW"
    ]
)

print(df)

df["Pearson_N"] = (
    df["Pearson"] - df["Pearson"].min()
) / (
    df["Pearson"].max() - df["Pearson"].min() + 1e-10
)

df["Cosine_N"] = (
    df["Cosine"] - df["Cosine"].min()
) / (
    df["Cosine"].max() - df["Cosine"].min() + 1e-10
)

df["RMSE_N"] = 1 - (
    (df["RMSE"] - df["RMSE"].min()) /
    (df["RMSE"].max() - df["RMSE"].min() + 1e-10)
)

df["Euclidean_N"] = 1 - (
    (df["Euclidean"] - df["Euclidean"].min()) /
    (df["Euclidean"].max() - df["Euclidean"].min() + 1e-10)
)

df["DTW_N"] = 1 - (
    (df["DTW"] - df["DTW"].min()) /
    (df["DTW"].max() - df["DTW"].min() + 1e-10)
)

w_corr = 0.30
w_cos = 0.20
w_rmse = 0.20
w_euc = 0.10
w_dtw = 0.20

df["Score"] = (
    w_corr * df["Pearson_N"] +
    w_cos * df["Cosine_N"] +
    w_rmse * df["RMSE_N"] +
    w_euc * df["Euclidean_N"] +
    w_dtw * df["DTW_N"]
)

df = df.sort_values(
    "Score",
    ascending=False
).reset_index(drop=True)

print("\n========== FINAL RANKING ==========\n")

print(df[
    [
        "Cluster",
        "Pixels",
        "Score",
        "Pearson",
        "Cosine",
        "RMSE",
        "Euclidean",
        "DTW"
    ]
])

df.to_csv("Cluster_Similarity_Ranking.csv", index=False)

print("\nResults saved to Cluster_Similarity_Ranking.csv")

print("=" * 60)
print("STEP 6/7 : Creating figures")
print("=" * 60)

plt.figure(figsize=(14, 8))

plt.plot(
    matched_dates,
    ref_curve,
    color="black",
    linewidth=3,
    label="Reference"
)

for cid in sorted(cluster_curves.keys()):

    plt.plot(
        matched_dates,
        cluster_curves[cid],
        linewidth=1.5,
        alpha=0.8,
        label=f"Cluster {cid}"
    )

plt.xlabel("Date")
plt.ylabel("Mean Value")
plt.title("Reference vs All Cluster Mean Curves")
plt.grid(True)

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=8
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.bar(
    df["Cluster"].astype(str),
    df["Score"]
)

plt.xlabel("Cluster")
plt.ylabel("Similarity Score")
plt.title("Cluster Similarity Ranking")
plt.grid(axis="y")

plt.tight_layout()
plt.show()

BEST_N = 5

best_clusters = df.head(BEST_N)["Cluster"].values

best_map = np.isin(cluster_map, best_clusters).astype(np.uint8)

with rasterio.open(cluster_map_path) as src:
    profile = src.profile.copy()

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw"
)

with rasterio.open("Best_Clusters.tif", "w", **profile) as dst:
    dst.write(best_map, 1)

rank_map = np.zeros_like(cluster_map, dtype=np.uint8)

for rank, cid in enumerate(df["Cluster"], start=1):
    rank_map[cluster_map == cid] = rank

with rasterio.open("Cluster_Rank_Map.tif", "w", **profile) as dst:
    dst.write(rank_map, 1)

print("\n========== TOP CLUSTERS ==========\n")

print(df.head(10)[
    [
        "Cluster",
        "Pixels",
        "Score",
        "Pearson",
        "RMSE",
        "DTW"
    ]
])

TOP_N = 3

top = df.head(TOP_N)

plt.figure(figsize=(12, 6))

plt.plot(
    matched_dates,
    ref_curve,
    "-o",
    color="black",
    linewidth=3,
    markersize=6,
    label="Reference"
)

colors = ["red", "blue", "green"]

for color, (_, row) in zip(colors, top.iterrows()):

    cid = int(row["Cluster"])

    plt.plot(
        matched_dates,
        cluster_curves[cid],
        "-o",
        color=color,
        linewidth=2,
        markersize=5,
        label=f'Rank {top.index[top.Cluster == cid][0] + 1} | Cluster {cid} | Score={row["Score"]:.3f}'
    )

plt.xlabel("Date")
plt.ylabel("Mean Value")
plt.title("Reference vs Top Candidate Clusters")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\n========== TOP 3 ==========\n")

print(
    top[
        [
            "Cluster",
            "Score",
            "Pearson",
            "Cosine",
            "RMSE",
            "DTW"
        ]
    ]
)
