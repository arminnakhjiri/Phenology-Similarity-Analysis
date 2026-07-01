# Raster Time-Series Similarity Analysis

This repository provides a **Python workflow** for identifying raster clusters whose temporal behavior most closely matches a reference time series. The workflow combines multiple similarity metrics—including Pearson correlation, RMSE, Euclidean distance, Cosine similarity, and Dynamic Time Warping (DTW)—into a single ranking score for objective comparison.

Although developed for vegetation phenology using NDVI, the methodology is applicable to any raster time-series variable.

---

## 📌 Overview

- **Input 1:** Reference time-series GeoTIFF
- **Input 2:** Time-series GeoTIFF of the study area
- **Input 3:** Cluster map generated from time-series clustering
- **Method:** Multi-metric similarity analysis
- **Output:** Ranked cluster table, binary map of best-matching clusters, and cluster rank map

---

## 🧰 Python Dependencies

```bash
pip install numpy pandas rasterio matplotlib scipy tslearn
```

---

## 🧪 Workflow

The script performs the following steps:

1. Load the reference and study-area time-series rasters.
2. Match acquisition dates between the two datasets.
3. Compute the mean temporal profile of the reference area.
4. Compute the mean temporal profile of each cluster.
5. Calculate multiple similarity metrics for every cluster:
   - Pearson correlation
   - RMSE
   - Euclidean distance
   - Cosine similarity
   - Dynamic Time Warping (DTW)
6. Normalize the metrics and combine them into a weighted similarity score.
7. Rank all clusters according to their overall similarity.
8. Export tables, figures, and GeoTIFF outputs.

---

## 📊 Similarity Metrics

The final score combines complementary measures of similarity:

| Metric | Interpretation |
|---------|----------------|
| Pearson correlation | Linear agreement between curves |
| Cosine similarity | Shape similarity |
| RMSE | Absolute difference |
| Euclidean distance | Overall curve distance |
| DTW | Similarity allowing temporal shifts |

Each metric is normalized before computing the weighted final score.

---

## ⚙️ Adjustable Parameters

```python
MAX_DAY_DIFFERENCE = 3
MIN_PIXELS_PER_CLUSTER = 50

w_corr = 0.30
w_cos = 0.20
w_rmse = 0.20
w_euc = 0.10
w_dtw = 0.20
```

Users can modify the date matching tolerance, minimum cluster size, and weighting scheme according to their application.

---

## 📁 Outputs

The workflow produces:

- **Cluster_Similarity_Ranking.csv** — ranked similarity table
- **Best_Clusters.tif** — binary raster containing the top-ranked clusters
- **Cluster_Rank_Map.tif** — raster where pixel values represent cluster rank
- Multiple diagnostic figures comparing temporal profiles and ranking scores

---

## Applications

Typical applications include:

- Crop identification
- Phenology matching
- Land-cover comparison
- Vegetation monitoring
- Reference-based classification
- Environmental time-series analysis
- Unsupervised cluster interpretation

---

## 🔗 Related Repositories

This project is part of a three-step raster time-series analysis workflow:

1. **Raster-TimeSeries-Builder**  
   Automatically extracts a specified band from multiple raster stacks and combines it into a chronologically ordered multi-band GeoTIFF.  
   https://github.com/arminnakhjiri/Raster-TimeSeries-Builder

2. **Raster-TimeSeries-Clustering-DTW-KMeans**  
   Clusters raster pixel time series using Dynamic Time Warping (DTW) K-Means to identify regions with similar temporal behavior.  
   https://github.com/arminnakhjiri/Raster-TimeSeries-Clustering-DTW-KMeans

3. **Raster-TimeSeries-Similarity-Analysis** *(this repository)*  
   Compares clustered temporal signatures against a reference time series using multiple similarity metrics and ranks the clusters by overall similarity.

---

## 📄 License

MIT License

---

## Author

**Armin Nakhjiri**

Remote Sensing Scientist

✉️ Nakhjiri.Armin@gmail.com

---

*Simple tools for efficient geospatial data processing.*
