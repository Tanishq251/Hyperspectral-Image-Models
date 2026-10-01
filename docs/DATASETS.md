# 📦 Datasets

[← Back to README](../README.md)

---

Datasets are hosted on the HuggingFace Hub at [`Tanishq165/HSI_Datasets`](https://huggingface.co/datasets/Tanishq165/HSI_Datasets) (~20.1 GB, Apache 2.0) and are downloaded automatically the first time you use them — no manual download step. `config/dataset.yaml` registers all 24, storing dimensions, band counts, class counts, per-class sample counts, and human-readable class names, which power dataset listing and dataset-aware visualization.

| # | Dataset | Sensor | Scene Type | Region |
|---|---------|--------|------------|--------|
| 1 | **Augsburg** | DAS Specim | Urban | Augsburg, Germany |
| 2 | **Berlin** | HyMap | Urban | Berlin, Germany |
| 3 | **Botswana** | NASA EO-1 Hyperion | Wetland/Vegetation | Okavango Delta, Africa |
| 4 | **Chikusei** | Headwall Photonics | Agricultural/Urban | Chikusei, Japan |
| 5 | **Dioni** | AVIRIS-NG | Mixed | Dioni, Greece |
| 6 | **Holden** | MRO CRISM | Planetary | Mars |
| 7 | **Houston 2013** | ITRES CASI-1500 | Urban | Houston, TX, USA |
| 8 | **Houston 2018** | AVIRIS-NG | Urban | Houston, TX, USA |
| 9 | **Indian Pines** | NASA AVIRIS | Agriculture/Forest | Indiana, USA |
| 10 | **KSC** | NASA AVIRIS | Wetland/Vegetation | Florida, USA |
| 11 | **Loukia** | AVIRIS-NG | Mixed | Loukia, Greece |
| 12 | **Muufl** | ITRES CASI-1500 | Urban/Vegetation | Mississippi, USA |
| 13 | **NiliFossae** | MRO CRISM | Planetary | Mars |
| 14 | **Pavia Center** | ROSIS | Urban | Pavia, Italy |
| 15 | **Pavia University** | ROSIS | Urban | Pavia, Italy |
| 16 | **Pingan** (QUH) | Gaiasky mini2-VNIR (UAV) | Coastal/Urban | Qingdao, China |
| 17 | **Qingyun** (QUH) | Gaiasky mini2-VNIR (UAV) | Urban/Residential | Qingdao, China |
| 18 | **Salinas** | NASA AVIRIS | Agriculture | Salinas Valley, USA |
| 19 | **Tangdaowan** (QUH) | Gaiasky mini2-VNIR (UAV) | Coastal Wetland Park | Qingdao, China |
| 20 | **Trento** | AISA Eagle | Rural | Trento, Italy |
| 21 | **Utopia** | MRO CRISM | Planetary | Mars |
| 22 | **WHU-Hi-HanChuan** | Headwall Nano | Agricultural | HanChuan, China |
| 23 | **WHU-Hi-HongHu** | Headwall Nano | Agricultural | HongHu, China |
| 24 | **WHU-Hi-LongKou** | Headwall Nano | Agricultural | LongKou, China |

| Dataset | Config Key | Size (H×W) | Bands | Classes | Sensor | Region |
|---------|-----------|:----------:|:-----:|:-------:|--------|--------|
| **Augsburg** | `Augsburg` | 332×485 | 180 | 7 | DAS Specim | Augsburg, Germany |
| **Berlin** | `Berlin` | 1723×476 | 244 | 8 | HyMap | Berlin, Germany |
| **Botswana** | `Botswana` | 1476×256 | 145 | 14 | NASA EO-1 Hyperion | Okavango Delta, Africa |
| **Chikusei** | `Chikusei` | 2517×2335 | 128 | 19 | Headwall Photonics | Chikusei, Japan |
| **Dioni** | `Dioni` | 250×1376 | 176 | 12 | AVIRIS-NG | Dioni, Greece |
| **Holden** | `Holden` | 595×440 | 418 | 6 | MRO CRISM | Mars |
| **Houston 2013** | `Houston13` | 349×1905 | 144 | 15 | ITRES CASI-1500 | Houston, TX, USA |
| **Houston 2018** | `Houston18` | 1202×4768 | 48 | 20 | AVIRIS-NG | Houston, TX, USA |
| **Indian Pines** | `Indian_Pines` | 200×145 | 145 | 16 | NASA AVIRIS | Indiana, USA |
| **KSC** | `KSC` | 512×614 | 176 | 13 | NASA AVIRIS | Florida, USA |
| **Loukia** | `Loukia` | 249×945 | 176 | 14 | AVIRIS-NG | Loukia, Greece |
| **Muufl** | `Muufl` | 325×220 | 64 | 11 | ITRES CASI-1500 | Mississippi, USA |
| **NiliFossae** | `NiliFossae` | 478×593 | 425 | 9 | MRO CRISM | Mars |
| **Pavia Center** | `Pavia Center` | 1096×715 | 102 | 9 | ROSIS | Pavia, Italy |
| **Pavia University** | `Pavia University` | 610×340 | 103 | 9 | ROSIS | Pavia, Italy |
| **Pingan** (QUH) | `Pingan` | 1230×1000 | 176 | 10 | Gaiasky mini2-VNIR (UAV) | Qingdao, China |
| **Qingyun** (QUH) | `Qingyun` | 880×1360 | 176 | 6 | Gaiasky mini2-VNIR (UAV) | Qingdao, China |
| **Salinas** | `Salinas` | 512×217 | 204 | 16 | NASA AVIRIS | Salinas Valley, USA |
| **Tangdaowan** (QUH) | `Tangdoaowan` | 1740×860 | 176 | 18 | Gaiasky mini2-VNIR (UAV) | Qingdao, China |
| **Trento** | `Trento` | 166×600 | 63 | 6 | AISA Eagle | Trento, Italy |
| **Utopia** | `Utopia` | 478×595 | 432 | 9 | MRO CRISM | Mars |
| **WHU-Hi-HanChuan** | `WHU-Hi-HanChuan` | 1217×303 | 274 | 16 | Headwall Nano | HanChuan, China |
| **WHU-Hi-HongHu** | `WHU-Hi-HongHu` | 940×475 | 270 | 22 | Headwall Nano | HongHu, China |
| **WHU-Hi-LongKou** | `WHU-Hi-LongKou` | 550×400 | 270 | 9 | Headwall Nano | LongKou, China |




### 🚁 The QUH Sub-Datasets (Pingan · Qingyun · Tangdaowan)

Three of the scenes above come from the **Qingdao UAV-borne Hyperspectral (QUH)** benchmark — UAV data, not satellite. All three were captured over Qingdao, China with a **Gaiasky mini2-VNIR** imaging spectrometer flown on a **DJI Matrice 600 Pro** at 300 m altitude, giving ~0.15 m ground resolution and **176 bands over 400–1000 nm**. Tangdaowan and Qingyun were surveyed on **18 May 2021**.

Each was designed to stress a different failure mode, which makes them useful as a difficulty axis in a benchmark:

| Dataset | Classes | Designed Challenge |
|---------|:-------:|--------------------|
| **Tangdaowan** | 18 | **High inter-class spectral similarity** — four vegetation species (Coniferous pine, *Buxus sinica*, *Populus*, *Ulmus pumila* L.) and three pavements (Flagging, Boardwalk, Gravel road) with near-identical spectra |
| **Qingyun** | 6 | **Shadow occlusion** — large regions obscured by building shadows over Trees, Car and Asphalt road; a robustness test |
| **Pingan** | 10 | **Extreme scale variation** — very large classes (Seawater, Road) alongside very small ones (Ship, Car) in one scene |

The authors note these are **harder than Indian Pines or Pavia University** because of high inter-class *and* intra-class similarity — worth saying explicitly in a paper, since scores on them are not comparable to the classic scenes.

**Source:** H. Fu et al., *"Three-dimensional singular spectrum analysis for precise land cover classification from UAV-borne hyperspectral benchmark datasets,"* **ISPRS Journal of Photogrammetry and Remote Sensing**, vol. 203, pp. 115–134, 2023. [DOI](https://doi.org/10.1016/j.isprsjprs.2023.07.013) · [Dataset (Zenodo, CC-BY-4.0)](https://zenodo.org/records/8223066) · [GitHub](https://github.com/Hang-Fu/QUH-classification-dataset)

If you use any of these three scenes, cite that paper in addition to this framework.


```bash
python main.py --list-datasets
```

Full per-dataset details (sensors, resolutions, class breakdowns, loading notes) live on the dataset card at [🤗 Tanishq165/HSI_Datasets](https://huggingface.co/datasets/Tanishq165/HSI_Datasets). That card is especially useful when preparing figures or writing a methods section and you need the exact class names used by the codebase.

**Loading notes:** the loader auto-transposes to `(H, W, Bands)`, treats class `0` as background/unlabeled and excludes it from training, and falls back to `h5py` for HDF5-based v7.3 `.mat` files. Several scenes (Indian Pines, Houston18) are heavily class-imbalanced — consider weighted loss or balanced sampling.

---
