# sdb-atl24-sentinel2-aegean
# Desk-based satellite-derived bathymetry from ICESat-2 ATL24 and Sentinel-2

Reproducible code and data for the manuscript:
**"Desk-based satellite-derived bathymetry from ICESat-2 ATL24 and Sentinel-2 for the
Turkish Aegean coast, with independent reference-grade validation."**
Fuat Buğra İdrisoğlu, Balıkesir University. ORCID 0009-0000-9235-9341.

## What this repository does

Running it reproduces the quantitative results of the paper from the bundled
training tables and the Florida reference-lidar table:

- **Table 1** — random forest vs. empirical SDB models at Ayvalık and the Gulf of
  Edremit, under 500 m spatial block cross-validation.
- **Transfer** — train one Aegean site, test the other, with block-bootstrap 95% CIs.
- **Table 2** — Florida reference-grade validation against NOAA lidar: ATL24 labels
  vs. lidar and Sentinel-2 random forest SDB vs. lidar, with 95% CIs and per-depth-band
  error (the main "denoising" result).
- **Sensitivity** — ATL24 confidence threshold (0.6 / 0.7 / 0.8).
- **Bottom type** — block-CV RMSE by MEDISEH *Posidonia oceanica* probability class.
- **Figure 2** — the Florida validation density scatters and depth-band error bars.

Outputs are written to `/results` (`results.txt` and `figure2_florida_validation.png`).

## How to run

On Code Ocean, click **Reproducible Run**; the entry point is `code/run.sh`.
Locally:

```bash
pip install -r environment/requirements.txt
cd code && python run.py
```

Runtime is a few minutes. To reproduce the Florida numbers at full size, set
`FL_SUBSAMPLE` in `code/run.py` to a value larger than the table (the capsule uses a
30,000-pixel random subset for speed; the paper uses all pixels, giving 0.73 m for
ATL24 and 0.35 m for the SDB, within rounding of the subset values).

## Data

All inputs in `data/` are derived from openly available sources:

- `ayvalik_training.csv`, `edremit_training.csv` — ATL24 depths aggregated to the
  Sentinel-2 10 m grid with the SWIR-subtracted reflectance bands (columns `k_B1`…`k_B11`).
- `florida_training.csv` — the same for the Sugarloaf Key (Florida Keys) reference site.
- `florida_lidar_validation.csv` — co-located 2016 NOAA NGS topobathymetric lidar depth
  per pixel, used as the independent reference.
- `ayvalik_isolation.csv`, `edremit_isolation.csv` — coarse-outlier flags.
- `posidonia_mediseh.tiff` — MEDISEH *P. oceanica* occurrence-probability layer.

Primary products are openly available and are not redistributed here: ICESat-2 ATL24
(NASA NSIDC, https://nsidc.org/data/ATL24), Sentinel-2 L2A (Copernicus / Google Earth
Engine), and the NOAA lidar (NOAA Digital Coast, https://coast.noaa.gov/dataviewer).

## Reflectance convention

Training tables store `k_Bi = raw_Bi − B11` for the visible/red-edge bands, and raw
values for `B8` and `B11`. `sdb_core.features()` reconstructs raw reflectance by adding
B11 back and forms the two stabilized Stumpf log-ratios.

## Note on AI assistance

A large language model (Claude, Anthropic) assisted with code development and language
editing. All analyses and results were checked by the author.

## License

Code: MIT. Bundled derived data: CC BY 4.0.
