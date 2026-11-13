# -*- coding: utf-8 -*-
"""Core feature engineering and data loading for the ATL24 + Sentinel-2 SDB workflow.

Reflectance convention: the training tables store SWIR-subtracted bands
(k_Bi = raw_Bi - B11 for the visible/red-edge bands; k_B8, k_B11 are raw).
features() reconstructs raw reflectance by adding B11 back and forms the two
stabilized Stumpf log-ratios.
"""
import os
import numpy as np
import pandas as pd

# resolve data directory (Code Ocean mounts inputs at /data; fall back to ../data)
DATA = "/data" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "..", "data")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B8", "B11", "r23", "r24"]
BLOCK = 500.0          # m, spatial block size
LAND_THR = 0.03        # raw B8 above this is land / mixed
DEPTH_MAX = 15.0       # training/evaluation depth cap (m)
DEPTH_SLICES = [0, 2, 4, 6, 8, 10, 15, 30]

TABLES = {"ayvalik": "ayvalik_training.csv", "edremit_kuzey": "edremit_training.csv"}
ISOLATION = {"ayvalik": "ayvalik_isolation.csv", "edremit_kuzey": "edremit_isolation.csv"}


def safe_ln(x):
    """Stabilized logarithm: smallest value is ln(10) > 0, so ratios stay single-signed."""
    return np.log(1e4 * np.maximum(x, 1e-3))


def features(d, p="k_"):
    """Build the 9-band feature frame from a table with prefixed columns."""
    f = pd.DataFrame(index=d.index)
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        f[b] = d[p + b] + d[p + "B11"]       # reconstruct raw reflectance
    f["B8"], f["B11"] = d[p + "B8"], d[p + "B11"]
    f["r23"] = safe_ln(f.B2) / safe_ln(f.B3)
    f["r24"] = safe_ln(f.B2) / safe_ln(f.B4)
    return f[BANDS]


def load_aegean():
    """Load both Aegean sites, add spatial-block labels, drop land and coarse outliers."""
    frames = []
    for site, fn in TABLES.items():
        d = pd.read_csv(os.path.join(DATA, fn))
        d["alan"] = site
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["blok"] = (d.alan + "_" + (d.px // BLOCK).astype(int).astype(str)
                 + "_" + (d.py // BLOCK).astype(int).astype(str))
    d["kara"] = d.k_B8 > LAND_THR
    iz = pd.concat([pd.read_csv(os.path.join(DATA, ISOLATION[s])).assign(alan=s) for s in TABLES])
    d = d.merge(iz[["alan", "px", "py", "kaba"]], on=["alan", "px", "py"], how="left")
    d = d[~d.kaba.fillna(False).astype(bool)].drop(columns="kaba").reset_index(drop=True)
    return d[~d.kara & (d.derinlik_m <= DEPTH_MAX)].copy()
