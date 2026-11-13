# -*- coding: utf-8 -*-
"""
Reproduce the quantitative results of:
"Desk-based satellite-derived bathymetry from ICESat-2 ATL24 and Sentinel-2 for
the Turkish Aegean coast, with independent reference-grade validation."

Outputs (written to ./results or /results on Code Ocean):
  - results.txt : Tables 1-2, transfer, confidence sensitivity, bottom-type, CIs
  - figure2_florida_validation.png : ATL24 vs lidar and SDB vs lidar + depth bars

Run: python run.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sdb_core as C

RESULTS = "/results" if os.path.isdir("/results") else os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
rng = np.random.default_rng(42)
rmse = lambda e: float(np.sqrt((e ** 2).mean()))
mae = lambda e: float(np.abs(e).mean())
def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((p - y) ** 2).sum() / ss) if ss > 0 else float("nan")
FL_SUBSAMPLE = 30000   # Florida pixels used in the capsule run (full set used in the paper)
def forest(trees=400):
    return RandomForestRegressor(trees, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)

out = open(os.path.join(RESULTS, "results.txt"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

def block_codes(df):
    bl = (df.px // C.BLOCK).astype(int).astype(str) + "_" + (df.py // C.BLOCK).astype(int).astype(str)
    return pd.factorize(bl)[0]

def block_cv(df, cols=None, kind="rf", trees=400):
    X = C.features(df); y = df.derinlik_m.values
    if cols is None: cols = C.BANDS
    code = block_codes(df); pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(5).split(X, groups=code):
        if kind == "rf":
            m = forest(trees).fit(X.iloc[tr][cols], y[tr])
        else:
            m = LinearRegression().fit(X.iloc[tr][cols], y[tr])
        pred[te] = np.clip(m.predict(X.iloc[te][cols]), 0, None)
    return pred, y

def block_ci(err, code, n=2000):
    e2 = err ** 2
    top = np.bincount(code, weights=e2); cnt = np.bincount(code); nb = len(top); v = []
    for _ in range(n):
        s = rng.integers(0, nb, nb); v.append(np.sqrt(top[s].sum() / cnt[s].sum()))
    return np.percentile(v, [2.5, 97.5])

log("=" * 70)
log("ATL24 + Sentinel-2 SDB — reproduction of reported results")
log("=" * 70)

# ---------- Table 1: Aegean model comparison ----------
log("\n[Table 1] Aegean depth-retrieval accuracy (500 m block cross-validation)")
d = C.load_aegean()
models = [("Stumpf blue/green", ["r23"], "lin"), ("Stumpf blue/red", ["r24"], "lin"),
          ("Lyzenga linear", ["B1", "B2", "B3", "B4", "B5"], "lin"), ("Random forest", C.BANDS, "rf")]
for site in C.TABLES:
    dd = d[d.alan == site].reset_index(drop=True)
    log(f"  {site} (n={len(dd)}):")
    for name, cols, kind in models:
        p, y = block_cv(dd, cols, kind); e = p - y
        log(f"    {name:18s} RMSE={rmse(e):.2f}  MAE={mae(e):.2f}  bias={e.mean():+.2f}  R2={r2(y,p):.2f}")

# ---------- Transfer + CI ----------
log("\n[Transfer] train one Aegean site, test the other (block-bootstrap 95% CI)")
for tr_s, te_s in [("ayvalik", "edremit_kuzey"), ("edremit_kuzey", "ayvalik")]:
    S = d[d.alan == tr_s]; T = d[d.alan == te_s].reset_index(drop=True)
    m = forest().fit(C.features(S), S.derinlik_m.values)
    e = np.clip(m.predict(C.features(T)), 0, None) - T.derinlik_m.values
    ci = block_ci(e, block_codes(T))
    log(f"  {tr_s} -> {te_s}: RMSE={rmse(e):.2f} CI[{ci[0]:.2f}-{ci[1]:.2f}] n={len(T)}")

# ---------- Florida reference-grade validation (Table 2) ----------
log("\n[Table 2] Florida reference-grade validation vs NOAA lidar")
ft = pd.read_csv(os.path.join(C.DATA, "florida_training.csv"))
ft = ft[(ft.k_B8 <= C.LAND_THR) & (ft.derinlik_m <= C.DEPTH_MAX)]
val = pd.read_csv(os.path.join(C.DATA, "florida_lidar_validation.csv"))
m = ft.merge(val[["px", "py", "lidar", "gecerli"]], on=["px", "py"], how="left")
m = m[m.gecerli.fillna(False).astype(bool)].reset_index(drop=True)
m = m[(m.lidar >= 0) & (m.lidar <= 10)].reset_index(drop=True)
if len(m) > FL_SUBSAMPLE:
    m = m.sample(FL_SUBSAMPLE, random_state=0).reset_index(drop=True)
    log(f"  (capsule uses a {FL_SUBSAMPLE:,}-pixel random subset of Florida; the paper uses all pixels)")
sdb, lidar = block_cv(m, trees=200)[0], m.lidar.values
atl = m.derinlik_m.values
code = block_codes(m)
a_ci, s_ci = block_ci(atl - lidar, code), block_ci(sdb - lidar, code)
log(f"  ATL24 labels vs lidar: RMSE={rmse(atl-lidar):.2f} CI[{a_ci[0]:.2f}-{a_ci[1]:.2f}] "
    f"MAE={mae(atl-lidar):.2f} bias={(atl-lidar).mean():+.2f} R2={r2(lidar,atl):.2f} n={len(m)}")
log(f"  Sentinel-2 SDB vs lidar: RMSE={rmse(sdb-lidar):.2f} CI[{s_ci[0]:.2f}-{s_ci[1]:.2f}] "
    f"MAE={mae(sdb-lidar):.2f} bias={(sdb-lidar).mean():+.2f} R2={r2(lidar,sdb):.2f}")
log(f"  non-overlapping CIs (denoising significant): {a_ci[0] > s_ci[1]}")
log("  RMSE by depth band (ATL24 / SDB):")
depth_bins = {}
for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
    mm = (lidar > lo) & (lidar <= hi)
    if mm.sum() > 5:
        depth_bins[f"{lo}-{hi}"] = (rmse((atl - lidar)[mm]), rmse((sdb - lidar)[mm]), int(mm.sum()))
        log(f"    {lo}-{hi} m (n={int(mm.sum())}): {depth_bins[f'{lo}-{hi}'][0]:.2f} / {depth_bins[f'{lo}-{hi}'][1]:.2f}")

# ---------- Confidence-threshold sensitivity ----------
log("\n[Sensitivity] ATL24 confidence threshold (Florida vs lidar)")
for thr in [0.6, 0.7, 0.8]:
    sub = m[m.guven >= thr].reset_index(drop=True)
    p = block_cv(sub, trees=200)[0]; lid = sub.lidar.values
    log(f"  conf>={thr}: ATL24={rmse(sub.derinlik_m.values-lid):.2f}  SDB={rmse(p-lid):.2f}  n={len(sub)}")

# ---------- Bottom-type error (MEDISEH Posidonia) ----------
log("\n[Bottom type] block-CV RMSE by MEDISEH P. oceanica probability (0-6 m)")
try:
    import rasterio
    from pyproj import Transformer
    with rasterio.open(os.path.join(C.DATA, "posidonia_mediseh.tiff")) as r:
        PZ = r.read(1).astype(float); PT = r.transform
    PZ[(PZ < 0) | (PZ > 1)] = np.nan
    TR = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
    parts = []
    for site in C.TABLES:
        dd = d[d.alan == site].reset_index(drop=True); dd = dd.assign(pred=block_cv(dd)[0])
        lon, lat = TR.transform(dd.px.values, dd.py.values)
        rr, cc = rasterio.transform.rowcol(PT, lon, lat); rr, cc = np.asarray(rr), np.asarray(cc)
        ok = (rr >= 0) & (rr < PZ.shape[0]) & (cc >= 0) & (cc < PZ.shape[1])
        dd["P"] = np.nan; dd.loc[ok, "P"] = PZ[rr[ok], cc[ok]]
        parts.append(dd[(dd.derinlik_m <= 6) & dd.P.notna()])
    A = pd.concat(parts)
    A["cls"] = pd.cut(A.P, [0, 0.2, 0.4, 1.0],
                      labels=["sand/bare(<0.2)", "mixed(0.2-0.4)", "seagrass(>=0.4)"], include_lowest=True)
    for c, g in A.groupby("cls", observed=True):
        e = g.pred - g.derinlik_m
        log(f"  {c:18s} n={len(g):5d}  RMSE={rmse(e):.2f}  bias={e.mean():+.2f}")
except Exception as ex:
    log(f"  (skipped bottom-type: {ex})")

# ---------- Figure 2 ----------
fig = plt.figure(figsize=(7.2, 6.4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.3, 1.0], hspace=0.38, wspace=0.28)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
for ax, (lab, nm, pred) in zip(axes, [("a", "ATL24 photons", atl), ("b", "Random forest SDB", sdb)]):
    hb = ax.hexbin(lidar, pred, gridsize=55, bins="log", cmap="turbo", mincnt=1, extent=(0, 11, 0, 11))
    ax.plot([0, 11], [0, 11], "k--", lw=1.0, alpha=0.7); ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"RMSE={rmse(pred-lidar):.2f} m\nR2={r2(lidar,pred):.2f}\nn={len(lidar):,}",
            transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
    ax.set_title(f"({lab}) {nm} vs. lidar", fontsize=9, loc="left")
    ax.set_xlabel("NOAA lidar depth (m)", fontsize=8.5); ax.set_ylabel("Estimated depth (m)", fontsize=8.5)
fig.colorbar(hb, ax=axes, fraction=0.035, pad=0.02).set_label("point count (log)", fontsize=8)
axc = fig.add_subplot(gs[1, :])
labels = list(depth_bins); x = np.arange(len(labels)); w = 0.38
axc.bar(x - w/2, [depth_bins[k][0] for k in labels], w, label="ATL24", color="#c44e52")
axc.bar(x + w/2, [depth_bins[k][1] for k in labels], w, label="SDB", color="#4c72b0")
axc.set_xticks(x); axc.set_xticklabels([k + " m" for k in labels]); axc.set_ylabel("RMSE vs. lidar (m)", fontsize=8.5)
axc.set_title("(c) Error by depth band", fontsize=9, loc="left"); axc.legend(fontsize=8, frameon=False); axc.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(RESULTS, "figure2_florida_validation.png"), dpi=300, bbox_inches="tight")
log("\nSaved figure2_florida_validation.png")
log("Done.")
out.close()
