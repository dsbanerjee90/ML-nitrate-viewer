import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm

# -------------------------------
# USER SETTINGS
# -------------------------------
BASE_DIR = Path("/data/proteus4/backup/dba/Documents/PML/projects/ML/dream_fresh/FINAL_withbathyrivmod/FINAL_predict_2/upload/compressed/")
FILE_PATTERN = "NO3_predictions_RF_paral_{year}_compressed.nc"

APP_TITLE = "Machine-Learned Predicted Nitrate"
APP_SUBTITLE = "Interactive viewer for daily gridded surface nitrate (1998–2018)"

DOI_TEXT = (
    "Banerjee et al. "
    "(DOI: 10.5194/bg-22-3769-2025; DOI: 10.1002/qj.70156)"
)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# -------------------------------
# HELPERS
# -------------------------------
@st.cache_data
def list_files():
    files = []
    for year in range(1998, 2019):
        f = BASE_DIR / FILE_PATTERN.format(year=year)
        if f.exists():
            files.append(f)
    return files

@st.cache_data
def open_dataset():
    files = list_files()
    if not files:
        raise FileNotFoundError("No NetCDF files found.")

    ds = xr.open_mfdataset(files, combine="by_coords")

    var_name = "prediction"
    time_name = "time"
    lat_name = "lat"
    lon_name = "lon"

    if var_name not in ds.variables:
        raise ValueError(f"Variable '{var_name}' not found.")
    if time_name not in ds.variables and time_name not in ds.coords:
        raise ValueError(f"Time variable '{time_name}' not found.")
    if lat_name not in ds.variables and lat_name not in ds.coords:
        raise ValueError(f"Latitude variable '{lat_name}' not found.")
    if lon_name not in ds.variables and lon_name not in ds.coords:
        raise ValueError(f"Longitude variable '{lon_name}' not found.")

    return ds, var_name, time_name, lat_name, lon_name

def mask_invalid(da):
    return xr.where(np.isfinite(da) & (da < 1e20) & (da > -1e20), da, np.nan)

def nearest_valid_ij(lat1d, lon1d, valid_mask_2d, click_lat, click_lon):
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    dist2 = (lat2d - click_lat) ** 2 + (lon2d - click_lon) ** 2
    dist2 = np.where(valid_mask_2d, dist2, np.nan)

    if np.all(np.isnan(dist2)):
        return None, None

    j, i = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(j), int(i)

def make_pretty_cmap():
    cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")  # light grey for land/masked points
    return cmap

# -------------------------------
# LOAD DATA
# -------------------------------
try:
    ds, var_name, time_name, lat_name, lon_name = open_dataset()
except Exception as e:
    st.error(f"Error opening dataset: {e}")
    files = list_files()
    if files:
        ds_debug = xr.open_dataset(files[0])
        st.write("First file:", str(files[0]))
        st.write("Variables:", list(ds_debug.variables))
        st.write("Coordinates:", list(ds_debug.coords))
        st.write("Dimensions:", dict(ds_debug.sizes))
    st.stop()

times = ds[time_name].values
dates = np.array(times).astype("datetime64[D]")

if len(dates) == 0:
    st.error("No time values found in dataset.")
    st.stop()

lat = ds[lat_name]
lon = ds[lon_name]

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Controls")

selected_date = st.sidebar.select_slider(
    "Select date",
    options=dates,
    value=dates[0]
)

# -------------------------------
# EXTRACT MAP DATA
# -------------------------------
data2d = ds[var_name].sel({time_name: selected_date}, method="nearest")
data2d = mask_invalid(data2d)

actual_time = data2d[time_name].values
actual_date = np.datetime_as_string(actual_time, unit="D")

# -------------------------------
# SPATIAL PLOT
# -------------------------------
st.subheader(f"Spatial map — {actual_date}")

if np.all(np.isnan(data2d.values)):
    st.warning("All values are invalid/NaN for this selected date.")
else:
    cmap = make_pretty_cmap()
    data_ma = np.ma.masked_invalid(data2d.values)

    valid_min = float(np.nanpercentile(data2d.values, 2))
    valid_max = float(np.nanpercentile(data2d.values, 98))

    if np.isclose(valid_min, valid_max):
        valid_min = float(np.nanmin(data2d.values))
        valid_max = float(np.nanmax(data2d.values))

    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
    ax.set_facecolor("#d9d9d9")

    pcm = ax.pcolormesh(
        lon.values,
        lat.values,
        data_ma,
        shading="auto",
        cmap=cmap,
        vmin=valid_min,
        vmax=valid_max
    )

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.96)
    cbar.set_label("Surface nitrate", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Machine-Learned Predicted Surface Nitrate", fontsize=16, pad=12, weight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(labelsize=10)

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)

    ax.text(
        0.99, 0.015,
        DOI_TEXT,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, style="italic",
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3)
    )

    st.pyplot(fig, clear_figure=True)

# -------------------------------
# LOCATION INSPECTION
# -------------------------------
st.subheader("Inspect a location")

min_lat = float(np.nanmin(lat.values))
max_lat = float(np.nanmax(lat.values))
min_lon = float(np.nanmin(lon.values))
max_lon = float(np.nanmax(lon.values))

col1, col2 = st.columns(2)

with col1:
    click_lat = st.number_input(
        "Latitude",
        value=float((min_lat + max_lat) / 2),
        min_value=min_lat,
        max_value=max_lat,
        format="%.4f"
    )

with col2:
    click_lon = st.number_input(
        "Longitude",
        value=float((min_lon + max_lon) / 2),
        min_value=min_lon,
        max_value=max_lon,
        format="%.4f"
    )

valid_mask_2d = np.isfinite(data2d.values)
j, i = nearest_valid_ij(lat.values, lon.values, valid_mask_2d, click_lat, click_lon)

if j is None or i is None:
    st.error("No valid ocean grid cell found near this location.")
    st.stop()

pv = data2d.isel(lat=j, lon=i).values
point_val = float(pv) if np.isfinite(pv) else np.nan

point_series = ds[var_name].isel(lat=j, lon=i)
point_series = mask_invalid(point_series)

point_lat = float(lat.values[j])
point_lon = float(lon.values[i])

st.write(f"Nearest valid ocean grid point: lat = {point_lat:.4f}, lon = {point_lon:.4f}")

if np.isfinite(point_val):
    st.write(f"Predicted nitrate on {actual_date}: **{point_val:.4f}**")
else:
    st.write(f"Predicted nitrate on {actual_date}: **NaN / invalid**")

# -------------------------------
# TIME SERIES PLOT
# -------------------------------
st.subheader("Time series at selected grid point")

if np.all(np.isnan(point_series.values)):
    st.warning("All time-series values are invalid/NaN at this selected grid point.")
else:
    fig2, ax2 = plt.subplots(figsize=(11, 4.8), dpi=140)

    ax2.plot(
        ds[time_name].values,
        point_series.values,
        linewidth=1.6
    )

    ax2.axvline(np.datetime64(actual_date), linestyle="--", linewidth=1.0, alpha=0.7)

    if np.isfinite(point_val):
        ax2.scatter(
            np.datetime64(actual_date),
            point_val,
            s=35,
            zorder=3
        )

    ax2.set_title(
        f"Machine-Learned Predicted Nitrate Time Series\nlat={point_lat:.3f}, lon={point_lon:.3f}",
        fontsize=15,
        pad=10,
        weight="bold"
    )
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("Surface nitrate", fontsize=11)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax2.text(
        0.99, 0.02,
        DOI_TEXT,
        transform=ax2.transAxes,
        ha="right", va="bottom",
        fontsize=9, style="italic",
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3)
    )

    st.pyplot(fig2, clear_figure=True)
