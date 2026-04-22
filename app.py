import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm

# -------------------------------
# USER SETTINGS
# -------------------------------
BASE_DIR = Path("compressed")
FILE_PATTERN = "NO3_predictions_RF_paral_{year}_compressed.nc"

APP_TITLE = "Machine-Learned Predicted Nitrate"
APP_SUBTITLE = "Interactive viewer for daily gridded surface nitrate (1998–2018)"
NITRATE_UNIT = "mmol N m$^{-3}$"

PAPER1_TITLE = "Improved understanding of nitrate trends, eutrophication indicators, and risk areas using machine learning"
PAPER1_LINK = "https://doi.org/10.5194/bg-22-3769-2025"

PAPER2_TITLE = "Assimilation of machine-learning-predicted nitrate to improve the quality of phytoplankton forecasting in the shelf-sea environment"
PAPER2_LINK = "https://doi.org/10.1002/qj.70156"

DATA_ZENODO_LINK = "https://doi.org/10.5281/zenodo.19695959"

DOI_TEXT = "Banerjee et al. (10.5194/bg-22-3769-2025; 10.1002/qj.70156)"

FUNDING_TEXT = (
    "This work is carried out at Plymouth Marine Laboratory (PML), Plymouth, UK. "
    "This research was supported by the Horizon Europe project "
    "'New Copernicus Capability for Tropic Ocean Networks' "
    "(NECCTON; grant agreement no. 101081273), and by the UK Natural "
    "Environment Research Council (NERC) National Capability – Science "
    "Single Centre Research programme and the Climate Linked Atlantic "
    "Sector Science (CLASS) project (NE/R015953/1)."
)

VAR_NAME = "prediction"
TIME_NAME = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# -------------------------------
# HEADER / INFO SECTION
# -------------------------------
st.markdown(
    """
This work is carried out at **Plymouth Marine Laboratory (PML), Plymouth, UK**.

**Visualisation created by:** Deep S. Banerjee (dba@pml.ac.uk)
"""
)

st.markdown("### Related papers")
st.markdown(
    f"""
- [{PAPER1_TITLE}]({PAPER1_LINK})
- [{PAPER2_TITLE}]({PAPER2_LINK})
"""
)

st.markdown("### Data access")
st.markdown(
    f"""
- [Download or browse the nitrate dataset on Zenodo]({DATA_ZENODO_LINK})
"""
)

st.info(FUNDING_TEXT)

# -------------------------------
# HELPERS
# -------------------------------
@st.cache_data
def list_available_years():
    years = []
    for year in range(1998, 2019):
        f = BASE_DIR / FILE_PATTERN.format(year=year)
        if f.exists():
            years.append(year)
    return years

def file_for_year(year):
    return BASE_DIR / FILE_PATTERN.format(year=year)

@st.cache_resource
def open_year_dataset(year):
    f = file_for_year(year)
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

    ds = xr.open_dataset(f)

    if VAR_NAME not in ds.variables:
        raise ValueError(f"Variable '{VAR_NAME}' not found in {f}")
    if TIME_NAME not in ds.variables and TIME_NAME not in ds.coords:
        raise ValueError(f"Time variable '{TIME_NAME}' not found in {f}")
    if LAT_NAME not in ds.variables and LAT_NAME not in ds.coords:
        raise ValueError(f"Latitude variable '{LAT_NAME}' not found in {f}")
    if LON_NAME not in ds.variables and LON_NAME not in ds.coords:
        raise ValueError(f"Longitude variable '{LON_NAME}' not found in {f}")

    return ds

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
    cmap.set_bad(color="#d9d9d9")
    return cmap

@st.cache_data
def get_dates_for_year(year):
    ds = open_year_dataset(year)
    dates = np.array(ds[TIME_NAME].values).astype("datetime64[D]")
    return dates

@st.cache_data
def get_map_for_date(year, selected_date):
    ds = open_year_dataset(year)
    data2d = ds[VAR_NAME].sel({TIME_NAME: selected_date}, method="nearest")
    data2d = mask_invalid(data2d).load()  # only one 2D slice
    actual_time = data2d[TIME_NAME].values
    actual_date = np.datetime_as_string(actual_time, unit="D")
    return data2d, actual_date

@st.cache_data
def get_point_series_all_years(j, i):
    all_times = []
    all_vals = []

    years = list_available_years()
    for year in years:
        f = file_for_year(year)
        with xr.open_dataset(f) as ds:
            point = mask_invalid(ds[VAR_NAME].isel(lat=j, lon=i)).load()
            all_times.append(np.array(point[TIME_NAME].values))
            all_vals.append(np.array(point.values, dtype=np.float32))

    times = np.concatenate(all_times)
    vals = np.concatenate(all_vals)
    return times, vals

# -------------------------------
# LOAD YEARS
# -------------------------------
years = list_available_years()
if not years:
    st.error("No NetCDF files found in 'compressed/'.")
    st.stop()

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Controls")

selected_year = st.sidebar.selectbox("Select year", years, index=0)

try:
    ds_year = open_year_dataset(selected_year)
    dates = get_dates_for_year(selected_year)
except Exception as e:
    st.error(f"Error opening dataset for {selected_year}: {e}")
    st.stop()

if len(dates) == 0:
    st.error("No time values found in selected dataset.")
    st.stop()

selected_date = st.sidebar.select_slider(
    "Select date",
    options=dates,
    value=dates[0]
)

lat = ds_year[LAT_NAME]
lon = ds_year[LON_NAME]

# -------------------------------
# EXTRACT MAP DATA
# -------------------------------
data2d, actual_date = get_map_for_date(selected_year, selected_date)

# -------------------------------
# SPATIAL PLOT
# -------------------------------
st.subheader(f"Spatial map — {actual_date}")

if np.all(np.isnan(data2d.values)):
    st.warning("All values are invalid/NaN for this selected date.")
else:
    cmap = make_pretty_cmap()
    data_vals = np.array(data2d.values, dtype=np.float32)
    data_ma = np.ma.masked_invalid(data_vals)

    valid_min = float(np.nanpercentile(data_vals, 2))
    valid_max = float(np.nanpercentile(data_vals, 98))

    if np.isclose(valid_min, valid_max):
        valid_min = float(np.nanmin(data_vals))
        valid_max = float(np.nanmax(data_vals))

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=140)
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

    valid_mask = np.isfinite(data_vals).astype(float)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    ax.contour(
        lon2d,
        lat2d,
        valid_mask,
        levels=[0.5],
        colors="black",
        linewidths=0.7
    )

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label(f"Surface nitrate [{NITRATE_UNIT}]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Machine-Learned Predicted Surface Nitrate", fontsize=14, pad=10, weight="bold")
    ax.set_xlabel("Longitude [°E]", fontsize=10)
    ax.set_ylabel("Latitude [°N]", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, color="0.35")

    ax.text(
        0.99, 0.015,
        DOI_TEXT,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5, style="italic",
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3)
    )

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

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

point_lat = float(lat.values[j])
point_lon = float(lon.values[i])

st.write(f"Nearest valid ocean grid point: lat = {point_lat:.4f}, lon = {point_lon:.4f}")

if np.isfinite(point_val):
    st.write(f"Predicted nitrate on {actual_date}: **{point_val:.4f} {NITRATE_UNIT}**")
else:
    st.write(f"Predicted nitrate on {actual_date}: **NaN / invalid**")

# -------------------------------
# TIME SERIES PLOT
# -------------------------------
st.subheader("Time series at selected grid point")

times_ts, vals_ts = get_point_series_all_years(j, i)

if np.all(np.isnan(vals_ts)):
    st.warning("All time-series values are invalid/NaN at this selected grid point.")
else:
    fig2, ax2 = plt.subplots(figsize=(8.8, 3.5), dpi=140)

    ax2.plot(
        times_ts,
        vals_ts,
        linewidth=1.1
    )

    if np.isfinite(point_val):
        ax2.scatter(
            np.datetime64(actual_date),
            point_val,
            s=28,
            zorder=3
        )

    ax2.set_title(
        f"Machine-Learned Predicted Nitrate Time Series\nlat={point_lat:.3f}, lon={point_lon:.3f}",
        fontsize=13,
        pad=8,
        weight="bold"
    )
    ax2.set_xlabel("Time", fontsize=10)
    ax2.set_ylabel(f"Surface nitrate [{NITRATE_UNIT}]", fontsize=10)
    ax2.tick_params(labelsize=9)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax2.text(
        0.99, 0.02,
        DOI_TEXT,
        transform=ax2.transAxes,
        ha="right", va="bottom",
        fontsize=8.5, style="italic",
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3)
    )

    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)
