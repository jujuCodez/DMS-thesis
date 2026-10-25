#!/usr/bin/env python3
# Fleet Manager Dashboard for Firestore "driver_logs_new"
# Focus: warnings & alerts. Nothing renders until a driver is selected.

import os
from datetime import datetime, date, time, timezone
from dateutil import parser as dtparser

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk
import hashlib

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- Config ----------------
COLLECTION = "driver_logs_new"
SERVICE_ACCOUNT_FILE = "service-account.local.json"
CATEGORIES = ["drowsy", "using_phone", "looking_away", "no_face","yawning"]

st.set_page_config(page_title="Fleet Driver Logs", layout="wide")
st.title("📊 Fleet Driver Logs Dashboard")
st.caption("Source: Firestore collection `driver_logs_new` (per-minute aggregates)")

# ---------------- Firestore (cached) ----------------
@st.cache_resource
def get_db():
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or SERVICE_ACCOUNT_FILE
    if path and not firebase_admin._apps:
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)
    elif not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()

db = get_db()
project_id = firebase_admin.get_app().project_id if firebase_admin._apps else "(unknown)"
st.caption(f"Connected project: `{project_id}`")

# ---------------- Helpers ----------------
def parse_ts_server(v):
    if v is None or pd.isna(v):
        return pd.NaT
    try:
        return v.to_datetime().astimezone(timezone.utc)
    except Exception:
        try:
            return v.astimezone(timezone.utc)
        except Exception:
            return pd.NaT

def parse_iso_utc(s):
    if s is None or pd.isna(s):
        return pd.NaT
    try:
        return dtparser.isoparse(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return pd.NaT

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

# ---------------- Fetch + normalize (cached) ----------------
@st.cache_data(ttl=60*2, show_spinner=True)
def fetch_logs(limit=50000):
    col = db.collection(COLLECTION)
    rows = []
    q = col.order_by("minute_start", direction=firestore.Query.DESCENDING).limit(limit)
    for doc in q.stream():
        d = doc.to_dict() or {}
        d["_id"] = doc.id
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    base = [
        "name", "session",
        "server_ts", "client_ts",
        "minute_start", "minute_end",
        "gps_samples", "avg_latitude", "avg_longitude",
        "counts",
    ]
    ensure_cols(df, base)

    df["server_dt"] = df["server_ts"].apply(parse_ts_server)
    df["client_dt"] = df["client_ts"].apply(parse_iso_utc)
    df["start_dt"]  = df["minute_start"].apply(parse_iso_utc)
    df["end_dt"]    = df["minute_end"].apply(parse_iso_utc)
    df["dt"]        = df["server_dt"].where(df["server_dt"].notna(), df["start_dt"])

    for c in ["gps_samples","avg_latitude","avg_longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # flatten counts
    for cat in CATEGORIES:
        for lvl in ["warn","alert"]:
            df[f"{cat}_{lvl}"] = df["counts"].apply(lambda c: (c or {}).get(cat, {}).get(lvl, 0))

    df["warn_total"]   = df[[f"{c}_warn"  for c in CATEGORIES]].sum(axis=1)
    df["alert_total"]  = df[[f"{c}_alert" for c in CATEGORIES]].sum(axis=1)
    df["events_total"] = df["warn_total"] + df["alert_total"]
    df["alert_rate_%"] = (df["alert_total"] / df["events_total"]).fillna(0) * 100

    df["driver"] = df["name"].fillna("(unknown)")
    df["lat"] = df["avg_latitude"]
    df["lon"] = df["avg_longitude"]
    df["day"] = df["dt"].dt.tz_convert("UTC").dt.date if df["dt"].notna().any() else pd.NaT
    df["hour_utc"] = df["dt"].dt.tz_convert("UTC").dt.hour

    def severity_row(r):
        if r["alert_total"] > 0:
            return "alert"
        if r["warn_total"] > 0:
            return "warn"
        return "none"
    df["severity"] = df.apply(severity_row, axis=1)

    # Colors: red (alerts), yellow (warns), gray (none)
    def color_of(sev):
        if sev == "alert": return [220, 38, 38, 210]   # red
        if sev == "warn":  return [255, 221, 0, 210]   # yellow
        return [107, 114, 128, 160]                    # gray
    df["color"] = df["severity"].map(color_of)

    return df

df = fetch_logs()
if df.empty:
    st.warning("No logs found.")
    st.stop()

# ---------------- Sidebar: filters ----------------
st.sidebar.header("Filters")

# Date range (UTC)
min_dt, max_dt = df["dt"].min(), df["dt"].max()
dflt_start = (min_dt or pd.Timestamp.utcnow()).to_pydatetime().date()
dflt_end   = (max_dt or pd.Timestamp.utcnow()).to_pydatetime().date()
d1, d2 = st.sidebar.date_input("Date range (UTC)", value=(dflt_start, dflt_end))

drivers = sorted(df["driver"].dropna().unique().tolist())
driver_sel = st.sidebar.multiselect("Driver(s)", drivers, default=[])  # NONE selected initially
sessions = sorted(df["session"].dropna().unique().tolist())
session_sel  = st.sidebar.multiselect("Session(s)", sessions, default=[])

severity_sel = st.sidebar.multiselect("Severity", ["alert", "warn", "none"], default=["alert","warn","none"])
gps_only = st.sidebar.checkbox("Has GPS", value=False)
text_search = st.sidebar.text_input("Search session ID")

# Map rendering options
st.sidebar.markdown("### Map")
map_mode = st.sidebar.selectbox(
    "Rendering",
    ["Points (small, jittered)", "Hex bins (cluster)", "Heatmap (density)"],
    index=0,
)
if map_mode == "Points (small, jittered)":
    point_px = st.sidebar.slider("Point size (px)", 4, 24, 10)
else:
    bin_radius_m = st.sidebar.slider("Bin radius (meters)", 30, 250, 90)

# Apply base filters (date, session, severity, gps, search)
mask = pd.Series(True, index=df.index)
if isinstance(d1, date) and isinstance(d2, date):
    start_dt = datetime.combine(d1, time.min, tzinfo=timezone.utc)
    end_dt   = datetime.combine(d2, time.max, tzinfo=timezone.utc)
    mask &= df["dt"].between(start_dt, end_dt, inclusive="both")
if session_sel:
    mask &= df["session"].isin(session_sel)
if severity_sel:
    mask &= df["severity"].isin(severity_sel)
if gps_only:
    mask &= df["lat"].notna() & df["lon"].notna()
if text_search.strip():
    mask &= df["session"].astype(str).str.contains(text_search.strip(), case=False, na=False)

df_base = df[mask].copy()

# Gate everything until at least one driver is selected
if not driver_sel:
    st.info("Select one or more **Driver(s)** in the sidebar to view the dashboard.")
    st.stop()

# Now apply driver filter
dfv = df_base[df_base["driver"].isin(driver_sel)].sort_values("dt").copy()
if dfv.empty:
    st.info("No rows for the selected driver(s).")
    st.stop()

# ---------------- KPIs ----------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Alerts", f"{int(dfv['alert_total'].sum()):,}")
col2.metric("Warnings", f"{int(dfv['warn_total'].sum()):,}")
rate = (dfv["alert_total"].sum() / max(1, dfv["events_total"].sum())) * 100
col3.metric("Alert rate", f"{rate:.1f}%")
col4.metric("Drivers", dfv["driver"].nunique())
col5.metric("Sessions", dfv["session"].nunique())

# ---------------- Category breakdown ----------------
st.subheader("Category breakdown (warn vs alert)")
bars = []
for cat in CATEGORIES:
    bars.append({"category": cat, "level": "warn",  "count": int(dfv[f"{cat}_warn"].sum())})
    bars.append({"category": cat, "level": "alert", "count": int(dfv[f"{cat}_alert"].sum())})
bar_df = pd.DataFrame(bars)
fig_bar = px.bar(bar_df, x="category", y="count", color="level", barmode="group", text_auto=True)
fig_bar.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- Time series (alerts/warns + per-category) ----------------
st.subheader("Trends over time")
plot_opts = st.multiselect(
    "Series",
    ["alert_total","warn_total"] + [f"{c}_alert" for c in CATEGORIES] + [f"{c}_warn" for c in CATEGORIES],
    default=["alert_total","warn_total"] + [f"{c}_alert" for c in CATEGORIES],
)
long = []
ts = dfv.set_index("dt")
for c in plot_opts:
    s = pd.to_numeric(ts[c], errors="coerce").dropna()
    if s.empty:
        continue
    long.extend([{"dt": s.index[i], "value": v, "metric": c} for i, v in enumerate(s.values)])
if long:
    ts_df = pd.DataFrame(long)
    fig_ts = px.line(ts_df, x="dt", y="value", color="metric", render_mode="webgl")
    fig_ts.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No series to plot.")

# ---------------- Hour-of-day heatmap (alerts only) ----------------
st.subheader("Alert intensity by hour (UTC)")
hour_df = (
    dfv.groupby("hour_utc", dropna=True, as_index=False)["alert_total"]
       .sum()
       .sort_values("hour_utc")
)
if not hour_df.empty:
    fig_hour = px.bar(hour_df, x="hour_utc", y="alert_total")
    fig_hour.update_layout(height=260, xaxis_title="Hour of day (UTC)", yaxis_title="Alerts")
    st.plotly_chart(fig_hour, use_container_width=True)
else:
    st.caption("No alerts in current selection.")

# ---------------- Top drivers / sessions ----------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Top drivers (alerts)")
    top_drv = (
        dfv.groupby("driver", dropna=False)
           .agg(alerts=("alert_total","sum"),
                warns=("warn_total","sum"),
                sessions=("session","nunique"),
                minutes=("dt", lambda s: s.dt.floor("T").nunique()))
           .sort_values("alerts", ascending=False)
           .reset_index()
    )
    st.dataframe(top_drv, use_container_width=True)

with colB:
    st.subheader("Top sessions (alerts)")
    top_sess = (
        dfv.groupby("session", dropna=False)
           .agg(driver=("driver","first"),
                alerts=("alert_total","sum"),
                warns=("warn_total","sum"),
                minutes=("dt", lambda s: s.dt.floor("T").nunique()),
                first_dt=("dt","min"),
                last_dt=("dt","max"))
           .sort_values("alerts", ascending=False)
           .reset_index()
    )
    st.dataframe(top_sess, use_container_width=True)

# ---------------- Map: better handling for overlap + clustering ----------------
# ---------------- Map: better handling for overlap + clustering ----------------
st.subheader("Map")

map_df = dfv.dropna(subset=["lat","lon"]).copy()
if map_df.empty:
    st.caption("No GPS points in selection.")
else:
    # A: sensible center
    center_lat = float(map_df["lat"].median())
    center_lon = float(map_df["lon"].median())
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)

    # B: JSON-safe record selection shared by all modes
    # Derive per-category cols from CATEGORIES to avoid typos/mismatches
    cat_cols = [f"{c}_{lvl}" for c in CATEGORIES for lvl in ["alert", "warn"]]
    base_cols = [
    "lat","lon","driver","session","minute_start","minute_end",
    "alert_total","warn_total",
    "drowsy_alert","drowsy_warn",
    "using_phone_alert","using_phone_warn",
    "looking_away_alert","looking_away_warn",
    "no_face_alert","no_face_warn",
    "yawning_alert","yawning_warn",
    "severity","color","events_total","_id"
]

    # Optionally ensure columns exist (defensive)
    ensure_cols(map_df, cat_cols)
    map_df = map_df[base_cols].copy()

    # Pretty labels for tooltip
    PRETTY = {
        "drowsy": "Drowsy",
        "using_phone": "Phone",
        "looking_away": "Away",
        "no_face": "NoFace",
        "yawning": "Yawning",
    }
    cat_bits = " &nbsp; ".join(
        f"<b>{PRETTY[k]}:</b> {{{k}_alert}}/{{{k}_warn}}"
        for k in CATEGORIES if f"{k}_alert" in map_df.columns
    )

    # Tooltip
    tooltip = {
        "html": (
            "<b>{driver}</b><br/>"
            "<b>Session:</b> {session}<br/>"
            "<b>Minute:</b> {minute_start} → {minute_end}<br/>"
            "<b>Alerts:</b> {alert_total} &nbsp;&nbsp; "
            "<b>Warns:</b> {warn_total}<br/>"
            f"{cat_bits}<br/>"
        ),
        "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
    }

    layers = []
    # ... keep the rest of your layer logic unchanged ...


    if map_mode == "Points (small, jittered)":
        # --- Anti-overlap jitter (deterministic) ---
        grp_key = (map_df["lat"].round(5).astype(str) + "," + map_df["lon"].round(5).astype(str))
        counts = grp_key.map(grp_key.value_counts())
        need_jitter = counts > 1

        def jitter_row(r):
            h = int(hashlib.sha1(str(r["_id"]).encode()).hexdigest()[:8], 16)
            ang = (h % 360) * np.pi / 180.0
            # 3–12 meters; slightly larger with more events
            r_m = float(np.clip(3.0 + np.log1p(r["events_total"]) * 3.0, 3.0, 12.0))
            dlat = (r_m * np.sin(ang)) / 111320.0
            dlon = (r_m * np.cos(ang)) / (111320.0 * np.cos(np.radians(r["lat"])))
            return r["lat"] + dlat, r["lon"] + dlon

        lat_j, lon_j = [], []
        for i, r in map_df.iterrows():
            if need_jitter.loc[i]:
                a, b = jitter_row(r)
                lat_j.append(a); lon_j.append(b)
            else:
                lat_j.append(r["lat"]); lon_j.append(r["lon"])

        map_df["lat_j"] = lat_j
        map_df["lon_j"] = lon_j

        # Smaller, pixel-based points that don’t explode with zoom
        map_df["radius_px"] = (point_px * 0.6) + np.sqrt(map_df["events_total"].clip(lower=1)) * 1.8

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df.to_dict("records"),
                get_position=["lon_j","lat_j"],
                get_fill_color="color",
                get_radius="radius_px",
                radius_units="pixels",
                radius_min_pixels=2,
                radius_max_pixels=max(10, point_px),
                pickable=True,
                opacity=0.85,
            )
        )

    elif map_mode == "Hex bins (cluster)":
        # Separate hex layers for reds and yellows
        alerts = map_df[map_df["alert_total"] > 0][["lat","lon","alert_total"]].rename(columns={"alert_total":"w"})
        warns  = map_df[map_df["warn_total"]  > 0][["lat","lon","warn_total"]].rename(columns={"warn_total":"w"})

        RED_RANGE = [
            [255,235,238,40],[255,205,210,90],[255,138,128,140],
            [244,67,54,180],[229,57,53,210],[198,40,40,230]
        ]
        YELLOW_RANGE = [
            [255,253,231,40],[255,249,196,90],[255,241,118,140],
            [255,213,79,180],[255,193,7,210],[255,160,0,230]
        ]

        if not alerts.empty:
            layers.append(
                pdk.Layer(
                    "HexagonLayer",
                    data=alerts.to_dict("records"),
                    get_position=["lon","lat"],
                    get_weight="w",
                    radius=bin_radius_m,
                    elevation_scale=8,
                    elevation_range=[0, 1000],
                    extruded=True,
                    pickable=True,
                    color_range=RED_RANGE,
                )
            )
        if not warns.empty:
            layers.append(
                pdk.Layer(
                    "HexagonLayer",
                    data=warns.to_dict("records"),
                    get_position=["lon","lat"],
                    get_weight="w",
                    radius=bin_radius_m,
                    elevation_scale=8,
                    elevation_range=[0, 800],
                    extruded=True,
                    pickable=True,
                    color_range=YELLOW_RANGE,
                )
            )

    else:  # Heatmap (density)
        heat = map_df.copy()
        heat["weight"] = heat["alert_total"] * 2.0 + heat["warn_total"] * 1.0

        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=heat.to_dict("records"),
                get_position=["lon","lat"],
                get_weight="weight",
                aggregation="SUM",
                radius_pixels=int(bin_radius_m * 0.6),
                opacity=0.85,
            )
        )

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))

# ---------------- Raw table + CSV ----------------
st.subheader("Raw rows (latest first)")
cols_show = [
    "driver","session","minute_start","minute_end","dt",
    "avg_latitude","avg_longitude","gps_samples",
    "warn_total","alert_total","events_total","alert_rate_%",
] + [f"{c}_warn" for c in CATEGORIES] + [f"{c}_alert" for c in CATEGORIES] + ["_id"]
st.dataframe(dfv.sort_values("dt", ascending=False)[cols_show].head(2000), use_container_width=True, height=420)

csv = dfv[cols_show].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="driver_logs_export.csv", mime="text/csv")
