# app_kmz_ocsa_final.py  -- Part 1/3
import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import utm
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval
import tempfile
import os
from io import BytesIO
from pymongo import MongoClient
import certifi
import datetime
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
import gridfs
from bson.objectid import ObjectId
import math
import re

# Ag-Grid
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

# -----------------------
# Config MongoDB Atlas
# -----------------------
MONGO_URI = "mongodb+srv://admin_ro:FE5WRArgoigWXaXl@ocsa.bq4xvyc.mongodb.net/?appName=ocsa"
DB_NAME = "ocsa_database"
COLLECTION_NAME = "OCSA_database_test"   # placemark metadata
KMZ_GRIDFS_COLL = "OCSA_kmz"             # GridFS prefix

st.set_page_config(page_title="OCSA - KMZ Coordinate Utility", layout="wide")

# -----------------------
# Cached MongoDB connection + GridFS
# -----------------------
@st.cache_resource
def get_mongo_resources(uri=MONGO_URI, db_name=DB_NAME, coll_name=COLLECTION_NAME, kmz_gridfs_coll=KMZ_GRIDFS_COLL):
    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client[db_name]
    coll = db[coll_name]
    fs = gridfs.GridFS(db, collection=kmz_gridfs_coll)
    files_coll = db[f"{kmz_gridfs_coll}.files"]
    return client, db, coll, fs, files_coll

try:
    client, db, coll, fs, files_coll = get_mongo_resources()
except Exception as e:
    st.error(f"Gagal koneksi ke MongoDB Atlas: {e}")
    client = db = coll = fs = files_coll = None

# =========================================================
# Utility: Save uploaded file only once to avoid re-read errors
# =========================================================
def save_uploaded_file_once(uploaded_file, key):
    """Save uploaded file once to a temp file and reuse its path."""
    if uploaded_file is None:
        return None
    if key in st.session_state:
        return st.session_state[key]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz")
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    st.session_state[key] = tmp.name
    return tmp.name

# =========================================================
# GridFS helpers for KMZ
# =========================================================
def list_kmz_files_in_db(files_coll):
    """
    Return list of dicts (most recent first) with keys:
     - filename (string)
     - latest_id (ObjectId of latest upload)
     - uploadDate (datetime)
    Only unique filenames (latest version) are returned.
    """
    if files_coll is None:
        return []
    # Aggregate distinct filenames with latest uploadDate
    pipeline = [
        {"$sort": {"uploadDate": -1}},
        {"$group": {"_id": "$filename", "latest_id": {"$first": "$_id"}, "uploadDate": {"$first": "$uploadDate"}}},
        {"$sort": {"uploadDate": -1}}
    ]
    try:
        docs = list(files_coll.aggregate(pipeline))
        return [{"filename": d["_id"], "latest_id": d["latest_id"], "uploadDate": d["uploadDate"]} for d in docs]
    except Exception as e:
        st.warning(f"list_kmz_files_in_db error: {e}")
        return []

def save_kmz_to_gridfs(fs, buf: BytesIO, filename: str, metadata: dict = None):
    """Save BytesIO to GridFS and return inserted_id."""
    if fs is None:
        return None
    buf.seek(0)
    metadata = metadata or {}
    # store with provided filename and metadata
    file_id = fs.put(buf.read(), filename=filename, metadata=metadata, uploadDate=datetime.datetime.utcnow())
    return file_id

def fetch_kmz_from_gridfs(fs, files_coll, filename):
    """
    Fetch the most recent file bytes by filename.
    Returns bytes or None.
    """
    if fs is None or files_coll is None:
        return None
    try:
        doc = files_coll.find_one({"filename": filename}, sort=[("uploadDate", -1)])
        if not doc:
            return None
        grid_out = fs.get(doc["_id"])
        return grid_out.read()
    except Exception as e:
        st.warning(f"fetch_kmz_from_gridfs error: {e}")
        return None

def get_gridfs_file_list(files_coll):
    """Return list of files (filename, uploadDate) for selectbox."""
    items = list_kmz_files_in_db(files_coll)
    return items

# =========================================================
# Parse all placemarks in a KMZ file (path on disk)
# - Improved: This builds a parent map so we can compute full folder path
#   (e.g. "Pipeline/Segment 1") for each Placemark and return that as 'folder'.
# =========================================================
def parse_all_placemarks(kmz_path):
    """
    Returns list of placemarks:
    [{'folder': 'Pipeline/Segment 1', 'name': '0.0', 'lon': 104.1, 'lat': -2.9}, ...]
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    placemarks = []
    try:
        with zipfile.ZipFile(kmz_path, "r") as kmz:
            kml_files = [f for f in kmz.namelist() if f.lower().endswith(".kml")]
            if not kml_files:
                return []
            kml_file = kml_files[0]
            with kmz.open(kml_file, "r") as f:
                tree = ET.parse(f)
                root = tree.getroot()

                # Build parent map for all elements so we can walk up from a Placemark to its ancestor folders
                parent_map = {}
                for parent in root.iter():
                    for child in list(parent):
                        parent_map[child] = parent

                # Find all Placemark elements anywhere
                for pm in root.findall(".//kml:Placemark", ns):
                    # compute folder path by walking ancestors until root, collecting Folder name(s)
                    folder_names = []
                    cur = pm
                    while cur in parent_map:
                        cur = parent_map[cur]
                        # if current element is a Folder, attempt to read its name
                        if cur.tag.endswith('Folder'):
                            name_el = cur.find("kml:name", ns)
                            if name_el is not None and name_el.text and name_el.text.strip():
                                folder_names.append(name_el.text.strip())
                    # build folder path (top-down)
                    folder_path = "/".join(reversed(folder_names)) if folder_names else "Unnamed"

                    name_el = pm.find("kml:name", ns)
                    coord_el = pm.find(".//kml:coordinates", ns)
                    if name_el is None or coord_el is None or not coord_el.text:
                        continue
                    coords_text = coord_el.text.strip().split()
                    # take first coordinate set
                    coords = coords_text[0].split(",")
                    try:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        placemarks.append({
                            "folder": folder_path,
                            "name": name_el.text.strip(),
                            "lon": lon,
                            "lat": lat
                        })
                    except Exception:
                        continue
    except Exception as e:
        st.warning(f"parse_all_placemarks error: {e}")
    return placemarks

# =========================================================
# New utilities for KP interpolation (pipeline)
# =========================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in meters between two lat/lon using haversine."""
    R = 6371000.0  # earth radius meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------
# Extract pipeline KP points from KMZ
# ---------------------------
def get_pipeline_kps_from_kmz(kmz_path):
    """
    Extract placemarks that are part of pipeline (folder path contains 'pipeline')
    and return list of dicts:
    [{'kp': float, 'lat': float, 'lon': float, 'folder': str, 'name': str}, ...]
    - Will try to extract the first numeric token from placemark name (regex).
    - Sorts results by KP value ascending before returning.
    - Ensures duplicates removed (same coords and same KP).
    """
    kps = []
    try:
        pls = parse_all_placemarks(kmz_path)
        seen = set()
        for p in pls:
            folder_path = p.get("folder", "") or ""
            # Accept any folder path that contains 'pipeline' (case-insensitive)
            
            ALLOWED_SEGMENTS = {
                "Seg1_262_240","Seg2_240_222","Seg3_222_195","Seg4_195_174",
                "Seg5_174_157","Seg6_157_139","Seg7_139_129","Seg8_129_102",
                "Seg9_102_77","Seg10_77_52","Seg11_52_27","Seg12_27_8","Seg13_8_3"
            }

            folder_parts = folder_path.split("/")
            if not any(seg in folder_parts for seg in ALLOWED_SEGMENTS):
                continue

            name = (p.get("name") or "").strip()
            if not name:
                continue
            # Extract first numeric token (supports integers or decimals, optional sign)
            m = re.search(r"[-+]?\d*\.?\d+", name)
            if not m:
                # no numeric content -> skip
                continue
            try:
                kp_val = float(m.group(0))
                lat = float(p["lat"])
                lon = float(p["lon"])
                key = (round(lat,8), round(lon,8), float(kp_val))
                if key in seen:
                    continue
                seen.add(key)
                # include folder and original placemark name so caller can report which folder points came from
                kps.append({"kp": kp_val, "lat": lat, "lon": lon, "folder": folder_path, "name": name})
            except Exception:
                continue
    except Exception as e:
        st.warning(f"get_pipeline_kps_from_kmz error: {e}")
    # sort by kp ascending
    return sorted(kps, key=lambda x: x["kp"])

def interpolate_kp_by_distance(input_lat, input_lon, pipeline_points):
    """
    Given input coordinate and pipeline_points as list of {'kp','lat','lon','folder',...},
    find two nearest pipeline points by geographic distance and interpolate KP.
    Returns a dict: {'kp': float, 'p1': point_dict, 'p2': point_dict, 'd1': float, 'd2': float}
    Guarantees returned KP is within [min(kp1,kp2), max(kp1,kp2)].
    """
    result = {"kp": None, "p1": None, "p2": None, "d1": None, "d2": None}
    if not pipeline_points or len(pipeline_points) == 0:
        return result
    if len(pipeline_points) == 1:
        result["kp"] = float(pipeline_points[0]["kp"])
        result["p1"] = pipeline_points[0]
        result["p2"] = pipeline_points[0]
        result["d1"] = 0.0
        result["d2"] = 0.0
        return result

    # compute distances to all points
    pts = []
    for p in pipeline_points:
        d = haversine_distance(input_lat, input_lon, p["lat"], p["lon"])
        pts.append((d, p))
    pts.sort(key=lambda x: x[0])

    # take two nearest distinct points
    d1, p1 = pts[0]
    idx = 1
    while idx < len(pts):
        d2, p2 = pts[idx]
        if not (abs(p1["lat"] - p2["lat"]) < 1e-9 and abs(p1["lon"] - p2["lon"]) < 1e-9):
            break
        idx += 1
    if idx >= len(pts):
        # all pipeline points share same coordinates -> return average KP and same points
        ksum = sum([p["kp"] for (_, p) in pts])
        avg = float(ksum / len(pts))
        result.update({"kp": avg, "p1": pts[0][1], "p2": pts[0][1], "d1": pts[0][0], "d2": pts[0][0]})
        return result
    d2, p2 = pts[idx]

    # if any distance is zero (exact match), return that KP
    if d1 == 0:
        result.update({"kp": float(p1["kp"]), "p1": p1, "p2": p1, "d1": d1, "d2": d1})
        return result
    if d2 == 0:
        result.update({"kp": float(p2["kp"]), "p1": p2, "p2": p2, "d1": d2, "d2": d2})
        return result

    try:
        w1 = 1.0 / d1
        w2 = 1.0 / d2
        kp_est = (p1["kp"] * w1 + p2["kp"] * w2) / (w1 + w2)
    except Exception:
        weight1 = d2 / (d1 + d2) if (d1 + d2) != 0 else 0.5
        weight2 = d1 / (d1 + d2) if (d1 + d2) != 0 else 0.5
        kp_est = p1["kp"] * weight1 + p2["kp"] * weight2

    low = min(p1["kp"], p2["kp"])
    high = max(p1["kp"], p2["kp"])
    if kp_est < low:
        kp_est = low
    if kp_est > high:
        kp_est = high

    result.update({"kp": float(kp_est), "p1": p1, "p2": p2, "d1": d1, "d2": d2})
    return result


# =========================================================
# Add new coordinate to KMZ efficiently
# Returns BytesIO buffer of new KMZ or None if duplicate
# (unchanged from your original logic)
# =========================================================
def add_new_point_to_kmz(kmz_path, folder_name, point_name, lon, lat):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    ET.register_namespace('', ns['kml'])

    # Read existing KML and other files
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_files = [f for f in kmz.namelist() if f.lower().endswith(".kml")]
        if not kml_files:
            raise ValueError("File KML tidak ditemukan dalam KMZ.")
        kml_file = kml_files[0]
        kml_data = kmz.read(kml_file)
        other_files = [(f, kmz.read(f)) for f in kmz.namelist() if f != kml_file]

    root = ET.fromstring(kml_data)
    doc = root.find(".//kml:Document", ns)
    parent = doc if doc is not None else root

    # find folder element
    folder_el = None
    for f in root.findall(".//kml:Folder", ns):
        n = f.find("kml:name", ns)
        if n is not None and n.text and n.text.strip() == folder_name:
            folder_el = f
            break
    if folder_el is None:
        folder_el = ET.SubElement(parent, "{%s}Folder" % ns['kml'])
        ET.SubElement(folder_el, "{%s}name" % ns['kml']).text = folder_name

    # Avoid duplicates by name or exact coordinates
    for pm in folder_el.findall("kml:Placemark", ns):
        n = pm.find("kml:name", ns)
        c = pm.find(".//kml:coordinates", ns)
        if n is not None and n.text and n.text.strip().lower() == point_name.strip().lower():
            return None
        if c is not None and c.text:
            s = c.text.strip().split(",")
            try:
                if abs(float(s[0]) - lon) < 1e-8 and abs(float(s[1]) - lat) < 1e-8:
                    return None
            except:
                continue

    # create placemark
    new_pm = ET.SubElement(folder_el, "{%s}Placemark" % ns['kml'])
    ET.SubElement(new_pm, "{%s}name" % ns['kml']).text = point_name
    point = ET.SubElement(new_pm, "{%s}Point" % ns['kml'])
    ET.SubElement(point, "{%s}coordinates" % ns['kml']).text = f"{lon},{lat},0"

    updated_kml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    out_buf = BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out_kmz:
        out_kmz.writestr(kml_file, updated_kml)
        for name, data in other_files:
            out_kmz.writestr(name, data)
    out_buf.seek(0)
    return out_buf

# =========================================================
# Helper: load all records from MongoDB collection into DataFrame
# =========================================================
def load_records_from_mongo(collection):
    if collection is None:
        return pd.DataFrame()
    docs = list(collection.find({}))
    if len(docs) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # normalize datetime-like fields if present
    for col in ["finding_date", "finish_date", "created_at"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except:
                pass
    return df
# app_kmz_ocsa_final.py  -- Part 3/3 (akhir file)
# =========================================================
# STREAMLIT UI: Mode selector on sidebar
# =========================================================

st.sidebar.image("SIGAP_v2.png", width='stretch')
mode = st.sidebar.radio("Pilih Mode Aplikasi:", ["Coordinate Finder", "Input New Coordinate", "Dashboard Monitoring Event"])

# ------------------------
# MODE 1: Coordinate Finder (with KMZ DB selection filename-only)
# ------------------------
if mode == "Coordinate Finder":
    st.title(" 🔎 Coordinate Finder")

    # KMZ selection from DB (filename only)
    kmz_items = list_kmz_files_in_db(files_coll) if files_coll is not None else []
    kmz_filenames = [item["filename"] for item in kmz_items] if kmz_items else []

    finder_source = st.radio("Source file KMZ:", ("Pilih dari DB (latest version)", "Upload lokal"), index=0, horizontal=True)
    kmz_path = None

    if finder_source == "Pilih dari DB (latest version)":
        if not kmz_filenames:
            st.info("Belum ada file KMZ di database. Anda bisa meng-upload file KMZ lokal.")
            chosen_filename = None
        else:
            chosen_filename = st.selectbox("Pilih file KMZ (DB):", kmz_filenames)
            if chosen_filename:
                kmz_bytes = fetch_kmz_from_gridfs(fs, files_coll, chosen_filename)
                if kmz_bytes:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz")
                    tmp.write(kmz_bytes)
                    tmp.flush()
                    tmp.close()
                    kmz_path = tmp.name
                else:
                    st.error("Gagal mengambil KMZ dari DB.")
                    kmz_path = None
    else:
        uploaded = st.file_uploader("Upload file KMZ (local)", type=["kmz"], key="finder_upload")
        kmz_path = save_uploaded_file_once(uploaded, "finder_kmz") if uploaded else None

    name = st.text_input("Masukkan Nama Titik (Placemark):", key="finder_name")

    # preserve last result
    if "finder_result" not in st.session_state:
        st.session_state["finder_result"] = None

    # Automatically search when both file and name present (no button as requested)
    if kmz_path and name:
        try:
            # find_or_estimate_point may raise
            # using parse_all_placemarks-based function from earlier versions
            def find_or_estimate_point(kmz_path_local, name_local):
                placemarks_local = parse_all_placemarks(kmz_path_local)
                if not placemarks_local:
                    raise ValueError("Tidak ada placemark ditemukan di KMZ.")
                for p in placemarks_local:
                    if p["name"].strip().lower() == name_local.strip().lower():
                        return p
                # try numeric interpolation (as in previous code)
                num_points = []
                for p in placemarks_local:
                    digits = ''.join(c for c in p["name"] if (c.isdigit() or c == '.'))
                    try:
                        num_points.append((float(digits), p))
                    except:
                        continue
                try:
                    target = float(''.join(c for c in name_local if (c.isdigit() or c == '.')))
                except:
                    raise ValueError(f"Titik '{name_local}' tidak ditemukan dan tidak bisa diestimasi.")
                num_points.sort(key=lambda x: x[0])
                for i in range(len(num_points) - 1):
                    x1, p1 = num_points[i]
                    x2, p2 = num_points[i + 1]
                    if x1 <= target <= x2:
                        ratio = (target - x1) / (x2 - x1) if x2 != x1 else 0
                        lat = p1["lat"] + (p2["lat"] - p1["lat"]) * ratio
                        lon = p1["lon"] + (p2["lon"] - p1["lon"]) * ratio
                        return {"folder": p1["folder"], "name": f"{name_local} (Interpolated)", "lat": lat, "lon": lon}
                raise ValueError(f"Titik '{name_local}' tidak ditemukan dan tidak bisa diestimasi.")
            res = find_or_estimate_point(kmz_path, name)
            st.session_state["finder_result"] = {"folder": res.get("folder"), "name": res.get("name"), "lat": res.get("lat"), "lon": res.get("lon")}
        except Exception as e:
            st.session_state["finder_result"] = None
            st.error(str(e))

    # display result if exists
    if st.session_state.get("finder_result"):
        res = st.session_state["finder_result"]
        lat, lon = res["lat"], res["lon"]
        easting, northing, zone_num, zone_letter = utm.from_latlon(lat, lon)

        st.subheader("📍 Hasil Pencarian")
        st.write(f"**Nama Titik:** {res['name']}")
        st.write(f"**Folder:** {res.get('folder','-')}")
        st.write(f"**Latitude:** {lat:.8f}")
        st.write(f"**Longitude:** {lon:.8f}")
        st.write(f"**Easting:** {easting:.2f}")
        st.write(f"**Northing:** {northing:.2f}")
        st.write(f"**UTM Zone:** {zone_num}{zone_letter}")

        # Google Maps URL
        map_url = f"https://www.google.com/maps?q={lat},{lon}"
        st.markdown(f"[🌍 Lihat di Google Maps]({map_url})")

        # WhatsApp share button
        wa_message = (
            f"Lokasi koordinat hasil pencarian:%0A"
            f"Nama: {res['name']}%0A"
            f"Latitude: {lat:.8f}%0A"
            f"Longitude: {lon:.8f}%0A"
            f"Google Maps: https://www.google.com/maps?q={lat},{lon}"
        )
        wa_url = f"https://wa.me/?text={wa_message}"
        st.markdown(
            f'<a href="{wa_url}" target="_blank"><button style="background-color:#25D366;color:white;padding:8px 14px;border:none;border-radius:6px;font-weight:bold">Share via WhatsApp</button></a>',
            unsafe_allow_html=True
        )

        # stable map kept in session_state to reduce blinking
        if "finder_map" not in st.session_state or st.session_state["finder_map"] is None:
            st.session_state["finder_map"] = folium.Map(location=[lat, lon], zoom_start=13)

        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], popup=res["name"], icon=folium.Icon(color="red")).add_to(m)
        st.session_state["finder_map"] = m
        st_folium(st.session_state["finder_map"], width=800, height=500, key="finder_map_widget")
    else:
        st.info("Pilih file KMZ (DB) atau upload lokal, lalu masukkan nama titik untuk hasil pencarian (otomatis).")

# ------------------------
# MODE 2: Input New Coordinate (allow choosing KMZ from DB; save updated KMZ versions to GridFS)
# ------------------------
elif mode == "Input New Coordinate":
    st.title("📍 Input New Coordinate")

    # Allow selecting KMZ from DB or upload local
    kmz_items = list_kmz_files_in_db(files_coll) if files_coll is not None else []
    kmz_filenames = [item["filename"] for item in kmz_items] if kmz_items else []

    kmz_source = st.radio("Source file KMZ untuk di-update:", ("Pilih dari DB", "Upload lokal"), index=0, horizontal=True)
    kmz_path = None
    selected_db_filename = None

    if kmz_source == "Pilih dari DB":
        if not kmz_filenames:
            st.info("Belum ada file KMZ di DB. Silakan upload lokal.")
            kmz_source = "Upload lokal"
        else:
            selected_db_filename = st.selectbox("Pilih file KMZ (DB):", kmz_filenames)
            if selected_db_filename:
                kmz_bytes = fetch_kmz_from_gridfs(fs, files_coll, selected_db_filename)
                if kmz_bytes:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kmz")
                    tmp.write(kmz_bytes)
                    tmp.flush()
                    tmp.close()
                    kmz_path = tmp.name
                else:
                    st.error("Gagal mengambil file KMZ dari DB.")
                    kmz_path = None

    if kmz_source == "Upload lokal":
        uploaded = st.file_uploader("Upload file KMZ (local) untuk di-update", type=["kmz"], key="input_upload")
        kmz_path = save_uploaded_file_once(uploaded, "input_kmz") if uploaded else None
        # if user uploaded, we can set selected_db_filename to the uploaded basename for metadata
        if kmz_path:
            selected_db_filename = os.path.basename(kmz_path)

    if not kmz_path:
        st.info("Silakan pilih atau upload file KMZ untuk di-update.")
        st.stop()

    # parse placemarks once for folder selection and pipeline extraction
    placemarks = parse_all_placemarks(kmz_path)
    folders = sorted(set(p["folder"] for p in placemarks)) if placemarks else []
    folder_choice = st.selectbox("Pilih Folder:", folders + ["➕ Tambah Folder Baru"])

    if folder_choice == "➕ Tambah Folder Baru":
        folder_name = st.text_input("Masukkan Nama Folder Baru:")
    else:
        folder_name = folder_choice

    # basic inputs
    mode_input = st.selectbox("Metode Input Koordinat:", ["Manual Input", "Use Maps"], key="mode_input")

    # ----- NEW ORDER: Longitude -> Latitude (user inputs)
    lon = None
    lat = None
    
    if mode_input == "Manual Input":
        c1, c2 = st.columns(2)
        lon = c1.number_input("Longitude", format="%.8f", key="input_lon")
        lat = c2.number_input("Latitude", format="%.8f", key="input_lat")
    else:
        st.info("Silakan buka Google Maps untuk mendeteksi lokasi Anda lalu salin koordinat (format: lat,lon) dan tempelkan ke field di bawah.")
        # Button to open Google Maps in new tab
        maps_url = "https://www.google.com/maps"
        st.markdown(f'<a href="{maps_url}" target="_blank"><button style="background-color:#0b5fff;color:white;padding:8px 12px;border-radius:6px;border:none;">Open Google Maps</button></a>', unsafe_allow_html=True)

        # Input field where user pastes coordinates from Google Maps (format: lat,lon)
        pasted = st.text_input("Input koordinat saat ini (tempel dari Google Maps, format: lat,lon)", key="input_coords_paste", placeholder="contoh: -2.994567,104.753210")

        # parse pasted coordinates if present
        lat = None
        lon = None
        if pasted:
            # try forgiving parse: accept "lat, lon" or "lat lon"
            import re as _re
            s = pasted.strip()
            s = s.replace(';', ',').replace('|', ',')
            parts = [p.strip() for p in _re.split(r'[,\s]+', s) if p.strip()!='']
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    st.success(f"Koordinat diterima: Lat={lat:.8f}, Lon={lon:.8f}")
                    # store into session_state so subsequent actions use it
                    st.session_state['input_lat'] = lat
                    st.session_state['input_lon'] = lon
                except Exception as _e:
                    st.error("Format koordinat tidak valid. Pastikan format: lat,lon (angka).")
            else:
                st.warning("Format koordinat tidak lengkap. Pastikan menyalin dua angka (lat dan lon).")

        # Show interactive folium map for manual pick as fallback (unchanged)
        center = [placemarks[0]["lat"], placemarks[0]["lon"]] if placemarks else [-2.99, 104.75]
        if 'input_map' not in st.session_state:
            st.session_state['input_map'] = folium.Map(location=center, zoom_start=10)
        m = st.session_state['input_map']
        if 'input_lat' in st.session_state and st.session_state.get('input_lat') is not None and st.session_state.get('input_lon') is not None:
            try:
                m = folium.Map(location=[st.session_state.get('input_lat'), st.session_state.get('input_lon')], zoom_start=15)
                folium.Marker([st.session_state.get('input_lat'), st.session_state.get('input_lon')], popup='Lokasi Anda').add_to(m)
            except Exception:
                m = st.session_state['input_map']
        map_data = st_folium(m, width=900, height=500, key="input_map_widget")
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            st.session_state['input_lat'] = lat
            st.session_state['input_lon'] = lon
            st.success(f"Titik dipilih: Lat={lat:.6f}, Lon={lon:.6f}")

    # =========================================================
    # ----- KP Prediksi now computed only when user clicks "Cari Titik"
    # =========================================================

    # Preload pipeline KP points from KMZ (folder 'pipeline') for interpolation
    pipeline_points = get_pipeline_kps_from_kmz(kmz_path)

    # Ensure session key exists
    if "kp_pred_auto" not in st.session_state:
        st.session_state["kp_pred_auto"] = None

    # Button to trigger KP estimation — placed under lon/lat as requested
    if st.button("Cari Titik"):
        if lat is None or lon is None:
            st.warning("Masukkan Latitude dan Longitude terlebih dahulu.")
        elif not pipeline_points:
            st.error("Folder 'pipeline' tidak memiliki data titik KP.")
        else:
            try:
                kp_res = interpolate_kp_by_distance(lat, lon, pipeline_points)
                kp = kp_res.get("kp") if isinstance(kp_res, dict) else kp_res
                st.session_state["kp_pred_auto"] = kp
                st.session_state["kp_pred_meta"] = kp_res
                st.success("Berhasil menghitung KP Prediksi!")
                try:
                    p1 = kp_res.get("p1")
                    p2 = kp_res.get("p2")
                    if p1 and p2:
                        # show the full folder path of the two reference points (nearer point first)
                        st.info(f"Folder (reference 1): {p1.get('folder','-')}")
                        st.info(f"Folder (reference 2): {p2.get('folder','-')}")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Gagal menghitung KP Prediksi: {e}")

    # Display KP Prediksi as readonly (always updates)
    kp_display_value = st.session_state.get("kp_pred_auto", None)

    if kp_display_value is None:
        st.text_input("KP Prediksi (angka)", value="", disabled=True)
    else:
        st.text_input("KP Prediksi (angka)", value=f"{float(kp_display_value):.3f}", disabled=True)

    # point name input (kept later so user can enter after coords if desired)
    point_name = st.text_input("Nama Titik (Placemark):", key="new_point_name")

    # tambahan parameter (after KP)
    kejadian_options = ["IT", "Leaking", "➕ Tambahkan kejadian baru"]
    kejadian_choice = st.selectbox("Kejadian:", kejadian_options)
    kejadian_text = None
    if kejadian_choice == "➕ Tambahkan kejadian baru":
        kejadian_text = st.text_input("Masukkan kejadian baru:")

    # dates & times
    finding_date = st.date_input("Finding Date", value=datetime.date.today())

    # Finding Time (fixed)
    if "finding_time" not in st.session_state:
        st.session_state["finding_time"] = datetime.datetime.now().time()

    finding_time = st.time_input(
        "Finding Time",
        value=st.session_state["finding_time"],
        key="finding_time_input"
    )
    st.session_state["finding_time"] = finding_time

    location_text = st.text_input("Location:")
    pipeline_class = st.text_input("Pipeline Classification:")

    finish_date = st.date_input("Finish Date", value=datetime.date.today())

    # Finish Time (fixed)
    if "finish_time" not in st.session_state:
        st.session_state["finish_time"] = datetime.datetime.now().time()

    finish_time = st.time_input(
        "Finish Time",
        value=st.session_state["finish_time"],
        key="finish_time_input"
    )
    st.session_state["finish_time"] = finish_time

    est_oil = st.number_input("Estimated Total Oil Recovery (Liter)", format="%.2f", step=0.01)
    emergency_action = st.text_area("Emergency Action:")
    status = st.selectbox("Status:", ["In Progress", "Close"])

    if st.button("📤 Tambahkan Titik"):
        # Basic validation
        if not folder_name or not point_name or lat is None or lon is None:
            st.warning("Lengkapi semua input dasar: Folder, Nama Titik, dan koordinat.")
        else:
            try:
                # 1) Update KMZ (local)
                buf = add_new_point_to_kmz(kmz_path, folder_name, point_name, float(lon), float(lat))
                if buf is None:
                    st.warning("Titik dengan nama atau koordinat yang sama sudah ada di folder tersebut. Tidak ada perubahan pada KMZ.")
                else:
                    # allow download
                    st.download_button("⬇️ Download file_updated.kmz", data=buf, file_name="file_updated.kmz",
                                       mime="application/vnd.google-earth.kmz")

                    # 1b) Save KMZ result to GridFS as a new version
                    if fs is not None and files_coll is not None:
                        try:
                            # Always create updated filename using requested format
                            # Format: kmz_updated_YYYYMMDDTHHMMSS.kmz
                            base_filename = f"kmz_updated_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.kmz"
                            save_filename = base_filename

                            buf.seek(0)
                            metadata = {
                                "source": "streamlit_app",
                                "uploader": "streamlit_user",
                                "created_at": datetime.datetime.utcnow().isoformat(),
                                "original_basename": os.path.basename(kmz_path)
                            }
                            file_id = save_kmz_to_gridfs(fs, buf, save_filename, metadata=metadata)
                            if file_id:
                                st.success(f"KMZ hasil update tersimpan ke GridFS (id: {file_id}). (filename: {save_filename})")
                            else:
                                st.warning("Gagal menyimpan KMZ ke GridFS.")
                        except Exception as e:
                            st.error(f"Gagal menyimpan KMZ ke GridFS: {e}")
                    else:
                        st.warning("GridFS tidak tersedia, file KMZ tidak disimpan ke DB.")

                # 2) Save record to MongoDB Atlas (placemark metadata)
                if coll is None:
                    st.error("Tidak terhubung ke MongoDB Atlas. Metadata tidak disimpan.")
                else:
                    kejadian_value = kejadian_text if kejadian_choice == "➕ Tambahkan kejadian baru" else kejadian_choice
                    # read kp_pred from session (readonly)
                    kp_record_val = st.session_state.get("kp_pred_auto", None)
                    # prepare record - serialize dates/times to ISO strings
                    record = {
                        "folder": folder_name,
                        "point_name": point_name,
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "kejadian": kejadian_value,
                        "kp_prediksi": float(kp_record_val) if kp_record_val is not None else None,
                        "finding_date": finding_date.isoformat() if isinstance(finding_date, datetime.date) else str(finding_date),
                        "finding_time": st.session_state["finding_time"].strftime("%H:%M:%S") if isinstance(st.session_state["finding_time"], datetime.time) else str(st.session_state["finding_time"]),
                        "location": location_text,
                        "pipeline_classification": pipeline_class,
                        "finish_date": finish_date.isoformat() if isinstance(finish_date, datetime.date) else str(finish_date),
                        "finish_time": st.session_state["finish_time"].strftime("%H:%M:%S") if isinstance(st.session_state["finish_time"], datetime.time) else str(st.session_state["finish_time"]),
                        "estimated_total_oil_recovery_liter": float(est_oil) if est_oil is not None else None,
                        "emergency_action": emergency_action,
                        "status": status,
                        "method_input": mode_input,
                        "kmz_source_filename": selected_db_filename if selected_db_filename else os.path.basename(kmz_path),
                        "created_at": datetime.datetime.utcnow().isoformat()
                    }
                    try:
                        coll.insert_one(record)
                        st.success("Titik dan informasi berhasil disimpan ke MongoDB Atlas (collection).")
                    except Exception as e:
                        st.error(f"Gagal menyimpan metadata ke MongoDB Atlas: {e}")

            except Exception as e:
                st.error(f"Gagal menambahkan titik ke KMZ: {e}")

# ------------------------
# MODE 3: Dashboard Monitoring Event
# ------------------------
else:
    st.title("📊 Dashboard Monitoring Event")
    if coll is None:
        st.error("Tidak dapat terhubung ke MongoDB Atlas.")
        st.stop()

    # Load records
    df = load_records_from_mongo(coll)
    if df.empty:
        st.warning("Belum ada data dalam database.")
        st.stop()

    # Ensure latitude & longitude present as floats
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SUMMARY CARDS
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Kejadian", len(df))
    with c2:
        st.metric("In Progress", int((df.get("status") == "In Progress").sum()))
    with c3:
        st.metric("Close", int((df.get("status") == "Close").sum()))
    with c4:
        if "kejadian" in df.columns and not df["kejadian"].isna().all():
            st.metric("Kejadian Terbanyak", df["kejadian"].mode().iat[0])
        else:
            st.metric("Kejadian Terbanyak", "-")

    st.markdown("---")
    
    # CHART: jumlah kejadian per bulan (with labels on top)
    st.subheader("📈 Jumlah Kejadian per Bulan")
    if "finding_date" in df.columns:
        df["finding_month"] = pd.to_datetime(df["finding_date"], errors="coerce").dt.to_period('M').astype(str)
        monthly = df.groupby("finding_month").size().reset_index(name="count")
        if not monthly.empty:
            fig = px.bar(monthly, x="finding_month", y="count",
                         title="Jumlah Kejadian per Bulan",
                         labels={"finding_month":"Bulan","count":"Jumlah"},
                         text="count")
            fig.update_traces(textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle= -45)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Tidak ada data tanggal valid untuk chart bulanan.")
    else:
        st.info("Kolom 'finding_date' tidak tersedia.")

    st.markdown("---")

    # MAP: show all events with a large leaflet pane using satellite tiles
    st.subheader("🗺️ Peta Lokasi Kejadian (Satellite View)")
    # compute center
    lat_mean = df["latitude"].mean() if "latitude" in df.columns else -2.99
    lon_mean = df["longitude"].mean() if "longitude" in df.columns else 104.75

    # create map with no default tiles then add ESRI satellite for satellite view
    map_events = folium.Map(location=[lat_mean, lon_mean], zoom_start=10, tiles=None)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri.WorldImagery', name='Esri.WorldImagery', overlay=False, control=False).add_to(map_events)

    # color mapping by status
    def status_color(s):
        if s == "In Progress":
            return "red"
        elif s == "Close":
            return "yellow"
        else:
            return "blue"

    # add markers
    for _, r in df.iterrows():
        if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
            popup_html = (
                f"<b>{r.get('point_name','-')}</b><br>"
                f"Folder: {r.get('folder','-')}<br>"
                f"Kejadian: {r.get('kejadian','-')}<br>"
                f"Finding: {r.get('finding_date','-')} {r.get('finding_time','-')}<br>"
                f"KP: {r.get('kp_prediksi','-')}<br>"
                f"Status: {r.get('status','-')}"
            )
            folium.CircleMarker(
                [float(r["latitude"]), float(r["longitude"])],
                radius=6,
                color=status_color(r.get("status")),
                fill=True,
                fill_opacity=0.8,
                popup=popup_html
            ).add_to(map_events)

    st_folium(map_events, width=1200, height=650, key="dashboard_map")

    st.markdown("---")

    # TABLE: Ag-Grid interactive table with filters and top-N selection
    st.subheader("📋 Tabel Kejadian (Ag-Grid)")
    # Default sort/drop columns for display: remove _id if present
    display_df = df.copy()
    if "_id" in display_df.columns:
        display_df["_id"] = display_df["_id"].astype(str)

    # choose how many rows to show
    row_options = [10, 50, 100, "All"]
    default_rows = 10
    show_rows = st.selectbox("Tampilkan baris:", row_options, index=0)
    if show_rows == "All":
        display_show_df = display_df
    else:
        display_show_df = display_df.head(int(show_rows))

    # configure Ag-Grid
    gb = GridOptionsBuilder.from_dataframe(display_show_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=int(default_rows))
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    AgGrid(
        display_show_df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        height=400,
        fit_columns_on_grid_load=True,
        reload_data=True
    )

    st.markdown("---")

    # ANALISIS POLA: Clustering hotspot (KMeans)
    st.subheader("🤖 Analisis Pola Kejadian (KMeans Hotspot Preview)")
    coords = df.dropna(subset=["latitude","longitude"])[["latitude","longitude"]]
    if len(coords) < 3:
        st.info("Data tidak cukup untuk clustering (minimal 3 titik).")
    else:
        k_opt = st.slider("Jumlah cluster (k)", min_value=2, max_value=8, value=3)
        kmeans = KMeans(n_clusters=k_opt, n_init='auto', random_state=0)
        labels = kmeans.fit_predict(coords)
        coords = coords.copy()
        coords["cluster"] = labels
        st.write("Jumlah masing-masing cluster:")
        st.write(pd.Series(labels).value_counts().sort_index())

        # map clusters
        cluster_map = folium.Map(location=[lat_mean, lon_mean], zoom_start=10)
        colors = ["red", "blue", "green", "purple", "orange", "cadetblue", "darkgreen", "darkred"]
        for idx, row in coords.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=6,
                color=colors[int(row["cluster"]) % len(colors)],
                fill=True,
                fill_opacity=0.7,
                popup=f"Cluster {int(row['cluster'])}"
            ).add_to(cluster_map)
        st_folium(cluster_map, width=1200, height=600, key="cluster_map")

    st.success("Dashboard siap. Gunakan filter & export sesuai kebutuhan.")