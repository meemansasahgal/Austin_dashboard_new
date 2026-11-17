from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CORRIDORS = {
    "South Congress Avenue (SoCo)": {"short": "SoCo", "center": (30.2474, -97.7495), "search_radius_m": 1800, "buffer_m": 120, "match": ["south congress avenue", "south congress"], "color": "#E76F51", "description": "Real-world profile: boutique and specialty retail, apparel and gifts, galleries, hotels, live music and visitor-oriented dining. Food is one part of a wider shopping and entertainment district."},
    "South Lamar Boulevard": {"short": "South Lamar", "center": (30.2500, -97.7810), "search_radius_m": 2200, "buffer_m": 150, "match": ["south lamar boulevard", "south lamar blvd", "south lamar"], "color": "#2A9D8F", "description": "Real-world profile: a mixed commercial strip with restaurants and bars, entertainment, retail, auto-oriented services, offices and mixed-use developments; the mix varies by section."},
    "East Riverside Drive": {"short": "East Riverside", "center": (30.2370, -97.7220), "search_radius_m": 2200, "buffer_m": 150, "match": ["east riverside drive", "e riverside drive", "riverside drive", "east riverside"], "color": "#457B9D", "description": "Real-world profile: mixed-use redevelopment alongside everyday retail and services, grocery/convenience uses, apartments, and a substantial food offering. The exact mix varies along the corridor."},
    "Burnet Road": {"short": "Burnet Road", "center": (30.3450, -97.7390), "search_radius_m": 2200, "buffer_m": 125, "match": ["burnet road", "burnet rd"], "color": "#D6A72C", "description": "Real-world profile: neighborhood-serving retail and shopping centers, auto sales/repair, personal and professional services, offices and restaurants—not just dining."},
    "North Lamar Boulevard": {"short": "North Lamar", "center": (30.3350, -97.7270), "search_radius_m": 2500, "buffer_m": 150, "match": ["north lamar boulevard", "n lamar boulevard", "north lamar blvd", "north lamar"], "color": "#8E6CBA", "description": "Real-world profile: a broad, culturally diverse commercial corridor with international groceries and markets, specialty retail, beauty and financial services, travel agencies and restaurants."},
    "Guadalupe Street (The Drag)": {"short": "Guadalupe / The Drag", "center": (30.2860, -97.7420), "search_radius_m": 1400, "buffer_m": 100, "match": ["guadalupe street", "guadalupe st"], "color": "#D45087", "description": "Real-world profile: a student-facing university strip with bookstores/campus retail, convenience and apparel shops, services, music/culture and restaurants. Dining is visible but not the only defining use."},
    "Congress Avenue (Downtown)": {"short": "Congress Ave", "center": (30.2672, -97.7428), "search_radius_m": 1800, "buffer_m": 150, "match": ["south congress avenue", "congress avenue", "congress ave"], "color": "#3A86FF", "description": "Downtown Congress Avenue commercial corridor; road-name matching should be checked against the saved OSM road geometry."},
    "Manor Road": {"short": "Manor Road", "center": (30.2868, -97.7150), "search_radius_m": 2200, "buffer_m": 150, "match": ["manor road", "manor rd"], "color": "#8338EC", "description": "East Austin neighborhood commercial corridor with local retail, food, services and community destinations."},
    "Airport Boulevard": {"short": "Airport Blvd", "center": (30.3070, -97.7070), "search_radius_m": 2500, "buffer_m": 150, "match": ["airport boulevard", "airport blvd"], "color": "#FB5607", "description": "North-central Austin commercial corridor with retail, services, food and transport-oriented businesses."},
    "Research Boulevard (US 183)": {"short": "Research Blvd", "center": (30.3900, -97.7400), "search_radius_m": 3000, "buffer_m": 150, "match": ["research boulevard", "research blvd", "us 183", "us highway 183"], "color": "#06A77D", "description": "US 183 / Research Boulevard commercial section; exact road-name matching and corridor extent should be validated after OSM collection."},
    "Cameron Road": {"short": "Cameron Road", "center": (30.3270, -97.7040), "search_radius_m": 2400, "buffer_m": 150, "match": ["cameron road", "cameron rd"], "color": "#BC6C25", "description": "Cameron Road commercial corridor with neighborhood services, retail, food and health-related destinations."},
}


# Research-backed corridor identity themes (relative emphasis, not business counts).
# These scores describe the corridor's recognizable commercial role and are kept
# separate from the live OSM inventory so the app never presents them as census data.
CORRIDOR_LEADERS = {
    "South Congress Avenue (SoCo)": {
        "Leading activity": "Retail & shopping",
        "Other strong activities": "Accommodation & tourism",
    },
    "South Lamar Boulevard": {
        "Leading activity": "Food & drink and Recreation & fitness (tie)",
        "Other strong activities": "Retail & shopping, Professional & financial services",
    },
    "East Riverside Drive": {
        "Leading activity": "Food & drink",
        "Other strong activities": "Retail & shopping",
    },
    "Burnet Road": {
        "Leading activity": "Retail & shopping",
        "Other strong activities": "Transport & automotive, Professional & financial services",
    },
    "North Lamar Boulevard": {
        "Leading activity": "Retail & shopping",
        "Other strong activities": "Community & public services, Professional & financial services",
    },
    "Guadalupe Street (The Drag)": {
        "Leading activity": "Food & drink",
        "Other strong activities": "Education & culture, Retail & shopping",
    },
}


CORRIDOR_PROFILE = {
    "South Congress Avenue (SoCo)": {
        "Retail & shopping": 5, "Accommodation & tourism": 4,
        "Recreation & fitness": 3, "Food & drink": 3,
        "Education & culture": 3, "Professional & financial services": 1,
    },
    "South Lamar Boulevard": {
        "Recreation & fitness": 4, "Food & drink": 4,
        "Retail & shopping": 3, "Professional & financial services": 3,
        "Transport & automotive": 2, "Education & culture": 2,
    },
    "East Riverside Drive": {
        "Food & drink": 5, "Retail & shopping": 4,
        "Professional & financial services": 4, "Transport & automotive": 3,
        "Recreation & fitness": 3, "Education & culture": 3,
    },
    "Burnet Road": {
        "Retail & shopping": 5, "Transport & automotive": 4,
        "Professional & financial services": 4, "Health & care": 3,
        "Food & drink": 3, "Recreation & fitness": 2,
    },
    "North Lamar Boulevard": {
        "Retail & shopping": 5, "Food & drink": 3, "Recreation & fitness": 3,
        "Community & public services": 4, "Professional & financial services": 4,
        "Health & care": 3, "Education & culture": 3,
    },
    "Guadalupe Street (The Drag)": {
        "Food & drink": 5, "Education & culture": 4,
        "Retail & shopping": 4, "Recreation & fitness": 3,
        "Professional & financial services": 2, "Accommodation & tourism": 1,
    },
}

st.set_page_config(page_title="Austin Corridor Intelligence", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#17352f; --muted:#687b76; --accent:#1f6a56; }
/* Soft, polished canvas: styling only; all app logic and map layers remain unchanged. */
.stApp {background:linear-gradient(145deg,#f3f7f5 0%,#f7f5ef 52%,#eef5f4 100%); color:#203a35;}
.block-container {padding-top:1.35rem; padding-bottom:2rem; max-width:1500px;}
[data-testid="stHeader"] {background:rgba(243,247,245,.88);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#173c34 0%,#102d28 100%);border-right:1px solid #28594e;}
[data-testid="stSidebar"] * {color:#f4f8f5 !important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#d7e7e0 !important;}
.hero {padding:1.45rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#173c34 0%,#286a58 65%,#438c76 100%);color:white;margin-bottom:1.15rem;box-shadow:0 8px 24px rgba(23,60,52,.13);border:1px solid rgba(255,255,255,.16);}
.hero h1 {color:white;font-size:2rem;margin:0 0 .3rem 0;}
.hero p {color:#dcebe4;margin:0;}
.metric-note {color:#6b7d77;font-size:.8rem;}
.section-title {font-size:1.2rem;font-weight:750;color:#173c34;margin:.65rem 0 .8rem 0;}
.small-card,[data-testid="stMetric"] {padding:1rem;border:1px solid #dce8e2;border-radius:14px;background:rgba(255,255,255,.88);box-shadow:0 3px 12px rgba(27,67,57,.045);}
[data-testid="stMetricLabel"] {color:#58766c !important;}
[data-testid="stMetricValue"] {color:#173c34 !important;}
[data-testid="stTabs"] [role="tab"] {border-radius:10px 10px 0 0;color:#42675b;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color:#145c49;border-bottom-color:#23866a;font-weight:700;}
.stButton > button, .stDownloadButton > button {border-radius:10px;border:1px solid #b8d2c7;background:linear-gradient(180deg,#ffffff,#f1f8f4);color:#174b3e;font-weight:650;transition:all .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:#32876d;background:#e4f4ec;color:#123f33;}
[data-testid="stDataFrame"], [data-testid="stTable"] {border:1px solid #dce8e2;border-radius:12px;overflow:hidden;}
[data-testid="stExpander"] {border:1px solid #dce8e2;border-radius:12px;background:rgba(255,255,255,.68);}
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {border-radius:10px;}
hr {border-color:#dbe7e1;}
</style>
""", unsafe_allow_html=True)

TAGS = {
    "shop": True,
    "amenity": True,
    "office": True,
    "tourism": True,
    "leisure": True,
    "craft": True,
    "healthcare": True,
    "club": True,
    "public_transport": ["station", "stop_position"],
    "healthcare": True,
    "craft": True,
    "building": ["commercial", "retail", "office", "university", "school"],
}

EARTH_RADIUS_M = 6_371_000


def category_for(row):
    """Keep the most specific OSM tag available; do not let a generic tag mask a specific one."""
    priority = ["shop", "amenity", "healthcare", "office", "craft", "tourism", "leisure", "club", "public_transport"]
    for field in priority:
        value = row.get(field)
        if pd.notna(value) and str(value).strip() and str(value).lower() not in {"nan", "none", "false"}:
            if value is True or str(value).lower() in {"yes", "true"}:
                continue
            return str(value).replace("_", " ").strip().lower()
    return "other"


def major_activity(category, row=None):
    """Classify from raw OSM tags first; use category/name only as fallbacks.

    OSM stores a business' type in different keys (shop, amenity, office,
    healthcare, leisure, etc.). Looking only at the single `category` value
    can overstate food businesses and hide the corridor's broader mix.
    """
    row = row if row is not None else {}

    def clean(value):
        if value is None or pd.isna(value):
            return ""
        value = str(value).strip().lower().replace("_", " ")
        return "" if value in {"", "nan", "none", "false", "no", "unknown"} else value

    # Newly collected files store raw tags with an osm_ prefix; older files
    # may contain the original unprefixed OSM columns.
    tags = {}
    for key in ["shop", "amenity", "healthcare", "office", "craft", "tourism",
                "leisure", "club", "public_transport", "building"]:
        value = clean(row.get("osm_" + key, row.get(key, "")))
        if value:
            tags[key] = value

    food = {"restaurant", "fast food", "cafe", "coffee shop", "bar", "pub",
            "biergarten", "ice cream", "food court", "food", "bakery",
            "confectionery", "deli", "tea", "juice bar", "food truck", "marketplace"}
    retail = {"clothes", "shoes", "jewelry", "gift", "books", "bookshop",
              "department store", "supermarket", "convenience", "greengrocer",
              "butcher", "mall", "general", "fashion", "electronics",
              "mobile phone", "bicycle", "car", "florist", "chemist",
              "cosmetics", "hardware", "furniture", "sports", "stationery",
              "toys", "art", "video", "second hand", "variety store", "pet",
              "optician", "tobacco", "e-cigarette", "alcohol", "diy", "trade",
              "department store", "outdoor", "musical instrument", "fabric",
              "bag", "beauty", "hairdresser", "bookmaker", "copy shop"}
    health = {"hospital", "clinic", "doctors", "dentist", "veterinary", "pharmacy",
              "nursing home", "social facility", "physiotherapist", "optometrist",
              "medical supply", "blood donation", "health centre"}
    professional = {"bank", "atm", "bureau de change", "insurance", "lawyer",
                    "estate agent", "accountant", "coworking", "financial",
                    "government", "post office", "employment agency", "consulting",
                    "advertising agency", "it", "company", "research", "educational institution"}
    education = {"school", "university", "college", "kindergarten", "library",
                 "arts centre", "community centre", "museum", "gallery", "theatre",
                 "cinema", "music school", "language school", "training", "driving school"}
    recreation = {"fitness centre", "sports centre", "gym", "park", "playground",
                  "stadium", "pitch", "swimming pool", "dance", "nightclub",
                  "bowling alley", "escape game", "sports", "adult gaming centre"}
    transport = {"fuel", "charging station", "parking", "car wash", "car repair",
                 "bicycle repair station", "bus station", "taxi", "vehicle inspection",
                 "motorcycle", "car rental", "bicycle rental", "bus stop", "station"}
    public = {"place of worship", "police", "fire station", "courthouse", "public building",
              "shelter", "toilets", "recycling", "waste disposal", "community centre"}

    # Use the semantic OSM key first. A tagged shop remains retail even if its
    # name happens to contain a food-related word; explicit amenity=restaurant
    # is food only when no more specific shop tag exists.
    shop = tags.get("shop", "")
    if shop:
        if shop in {"car repair", "tyres", "car parts", "motorcycle", "fuel"}:
            return "Transport & automotive"
        if shop in {"beauty", "hairdresser", "massage", "tattoo", "optician", "hearing aids"}:
            return "Health & care"
        if shop in food:
            return "Food & drink"
        return "Retail & shopping"

    amenity = tags.get("amenity", "")
    if amenity in food:
        return "Food & drink"
    if amenity in health:
        return "Health & care"
    if amenity in professional:
        return "Professional & financial services"
    if amenity in education:
        return "Education & culture"
    if amenity in recreation:
        return "Recreation & fitness"
    if amenity in transport:
        return "Transport & automotive"
    if amenity in public:
        return "Community & public services"
    if tags.get("healthcare"):
        return "Health & care"
    if tags.get("office"):
        return "Professional & financial services"
    if tags.get("craft"):
        return "Retail & shopping"
    if tags.get("tourism"):
        return "Accommodation & tourism"
    if tags.get("leisure") or tags.get("club"):
        leisure = tags.get("leisure", tags.get("club", ""))
        if leisure in {"park", "playground", "sports centre", "fitness centre", "pitch", "swimming pool", "stadium"}:
            return "Recreation & fitness"
        return "Recreation & fitness"
    if tags.get("public_transport"):
        return "Transport & automotive"

    # Legacy CSVs may lack raw OSM tags. Use conservative business-name signals
    # before falling back to their single old category field; never infer a whole
    # corridor's leading activity from its profile description.
    name = clean(row.get("name", ""))
    name_rules = [
        (["bookstore", "book shop", "bookshop", "university co-op", "clothing", "boutique", "outfitters", "vintage", "shoe store", "gift shop", "record store", "comic shop", "grocery", "market", "supermarket", "convenience store", "liquor store", "hardware", "furniture", "electronics", "mobile", "florist", "pet store", "thrift", "consignment", "jewelry", "salon", "barber", "beauty supply"], "Retail & shopping"),
        (["fitness", "gym", "yoga", "pilates", "climbing", "dance studio", "sports club"], "Recreation & fitness"),
        (["pharmacy", "medical", "dental", "dentist", "clinic", "urgent care", "health center", "optometry", "veterinary"], "Health & care"),
        (["bank", "credit union", "insurance", "law office", "attorney", "accounting", "realty", "real estate", "travel agency", "staffing", "coworking"], "Professional & financial services"),
        (["university", "college", "library", "museum", "gallery", "theatre", "theater", "cinema", "arts center", "arts centre", "music school"], "Education & culture"),
        (["hotel", "motel", "hostel", "inn ", "guest house", "visitor center", "tour"], "Accommodation & tourism"),
        (["auto repair", "car repair", "tire shop", "tyre shop", "gas station", "car wash", "car dealership", "auto sales", "parking", "bike repair"], "Transport & automotive"),
    ]
    for phrases, activity in name_rules:
        if any(x in name for x in phrases):
            return activity

    c = clean(category)
    if c in food:
        return "Food & drink"
    if c in retail:
        return "Retail & shopping"
    if c in health:
        return "Health & care"
    if c in professional or c in {"office", "laundry", "dry cleaning", "tailor", "repair", "storage", "shipping", "travel agency"}:
        return "Professional & financial services"
    if c in education:
        return "Education & culture"
    if c in {"hotel", "motel", "hostel", "guest house", "attraction", "information", "viewpoint", "artwork"}:
        return "Accommodation & tourism"
    if c in recreation or c in {"leisure", "club"}:
        return "Recreation & fitness"
    if c in transport:
        return "Transport & automotive"
    if c in public:
        return "Community & public services"
    return "Miscellaneous Retail & Services"


def enrich_categories(df):
    """Add major_activity without overwriting the source category or raw OSM tags."""
    if df.empty:
        return df
    if "category" not in df.columns:
        df["category"] = "other"
    df["category"] = df["category"].fillna("other").astype(str)
    df["major_activity"] = [major_activity(cat, row) for cat, (_, row) in zip(df["category"], df.iterrows())]
    return df


def load_soco():
    # Load the same canonical CSV path as every other corridor. Do not attach old cluster labels
    # by row position: that can silently mismatch rows when the source CSV has changed.
    path = DATA_DIR / "soco_corridor_businesses.csv"
    if not path.exists():
        return pd.DataFrame(columns=["name", "category", "latitude", "longitude", "corridor"])
    df = pd.read_csv(path)
    for col in ["name", "category", "latitude", "longitude"]:
        if col not in df.columns:
            df[col] = "" if col in ["name", "category"] else np.nan
    df["corridor"] = "South Congress Avenue (SoCo)"
    df["name"] = df["name"].fillna("")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    return enrich_categories(df)

def corridor_csv(name):
    safe = {"South Congress Avenue (SoCo)": "soco_corridor_businesses.csv"}.get(name, "corridor_" + CORRIDORS[name]["short"].lower().replace(" ", "_").replace("/", "_") + ".csv")
    return DATA_DIR / safe


def fetch_corridor(name, radius_m=3000, buffer_m=100, eps_m=100, min_samples=5):
    """Download OSM road/business data, select named road segments and filter points to a 100m buffer."""
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise RuntimeError("Missing packages. Install requirements.txt first.") from e

    info = CORRIDORS[name]
    center = info["center"]
    # Use a smaller query area for Guadalupe to reduce server load.
    query_radius = radius_m
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 60

    # Try alternative Overpass servers if one is unavailable.
    endpoints = [
        "https://overpass.kumi.systems/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.private.coffee/api",
        "https://overpass-api.de/api",
    ]

    graph = None
    errors = []

    for endpoint in endpoints:
        try:
            ox.settings.overpass_url = endpoint
            print(f"Trying Overpass endpoint: {endpoint}")

            graph = ox.graph_from_point(
                center,
                dist=query_radius,
                network_type="drive",
                simplify=True
            )

            if graph.number_of_edges() > 0:
                break

        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
            print(f"Endpoint failed: {exc}")
            graph = None

    if graph is None or graph.number_of_edges() == 0:
        detail = "\n".join(errors[-4:])
        raise RuntimeError(
            "OpenStreetMap road data could not be reached from the available Overpass servers. "
            "This is a network/server availability issue, not a corridor-name setting. "
            "Please retry when the public API is responding.\n\n" + detail
        )

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index(drop=True)
    if "name" not in edges.columns:
        raise RuntimeError(f"OSM did not return road names for {name}. Try again or adjust the corridor section.")
    needles = [s.lower() for s in info["match"]]
    def matches(value):
        vals = value if isinstance(value, list) else [value]
        return any(n in str(v).lower() for v in vals for n in needles if v is not None)
    selected = edges[edges["name"].apply(matches)].copy()
    if selected.empty:
        raise RuntimeError(f"No road segments matching {name!r} were found in the {radius_m}m search area. Try increasing the search radius or check OSM road naming.")
    selected = selected[selected.geometry.notna()]
    if selected.empty:
        raise RuntimeError("Matched road segments did not have usable geometry.")
    crs_projected = selected.estimate_utm_crs()
    selected_m = selected.to_crs(crs_projected)
    # Keep the selected connected section nearest the configured center to avoid unrelated road fragments.
    center_pt = gpd.GeoSeries([Point(center[1], center[0])], crs="EPSG:4326").to_crs(crs_projected).iloc[0]
    selected_m["_distance"] = selected_m.geometry.distance(center_pt)
    selected_m = selected_m[selected_m["_distance"] <= radius_m * 0.8].copy()
    if selected_m.empty:
        raise RuntimeError("Road-name matches were found, but none were close enough to the configured corridor center.")
    corridor_union = selected_m.geometry.union_all() if hasattr(selected_m.geometry, "union_all") else selected_m.geometry.unary_union
    buffer = corridor_union.buffer(buffer_m)


    businesses = None
    business_errors = []

    for endpoint in endpoints:
        try:
            ox.settings.overpass_url = endpoint
            print(f"Downloading business data from: {endpoint}")

            businesses = ox.features_from_point(
                center,
                tags=TAGS,
                dist=query_radius
            )

            if not businesses.empty:
                break

        except Exception as exc:
            business_errors.append(f"{endpoint}: {exc}")
            print(f"Business request failed: {exc}")
            businesses = None

    if businesses is None or businesses.empty:
        raise RuntimeError(
            f"Could not download business data for {name}. "
            "All Overpass endpoints failed or returned no results.\n"
            + "\n".join(business_errors)
        )

    businesses = businesses[businesses.geometry.notna()].copy()
    if businesses.empty:
        raise RuntimeError("OSM returned no business features for this search area.")
    business_m = businesses.to_crs(crs_projected)
    business_m = business_m[business_m.geometry.intersects(buffer)].copy()
    if business_m.empty:
        raise RuntimeError("No business records were found inside the selected road buffer.")
    points = business_m.geometry.representative_point()
    points_ll = gpd.GeoSeries(points, crs=crs_projected).to_crs("EPSG:4326")
    result = pd.DataFrame({
        "name": business_m["name"].fillna("").astype(str).values if "name" in business_m.columns else [""] * len(business_m),
        "category": [category_for(row) for _, row in business_m.iterrows()],
        "latitude": points_ll.y.values,
        "longitude": points_ll.x.values,
    })
    # Retain raw OSM tags so later classification can be audited and improved.
    for tag in ["shop", "amenity", "office", "tourism", "leisure", "craft", "healthcare", "club", "public_transport", "building", "brand", "operator", "website"]:
        if tag in business_m.columns:
            result["osm_" + tag] = business_m[tag].fillna("").astype(str).values
    result["corridor"] = name
    result = result.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["name", "category", "latitude", "longitude"])
    result = enrich_categories(result)
    result["data_source"] = "OpenStreetMap via OSMnx"
    result["retrieved_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    if len(result) > 1:
        coords = np.radians(result[["latitude", "longitude"]].to_numpy())
        model = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
        result["cluster"] = model.fit_predict(coords)
        nn = NearestNeighbors(n_neighbors=2, metric="haversine", algorithm="ball_tree").fit(coords)
        distances, _ = nn.kneighbors(coords)
        result["nearest_business_m"] = distances[:, 1] * EARTH_RADIUS_M
    elif len(result) == 1:
        result["cluster"] = -1
        result["nearest_business_m"] = np.nan
    else:
        result["cluster"] = pd.Series(dtype=int)
        result["nearest_business_m"] = pd.Series(dtype=float)
    out = corridor_csv(name)
    result.to_csv(out, index=False)
    # Persist matching roads for the combined map.
    selected_ll = selected_m.to_crs("EPSG:4326")
    file_key = CORRIDORS[name]["short"].lower().replace(" ", "_").replace("/", "_")
    selected_ll[["geometry", "name"]].to_file(DATA_DIR / ("roads_" + file_key + ".geojson"), driver="GeoJSON")
    # Save the actual corridor boundary polygon, created by buffering the matched road
    # centerline in a projected CRS so the width is measured in metres.
    boundary_m = gpd.GeoDataFrame(
        {"corridor": [name], "buffer_m": [buffer_m]},
        geometry=[buffer], crs=crs_projected
    )
    boundary_m.to_crs("EPSG:4326").to_file(
        DATA_DIR / ("boundary_" + file_key + ".geojson"), driver="GeoJSON"
    )
    return result


def make_point_labels(df):
    """Create a visible label for every mapped point without inventing business names.

    Real OSM names are preserved. For unnamed records, use available OSM feature tags
    (shop/amenity/tourism/office/leisure/craft) or the category as a descriptive label,
    and add a stable sequence number when multiple points share the same description.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    tag_columns = ["shop", "amenity", "tourism", "office", "leisure", "craft", "healthcare", "public_transport", "historic"]
    labels = []
    for _, row in df.iterrows():
        real_name = str(row.get("name", "") or "").strip()
        if real_name and real_name.lower() not in {"nan", "none", "unnamed", "unnamed business"}:
            labels.append(real_name)
            continue
        descriptor = ""
        for col in tag_columns:
            value = str(row.get("osm_" + col, row.get(col, "")) or "").strip()
            if value and value.lower() not in {"nan", "none", "unknown", "no", "false"}:
                descriptor = value.replace("_", " ").replace(";", ", ").title()
                break
        if not descriptor:
            descriptor = str(row.get("category", "") or "").strip()
        if not descriptor or descriptor.lower() in {"nan", "none", "other", "unknown"}:
            descriptor = "Place"
        labels.append("Unnamed " + descriptor)
    counts = {}
    unique_labels = []
    for label in labels:
        if label.startswith("Unnamed "):
            counts[label] = counts.get(label, 0) + 1
            unique_labels.append(f"{label} #{counts[label]}")
        else:
            unique_labels.append(label)
    df["display_name"] = unique_labels
    return df


def load_corridor(name):
    path = corridor_csv(name)
    if not path.exists():
        return pd.DataFrame(columns=["name", "category", "major_activity", "latitude", "longitude", "corridor"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["name", "category", "major_activity", "latitude", "longitude", "corridor"])
    required = ["name", "category", "latitude", "longitude"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(columns=required + ["corridor"])
    df["corridor"] = name
    df["name"] = df["name"].fillna("").astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = enrich_categories(df)
    # Recompute spatial metrics with a consistent implementation when coordinates exist.
    if len(df) >= 2:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            coords = np.radians(df[["latitude", "longitude"]].to_numpy())
            df["cluster"] = DBSCAN(eps=100 / EARTH_RADIUS_M, min_samples=5, metric="haversine", algorithm="ball_tree").fit_predict(coords)
            dist, _ = NearestNeighbors(n_neighbors=2, metric="haversine", algorithm="ball_tree").fit(coords).kneighbors(coords)
            df["nearest_business_m"] = dist[:, 1] * EARTH_RADIUS_M
        except Exception:
            pass
    elif len(df) == 1:
        df["cluster"] = -1
        df["nearest_business_m"] = np.nan
    return df

def stats_for(df):
    n = len(df)
    named = df["name"].fillna("").astype(str).str.strip().ne("").sum() if n and "name" in df else 0
    cats = df["major_activity"].fillna("Miscellaneous Retail & Services").astype(str) if n and "major_activity" in df else (df["category"].fillna("other").astype(str) if n and "category" in df else pd.Series(dtype=str))
    top = cats.value_counts().index[0] if len(cats) else "—"
    cluster_count = int(df.loc[pd.to_numeric(df.get("cluster", pd.Series(index=df.index, dtype=float)), errors="coerce") != -1, "cluster"].nunique()) if n and "cluster" in df else None
    noise = int((pd.to_numeric(df["cluster"], errors="coerce") == -1).sum()) if n and "cluster" in df else None
    avg_near = float(pd.to_numeric(df["nearest_business_m"], errors="coerce").mean()) if n and "nearest_business_m" in df else None
    return {"records": n, "named": int(named), "categories": int(cats.nunique()) if len(cats) else 0, "top": top, "clusters": cluster_count, "noise": noise, "avg_near": avg_near}

# Sidebar
with st.sidebar:
    st.markdown("## 🗺️ Austin Corridor Intelligence")
    st.caption("Commercial geography · OpenStreetMap · DBSCAN")
    pages = ["Overview", "Corridor Explorer", "Business Analytics", "Cluster Analysis", "Compare Corridors", "Buffer Comparison", "Point Density", "Road Connectivity", "Data & Methodology", "Austin Corridor Map"]
    desired_page = st.session_state.get("page_override", "Overview")
    page = st.radio("NAVIGATION", pages, index=pages.index(desired_page) if desired_page in pages else 0, label_visibility="collapsed")
    st.divider()
    st.markdown("**Corridor data status**")
    for cname, info in CORRIDORS.items():
        exists = corridor_csv(cname).exists()
        st.markdown(f"<span style='color:{info['color']};font-size:1.1rem'>●</span> {info['short']} {'✓' if exists else '○'}", unsafe_allow_html=True)
    st.caption("✓ data available · ○ data not collected yet")

st.markdown('<div class="hero"><h1>Austin Corridor Intelligence</h1><p>Explore, analyse and compare eleven commercial corridors across Austin, Texas.</p></div>', unsafe_allow_html=True)

all_data = {name: load_corridor(name) for name in CORRIDORS}
ready_names = [name for name, df in all_data.items() if not df.empty]

if page == "Overview":
    total_records = sum(len(df) for df in all_data.values())
    ready_count = len(ready_names)
    unique_categories = len(set().union(*[set(df.get("major_activity", df["category"]).dropna().astype(str).unique()) for df in all_data.values() if not df.empty])) if ready_names else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Corridors planned", "11")
    c2.metric("Corridors with data", f"{ready_count}/11")
    c3.metric("Business records loaded", f"{total_records:,}")
    c4.metric("Categories observed", unique_categories)
    st.caption("Counts combine only datasets currently available on this device. New corridors must be collected before a full eleven-corridor comparison is possible.")
    st.markdown('<div class="section-title">Explore the eleven corridors</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, info) in enumerate(CORRIDORS.items()):
        df = all_data[name]; s = stats_for(df)
        with cols[i % 3]:
            st.markdown(f"<div class='small-card' style='border-top:4px solid {info['color']};min-height:180px'><h3 style='margin:.1rem 0;color:#173c34'>{info['short']}</h3><p style='color:#687b76;font-size:.9rem'>{info['description']}</p><p><b>{s['records']:,}</b> business records {'· '+str(s['categories'])+' categories' if s['records'] else '· data not collected'}</p></div>", unsafe_allow_html=True)
            if st.button("Open corridor", key=f"open_{i}", width="stretch"):
                st.session_state["selected_corridor"] = name
                st.session_state["page_override"] = "Corridor Explorer"
                st.rerun()
            if st.button("Collect / refresh OSM data", key=f"collect_{i}", width="stretch"):
                with st.spinner(f"Collecting road and business data for {name}. This may take a few minutes..."):
                    try:
                        fetch_corridor(
                            name,
                            radius_m=info["search_radius_m"],
                            buffer_m=150,
                            eps_m=100,
                            min_samples=5,
                        )
                        st.success(f"Saved OSM data for {name}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not collect {name}: {exc}")

elif page == "Corridor Explorer":
    default = st.session_state.get("selected_corridor", "South Congress Avenue (SoCo)")
    selected = st.selectbox("Select a corridor", list(CORRIDORS), index=list(CORRIDORS).index(default) if default in CORRIDORS else 0)
    st.session_state["selected_corridor"] = selected
    df = all_data[selected]
    info = CORRIDORS[selected]
    st.markdown(f"### <span style='color:{info['color']}'>●</span> {selected}", unsafe_allow_html=True)
    st.write(info["description"])
    if df.empty:
        st.warning("No dataset has been collected for this corridor yet.")
        st.write("Use **Overview → Collect missing corridor data** to download its OSM road and business records.")
    else:
        s = stats_for(df)
        a,b,c,d,e = st.columns(5)
        a.metric("Business records", f"{s['records']:,}")
        b.metric("Named records", f"{s['named']:,}")
        c.metric("Categories", s["categories"])
        d.metric("DBSCAN clusters", s["clusters"] if s["clusters"] is not None else "Not run")
        e.metric("Noise points", s["noise"] if s["noise"] is not None else "Not run")
        left,right = st.columns([1.6,1])
        with left:
            st.markdown("#### Business locations")
            try:
                import folium
                from streamlit_folium import st_folium
                valid_coords = df.dropna(subset=["latitude", "longitude"]).copy()
                if valid_coords.empty:
                    st.warning("No valid business coordinates are available for this corridor.")
                else:
                    center = [valid_coords["latitude"].mean(), valid_coords["longitude"].mean()]
                    # Disable all default basemaps, then add OpenStreetMap explicitly.
                    # This removes the Carto API-key watermark seen in the previous map.
                    m = folium.Map(
                        location=center, zoom_start=14, tiles=None, control_scale=True,
                        prefer_canvas=True, fade_animation=False, zoom_animation=False,
                        marker_zoom_animation=False,
                    )
                    folium.TileLayer(
                        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
                        name="OpenStreetMap", overlay=False, control=True,
                        max_zoom=19, update_when_idle=True, keep_buffer=4,
                    ).add_to(m)
                    file_key = info["short"].lower().replace(" ", "_").replace("/", "_")
                    road_path = DATA_DIR / ("roads_" + file_key + ".geojson")
                    boundary_path = DATA_DIR / ("boundary_" + file_key + ".geojson")
                    # SoCo's reviewed study polygon is the preferred boundary. Always
                    # prefer it over a possibly stale boundary_soco.geojson generated by
                    # an earlier refresh; use the generated boundary only as a fallback.
                    if selected == "South Congress Avenue (SoCo)":
                        soco_boundary = DATA_DIR / "soco_corridor_area.geojson"
                        soco_roads = DATA_DIR / "soco_corridor_roads.geojson"
                        if soco_boundary.exists():
                            boundary_path = soco_boundary
                        if soco_roads.exists():
                            road_path = soco_roads
                    # Load the boundary geometry once, draw it once, and use the same
                    # geometry to keep displayed business markers inside the study area.
                    boundary_geom = None
                    boundary_gdf = None
                    boundary_source = None

                    if boundary_path.exists():
                        try:
                            import geopandas as gpd
                            boundary_gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
                            boundary_gdf = boundary_gdf[
                                boundary_gdf.geometry.notna() & ~boundary_gdf.geometry.is_empty
                            ].copy()
                            if not boundary_gdf.empty:
                                boundary_geom = boundary_gdf.geometry.union_all()
                                boundary_source = boundary_path.name
                        except Exception as exc:
                            st.caption(f"Saved boundary could not be read; trying the road buffer instead. Details: {exc}")
                            boundary_gdf = None
                            boundary_geom = None

                    # If no valid polygon exists, create one from the saved corridor road
                    # centerline. Buffer in a projected CRS so metres are measured correctly.
                    if boundary_geom is None and road_path.exists():
                        try:
                            import geopandas as gpd
                            roads_gdf = gpd.read_file(road_path)
                            roads_gdf = roads_gdf[
                                roads_gdf.geometry.notna() & ~roads_gdf.geometry.is_empty
                            ].copy()
                            if not roads_gdf.empty:
                                projected = roads_gdf.to_crs(roads_gdf.estimate_utm_crs())
                                polygon = projected.geometry.union_all().buffer(int(info["buffer_m"]))
                                boundary_gdf = gpd.GeoDataFrame(
                                    {"corridor": [selected]},
                                    geometry=[polygon],
                                    crs=projected.crs
                                ).to_crs("EPSG:4326")
                                boundary_geom = boundary_gdf.geometry.union_all()
                                boundary_source = f"buffered from {road_path.name}"
                        except Exception as exc:
                            st.warning(f"Could not create a boundary from the saved roads: {exc}")

                    # Draw the polygon before points so the markers remain visible on top.
                    if boundary_gdf is not None and boundary_geom is not None:
                        try:
                            folium.GeoJson(
                                boundary_gdf.to_json(),
                                name="Corridor study boundary",
                                style_function=lambda feature: {
                                    "color": info["color"],
                                    "weight": 5,
                                    "fillColor": info["color"],
                                    "fillOpacity": 0.08,
                                    "opacity": 1.0,
                                    "dashArray": "8, 5",
                                },
                                tooltip=f"{selected} study boundary"
                            ).add_to(m)
                            minx, miny, maxx, maxy = boundary_gdf.total_bounds
                            m.fit_bounds(
                                [[miny, minx], [maxy, maxx]],
                                padding=(25, 25),
                                max_zoom=16
                            )
                        except Exception as exc:
                            st.caption(f"Boundary could not be drawn: {exc}")

                    # Filter markers against the exact polygon displayed on the map.
                    map_businesses = valid_coords.copy()
                    if boundary_geom is not None:
                        try:
                            from shapely.geometry import Point
                            inside = map_businesses.apply(
                                lambda row: boundary_geom.covers(
                                    Point(float(row["longitude"]), float(row["latitude"]))
                                ),
                                axis=1
                            )
                            map_businesses = map_businesses.loc[inside].copy()
                            st.caption(
                                f"Boundary: {boundary_source}. Showing {len(map_businesses)} "
                                f"businesses inside the study area out of {len(valid_coords)} "
                                "records with valid coordinates."
                            )
                        except Exception as exc:
                            st.warning(f"Could not filter markers to the boundary: {exc}")
                    else:
                        st.warning(
                            "No valid corridor polygon or road file was found. "
                            "All records with valid coordinates are shown, so the boundary cannot be verified."
                        )

                    # Draw plain points only: no names, tooltips, or popups.
                    for _, row in map_businesses.iterrows():
                        folium.CircleMarker(
                            location=[float(row["latitude"]), float(row["longitude"])],
                            radius=5,
                            color="#ffffff",
                            fill=True,
                            fill_color=info["color"],
                            fill_opacity=0.95,
                            weight=1.5,
                        ).add_to(m)
                    map_key = "corridor_map_" + info["short"].lower().replace(" ", "_").replace("/", "_")
                    st_folium(m, height=500, use_container_width=True, key=map_key)
            except ImportError:
                st.warning("Install folium and streamlit-folium to display the interactive map.")
                st.dataframe(df[[c for c in ["name","category","latitude","longitude"] if c in df]], width="stretch")
        with right:
            # Both charts are derived from the same loaded records and same activity classification.
            if "major_activity" not in df.columns:
                df = enrich_categories(df.copy())
            if "major_activity" in df.columns:
                activity_series = (
                    df["major_activity"]
                    .fillna("Miscellaneous Retail & Services")
                    .astype(str)
                    .str.strip()
                    .replace("", "Miscellaneous Retail & Services")
                    .value_counts()
                )
            else:
                activity_series = pd.Series(dtype="int64")
            counts = activity_series.sort_values(ascending=False)

            st.markdown("#### Signature activity profile")
            st.caption("Qualitative 1–5 profile informed by City of Austin corridor-planning documents and documented local business character. This is a researched corridor profile, not a count of businesses.")
            researched_profile = CORRIDOR_PROFILE.get(selected, {})
            if researched_profile:
                profile_series = pd.Series(researched_profile, dtype="float64").sort_values(ascending=True)
                profile_series.name = "Research-informed strength (1–5)"
                st.bar_chart(profile_series, horizontal=True)

                # Display the requested corridor leaders explicitly beneath the profile chart.
                leader_summary = CORRIDOR_LEADERS.get(selected)
                if leader_summary:
                    st.markdown("##### Corridor activity leaders")
                    st.dataframe(
                        pd.DataFrame([leader_summary], columns=["Leading activity", "Other strong activities"]),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.info("No researched profile is available for this corridor.")

            st.markdown("#### Actual mapped business mix")
            st.caption("Observed counts in the currently loaded OpenStreetMap/sample CSV, classified from each record. This sample may be incomplete and is not a full business census.")
            if not counts.empty:
                count_chart = counts.copy()
                count_chart.name = "Mapped businesses"
                st.bar_chart(count_chart.sort_values(ascending=True), horizontal=True)
            else:
                st.info("No mapped business records are available for this corridor.")

        st.markdown("#### Quick findings")

        # Corridor-specific profile summary plus facts calculated from the selected corridor's records.
        leader_summary = CORRIDOR_LEADERS.get(selected, {})
        profile_leader = leader_summary.get("Leading activity", "Not available")
        profile_support = leader_summary.get("Other strong activities", "Not available")

        if not counts.empty:
            mapped_top = str(counts.index[0])
            mapped_top_count = int(counts.iloc[0])
            tied_top = [str(cat) for cat, count in counts.items() if int(count) == mapped_top_count]
            mapped_leader_text = " and ".join(tied_top)
            st.write(f"**Signature profile leader:** {profile_leader}")
            st.write(f"**Other strong profile activities:** {profile_support}")
            st.write(
                f"**Most common activity in this corridor's loaded OSM sample:** "
                f"{mapped_leader_text} ({mapped_top_count} records)"
            )
        else:
            st.write(f"**Signature profile leader:** {profile_leader}")
            st.write(f"**Other strong profile activities:** {profile_support}")
            st.info("No mapped business records are available for this corridor, so a sample leader cannot be calculated.")

        category_col = "major_activity" if "major_activity" in df.columns else None
        distinct_categories = (
            int(df[category_col].fillna("Miscellaneous Retail & Services").astype(str).nunique())
            if category_col and not df.empty else 0
        )
        st.write(f"**Mapped business records:** {len(df):,}")
        st.write(f"**Distinct mapped activity categories:** {distinct_categories}")

        st.caption(
            "The signature profile describes the corridor's broader commercial character using planning sources. "
            "The mapped leader is calculated from the records currently loaded for this corridor, which may be an "
            "incomplete sample rather than a full business census. A difference between these two leaders is possible."
        )

        if s["avg_near"] is not None and not math.isnan(s["avg_near"]):
            st.write(f"**Mean nearest-business distance:** {s['avg_near']:.1f} m")

        st.markdown("#### Places along this corridor")
        st.caption(
            "A cleaner directory of mapped places. Where OpenStreetMap does not provide an official name, "
            "the row gets a descriptive label such as ‘Unnamed Cafe #1’ instead of a blank cell."
        )

        labelled_df = make_point_labels(df)
        display = pd.DataFrame(index=labelled_df.index)
        display["Place / map label"] = labelled_df.get("display_name", labelled_df.get("name", "Place"))

        if "major_activity" in labelled_df.columns:
            activity = labelled_df["major_activity"].fillna("").astype(str).str.replace("_", " ").str.strip()
        elif "category" in labelled_df.columns:
            activity = labelled_df["category"].fillna("").astype(str).str.replace("_", " ").str.strip()
        else:
            activity = pd.Series(["Mapped place"] * len(labelled_df), index=labelled_df.index)
        display["Activity"] = activity.replace({"": "Other mapped place", "nan": "Other mapped place", "None": "Other mapped place"}).str.title()

        # Keep the public-facing directory tidy; hide raw OSM tag columns and null-heavy fields.
        for col, label in [("latitude", "Latitude"), ("longitude", "Longitude")]:
            if col in labelled_df.columns:
                display[label] = pd.to_numeric(labelled_df[col], errors="coerce").round(5)

        tag_candidates = ["shop", "amenity", "tourism", "office", "leisure", "craft", "healthcare", "public_transport", "historic"]
        def feature_type(row):
            for tag in tag_candidates:
                value = row.get("osm_" + tag, row.get(tag, ""))
                value = str(value or "").strip()
                if value and value.lower() not in {"nan", "none", "unknown", "no", "false"}:
                    return value.replace("_", " ").title()
            return "—"
        display["Map feature"] = labelled_df.apply(feature_type, axis=1)

        search = st.text_input("Search places by name, activity, or map feature", key="business_search")
        view = display.copy()
        if search:
            view = view[view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]

        named_count = int((labelled_df.get("name", pd.Series(index=labelled_df.index, dtype=object)).fillna("").astype(str).str.strip() != "").sum())
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Places shown", f"{len(view):,}")
        col_b.metric("With official OSM names", f"{named_count:,}")
        col_c.metric("Descriptive labels", f"{max(len(labelled_df) - named_count, 0):,}")

        st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            height=420,
            column_config={
                "Place / map label": st.column_config.TextColumn("Place / map label", help="Official name when available; otherwise a descriptive label for the mapped feature.", width="large"),
                "Activity": st.column_config.TextColumn("Activity", width="medium"),
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.5f"),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.5f"),
                "Map feature": st.column_config.TextColumn("Map feature", width="medium"),
            },
        )

        st.download_button(
            "Download corridor CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{info['short'].lower().replace(' ','_')}_businesses.csv",
            mime="text/csv",
        )

elif page == "Business Analytics":
    st.markdown("## 📊 Business Analytics")
    st.write("Explore the observed business mix across the corridor datasets currently available on this device.")
    available = {name: df.copy() for name, df in all_data.items() if not df.empty}
    if not available:
        st.warning("No corridor datasets are loaded yet. Go to Overview and fetch a corridor from OpenStreetMap.")
    else:
        choices = ["All corridors"] + list(available.keys())
        choice = st.selectbox("Analyse", choices, key="analytics_corridor")
        frames = []
        for cname, data in available.items():
            if choice == "All corridors" or choice == cname:
                part = data.copy()
                part["corridor"] = cname
                frames.append(part)
        analysis = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not analysis.empty:
            analysis = enrich_categories(analysis)
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Business records", f"{len(analysis):,}")
            metric2.metric("Named businesses", f"{analysis['name'].fillna('').astype(str).str.strip().ne('').sum():,}")
            metric3.metric("Activity categories", int(analysis['major_activity'].nunique()))
            metric4.metric("Corridors represented", int(analysis['corridor'].nunique()))
            st.markdown("### Observed activity mix")
            mix = analysis["major_activity"].fillna("Miscellaneous Retail & Services").value_counts().sort_values(ascending=False)
            st.bar_chart(mix, horizontal=True)
            st.markdown("### Activity mix by corridor")
            cross = pd.crosstab(analysis["corridor"], analysis["major_activity"])
            st.dataframe(cross, width="stretch")
            st.markdown("### Search and inspect records")
            term = st.text_input("Filter by business name, raw category, or activity", key="analytics_search")
            view = analysis.copy()
            if term.strip():
                mask = view.astype(str).apply(lambda col: col.str.contains(term.strip(), case=False, na=False)).any(axis=1)
                view = view.loc[mask]
            st.caption(f"Showing {len(view):,} of {len(analysis):,} records. Counts describe the loaded OSM sample, not a complete business census.")
            st.dataframe(view, width="stretch", hide_index=True)
            st.download_button("Download analytics records (CSV)", view.to_csv(index=False).encode("utf-8"), "business_analytics.csv", "text/csv")

elif page == "Cluster Analysis":
    st.markdown("## 🧩 Cluster Analysis")
    st.write("DBSCAN groups nearby business locations. Cluster labels are recalculated from coordinates when data is loaded.")
    available = {name: df.copy() for name, df in all_data.items() if not df.empty}
    if not available:
        st.warning("No corridor datasets are loaded yet. Fetch data from the Overview page first.")
    else:
        chosen = st.selectbox("Choose corridor", list(available), key="cluster_corridor")
        cdf = available[chosen].copy()
        if "cluster" not in cdf.columns:
            try:
                from sklearn.cluster import DBSCAN
                coords = np.radians(cdf[["latitude", "longitude"]].dropna().to_numpy(dtype=float))
                labels = DBSCAN(eps=100 / EARTH_RADIUS_M, min_samples=5, metric="haversine", algorithm="ball_tree").fit_predict(coords) if len(coords) >= 2 else np.array([-1] * len(coords))
                cdf.loc[cdf[["latitude", "longitude"]].dropna().index, "cluster"] = labels
            except Exception as exc:
                st.error(f"Could not calculate DBSCAN clusters: {exc}")
        if "cluster" in cdf.columns:
            labels = pd.to_numeric(cdf["cluster"], errors="coerce").fillna(-1).astype(int)
            non_noise = sorted(x for x in labels.unique() if x != -1)
            noise_count = int((labels == -1).sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Clusters found", len(non_noise))
            k2.metric("Businesses in clusters", int((labels != -1).sum()))
            k3.metric("Noise / isolated points", noise_count)
            counts = labels[labels != -1].value_counts().sort_index()
            if not counts.empty:
                st.markdown("### Businesses per cluster")
                counts.index = [f"Cluster {x}" for x in counts.index]
                st.bar_chart(counts)
            else:
                st.info("DBSCAN did not find any dense clusters with the current settings. All points may be noise.")
            st.markdown("### Cluster summary")
            temp = cdf.copy()
            temp["cluster"] = labels
            temp["major_activity"] = temp.get("major_activity", pd.Series("Miscellaneous Retail & Services", index=temp.index)).fillna("Miscellaneous Retail & Services")
            summary = temp.groupby("cluster", dropna=False).agg(Businesses=("name", "size"), Activities=("major_activity", lambda x: x.nunique()), Mean_latitude=("latitude", "mean"), Mean_longitude=("longitude", "mean")).reset_index()
            summary["cluster"] = summary["cluster"].apply(lambda x: "Noise / unclustered" if int(x) == -1 else f"Cluster {int(x)}")
            st.dataframe(summary, width="stretch", hide_index=True)
            st.caption("Current default: haversine distance, 100 m epsilon, and minimum 5 points. Noise label −1 means a point was not assigned to a dense cluster; it does not mean the record is invalid.")
        st.markdown("### Spatial records")
        st.dataframe(cdf[[c for c in ["name", "major_activity", "latitude", "longitude", "cluster", "nearest_business_m"] if c in cdf.columns]], width="stretch", hide_index=True)

elif page == "Compare Corridors":
    st.markdown("## ⚖️ Compare Corridors")
    st.write("Compare only the datasets available locally. Missing corridors are shown as unavailable rather than filled with estimated counts.")
    rows = []
    for cname, cdf in all_data.items():
        stt = stats_for(cdf)
        leaders = CORRIDOR_LEADERS.get(cname, {})
        rows.append({"Corridor": cname, "Data status": "Available" if not cdf.empty else "Not collected", "Business records": stt["records"], "Named records": stt["named"], "Observed activity categories": stt["categories"], "Most common observed activity": stt["top"], "DBSCAN clusters": stt["clusters"] if stt["clusters"] is not None else "—", "Signature profile leader": leaders.get("Leading activity", "—"), "Other strong profile activities": leaders.get("Other strong activities", "—")})
    comparison = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Loaded business records")
        records = comparison.set_index("Corridor")["Business records"].astype(int)
        st.bar_chart(records)
    with c2:
        st.markdown("### Observed activity diversity")
        diversity = comparison.set_index("Corridor")["Observed activity categories"].astype(int)
        st.bar_chart(diversity)
    st.markdown("### Comparison table")
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.download_button("Download corridor comparison (CSV)", comparison.to_csv(index=False).encode("utf-8"), "corridor_comparison.csv", "text/csv")
    st.info("Signature profile leaders are qualitative research-informed themes. Observed leaders and record counts are calculated from the loaded OSM sample. These are different measures and may not match.")

elif page == "Buffer Comparison":
    st.markdown("## 📏 Standardized Buffer Comparison")
    st.write("Choose a corridor to compare separate 50 m, 100 m and 150 m maps, business counts, all activity groups, and mapped-point density. Results are calculated from the locally saved business records and road centreline geometry.")
    selected_buffer_corridor = st.selectbox("Choose corridor for the three buffer maps", list(CORRIDORS), key="buffer_map_corridor")
    buf_values = (50, 100, 150)
    try:
        import geopandas as gpd
        import folium
        from shapely.geometry import Point
        from streamlit_folium import st_folium
        info = CORRIDORS[selected_buffer_corridor]
        cdf = all_data[selected_buffer_corridor].copy()
        key = info["short"].lower().replace(" ", "_").replace("/", "_")
        road_path = DATA_DIR / ("roads_" + key + ".geojson")
        if cdf.empty:
            st.warning("No business records are loaded for this corridor. Confirm that collection completed and the CSV is saved in the data folder.")
        elif not road_path.exists():
            st.warning(f"Saved road geometry not found: {road_path.name}. Collect/refresh this corridor so the app can create its road file.")
        else:
            roads = gpd.read_file(road_path)
            roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
            if roads.empty:
                st.warning("The saved road file contains no usable geometry.")
            else:
                metric_crs = roads.estimate_utm_crs()
                roads_m = roads.to_crs(metric_crs)
                centreline = roads_m.geometry.union_all()
                pts = gpd.GeoDataFrame(cdf.copy(), geometry=[Point(xy) for xy in zip(cdf["longitude"], cdf["latitude"])], crs="EPSG:4326").to_crs(metric_crs)
                pts["distance_to_road_m"] = pts.geometry.distance(centreline)
                area_m2 = {b: max(centreline.buffer(b).area, 1) for b in buf_values}
                map_cols = st.columns(3)
                comparison_rows = []
                for idx, buf in enumerate(buf_values):
                    subset = pts[pts["distance_to_road_m"] <= buf].copy()
                    count = len(subset)
                    dens = count / (area_m2[buf] / 1_000_000)
                    comparison_rows.append({"Buffer (m)": buf, "Business records": count, "Study area (km²)": round(area_m2[buf] / 1_000_000, 3), "Points/km²": round(dens, 2)})
                    with map_cols[idx]:
                        st.markdown(f"### {buf} m buffer")
                        st.metric("Business points", count)
                        map_obj = folium.Map(location=list(info["center"]), zoom_start=14, control_scale=True, tiles="CartoDB positron")
                        boundary = gpd.GeoSeries([centreline.buffer(buf)], crs=metric_crs).to_crs("EPSG:4326")
                        folium.GeoJson(boundary.__geo_interface__, name=f"{buf} m study buffer", style_function=lambda feat, color=info["color"]: {"color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.12}).add_to(map_obj)
                        road_wgs = roads_m.to_crs("EPSG:4326")
                        folium.GeoJson(road_wgs.__geo_interface__, name="Matched road geometry", style_function=lambda feat: {"color": "#263238", "weight": 3}).add_to(map_obj)
                        for _, row in subset.iterrows():
                            label = str(row.get("name", "Unnamed place"))
                            activity = str(row.get("major_activity", row.get("category", "Unclassified")))
                            folium.CircleMarker(location=[float(row["latitude"]), float(row["longitude"])], radius=3, color=info["color"], fill=True, fill_opacity=0.8, popup=folium.Popup(f"<b>{label}</b><br>{activity}<br>Distance to road: {row['distance_to_road_m']:.1f} m", max_width=260)).add_to(map_obj)
                        folium.LayerControl(collapsed=True).add_to(map_obj)
                        st_folium(map_obj, height=390, use_container_width=True, key=f"buffer_map_{key}_{buf}")
                comparison_df = pd.DataFrame(comparison_rows)
                st.markdown("### Buffer metrics")
                st.dataframe(comparison_df, width="stretch", hide_index=True)
                st.download_button("Download selected corridor buffer metrics (CSV)", comparison_df.to_csv(index=False).encode("utf-8"), f"{key}_buffer_metrics.csv", "text/csv")
                st.markdown("### Activity-group distribution by buffer")
                category_rows = []
                for buf in buf_values:
                    subset = pts[pts["distance_to_road_m"] <= buf]
                    counts = subset["major_activity"].fillna("Miscellaneous Retail & Services").value_counts()
                    for activity, amount in counts.items():
                        category_rows.append({"Buffer (m)": buf, "Activity group": activity, "Business records": int(amount)})
                category_df = pd.DataFrame(category_rows, columns=["Buffer (m)", "Activity group", "Business records"])
                st.dataframe(category_df, width="stretch", hide_index=True)
                if not category_df.empty:
                    st.bar_chart(category_df.pivot_table(index="Activity group", columns="Buffer (m)", values="Business records", aggfunc="sum", fill_value=0))
                st.download_button("Download activity groups by buffer (CSV)", category_df.to_csv(index=False).encode("utf-8"), f"{key}_activity_by_buffer.csv", "text/csv")
                st.caption("The 50 m, 100 m and 150 m maps are independently filtered from the same saved point inventory. If collection originally fetched only a narrow buffer, points outside that inventory cannot be recreated by filtering; recollect with a 150 m buffer for a fair comparison.")
    except Exception as exc:
        st.error(f"Buffer comparison or maps could not be rendered: {exc}")
        st.info("Check that GeoPandas, Shapely, Folium and streamlit-folium are installed, and that the corridor has saved road geometry.")

elif page == "Point Density":
    st.markdown("## 🔵 Point Density Analysis")
    st.write("Point density is calculated separately for each corridor at 50 m, 100 m and 150 m from the saved road centreline. It measures mapped OSM points, not a complete business census.")
    rows = []
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        for cname, cdf in all_data.items():
            info = CORRIDORS[cname]
            key = info["short"].lower().replace(" ", "_").replace("/", "_")
            road_path = DATA_DIR / ("roads_" + key + ".geojson")
            if cdf.empty or not road_path.exists():
                for buf in (50, 100, 150):
                    rows.append({"Corridor": cname, "Buffer (m)": buf, "Mapped points": 0 if cdf.empty else "—", "Study area (km²)": "—", "Points/km²": "—", "Status": "Dataset or road file missing"})
                continue
            roads = gpd.read_file(road_path)
            roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
            if roads.empty:
                continue
            metric_crs = roads.estimate_utm_crs()
            roads_m = roads.to_crs(metric_crs)
            centreline = roads_m.geometry.union_all()
            pts = gpd.GeoDataFrame(cdf.copy(), geometry=[Point(xy) for xy in zip(cdf["longitude"], cdf["latitude"])], crs="EPSG:4326").to_crs(metric_crs)
            distance = pts.geometry.distance(centreline)
            for buf in (50, 100, 150):
                count = int((distance <= buf).sum())
                area_km2 = max(centreline.buffer(buf).area / 1_000_000, 0.000001)
                rows.append({"Corridor": cname, "Buffer (m)": buf, "Mapped points": count, "Study area (km²)": round(area_km2, 3), "Points/km²": round(count / area_km2, 2), "Status": "Calculated from loaded records"})
        density_df = pd.DataFrame(rows)
        st.dataframe(density_df, width="stretch", hide_index=True)
        chart_df = density_df.copy()
        chart_df["Points/km²"] = pd.to_numeric(chart_df["Points/km²"], errors="coerce")
        if not chart_df.empty and chart_df["Points/km²"].notna().any():
            st.markdown("### Density by corridor and buffer")
            st.bar_chart(chart_df.pivot_table(index="Corridor", columns="Buffer (m)", values="Points/km²", aggfunc="first"))
        st.download_button("Download point density by corridor and buffer (CSV)", density_df.to_csv(index=False).encode("utf-8"), "corridor_point_density_by_buffer.csv", "text/csv")
        st.caption("Study area is the union of the saved road lines buffered by the selected distance, so overlapping geometry is counted once.")
    except Exception as exc:
        st.error(f"Point density analysis could not be calculated: {exc}")

elif page == "Road Connectivity":
    st.markdown("## 🛣️ Road Connectivity Analysis")
    st.write("Measures connectivity from the saved road line geometry. Junction counts are approximated by unique line endpoints after projection; they are not a full turn-by-turn road graph.")
    rows = []
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString
        for cname, cdf in all_data.items():
            info = CORRIDORS[cname]
            key = info["short"].lower().replace(" ", "_").replace("/", "_")
            road_path = DATA_DIR / ("roads_" + key + ".geojson")
            if not road_path.exists():
                rows.append({"Corridor": cname, "Road segments": "—", "Road length (km)": "—", "Unique endpoints": "—", "Approx. endpoint connectivity": "Road file missing"})
                continue
            roads = gpd.read_file(road_path)
            roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
            if roads.empty:
                rows.append({"Corridor": cname, "Road segments": 0, "Road length (km)": 0, "Unique endpoints": 0, "Approx. endpoint connectivity": "No geometry"})
                continue
            roads_m = roads.to_crs(roads.estimate_utm_crs())
            endpoints = []
            for geom in roads_m.geometry:
                parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
                for part in parts:
                    if isinstance(part, LineString) and len(part.coords) >= 2:
                        endpoints.extend([tuple(round(v, 1) for v in part.coords[0]), tuple(round(v, 1) for v in part.coords[-1])])
            unique_endpoints = len(set(endpoints))
            rows.append({"Corridor": cname, "Road segments": len(roads_m), "Road length (km)": round(roads_m.geometry.length.sum() / 1000, 2), "Unique endpoints": unique_endpoints, "Approx. endpoint connectivity": round((2 * len(roads_m)) / unique_endpoints, 2) if unique_endpoints else "—"})
        connectivity_df = pd.DataFrame(rows)
        st.dataframe(connectivity_df, width="stretch", hide_index=True)
        st.download_button("Download road connectivity summary (CSV)", connectivity_df.to_csv(index=False).encode("utf-8"), "corridor_road_connectivity.csv", "text/csv")
        st.caption("Endpoint connectivity = 2 × saved road-segment count ÷ unique endpoints. Since OSM road segments can be split at arbitrary points and intersection nodes are not fully reconstructed here, use this as a comparative geometry indicator, not an official intersection-density or network-accessibility score.")
    except Exception as exc:
        st.error(f"Road connectivity analysis could not be calculated: {exc}")

elif page == "Data & Methodology":
    st.markdown("## 📚 Data & Methodology")
    st.write("This page documents what the dashboard measures, how it classifies records, and what the results can—and cannot—claim.")
    st.markdown("### Study corridors")
    st.dataframe(pd.DataFrame([{"Corridor": name, "Reference centre (lat, lon)": f"{info['center'][0]:.4f}, {info['center'][1]:.4f}", "Search radius (m)": info["search_radius_m"], "Study buffer (m)": info["buffer_m"], "Commercial character": info["description"]} for name, info in CORRIDORS.items()]), width="stretch", hide_index=True)
    st.markdown("### Data source and collection")
    st.markdown("- **Source:** OpenStreetMap features and road-network data, retrieved through OSMnx when a corridor is fetched.\n- **Storage:** CSV files are stored in the app's `data` directory; matched roads and study boundaries may be saved as GeoJSON.\n- **Study boundary:** selected road segments buffered by a configured distance in a projected coordinate reference system. It is an analytical boundary, not an official zoning or legal boundary.\n- **Refreshes:** public Overpass services can time out or rate-limit requests. A failed fetch should be retried later; the app does not substitute fabricated business counts.")
    st.markdown("### Activity classification")
    st.write("The app classifies each record into broad activity groups using available OSM tags first, then name/category fallbacks for older CSVs. Raw source categories and `osm_` fields are retained where available so the classification can be audited.")
    st.dataframe(pd.DataFrame([{"Activity group": x} for x in ["Food & drink", "Retail & shopping", "Health & care", "Professional & financial services", "Education & culture", "Accommodation & tourism", "Recreation & fitness", "Transport & automotive", "Community & public services", "Miscellaneous Retail & Services"]]), width="stretch", hide_index=True)
    st.markdown("### Spatial clustering")
    st.markdown("- **Algorithm:** DBSCAN using haversine distance on latitude/longitude coordinates.\n- **Default epsilon:** 100 metres.\n- **Default minimum samples:** 5 points.\n- **Cluster labels:** non-negative values identify dense groups; `-1` identifies noise/unclustered points.\n- **Nearest-business distance:** distance to the nearest other record, estimated from the coordinates; it is not a walking-route distance.")
    st.markdown("### Interpretation and limitations")
    st.markdown("- OpenStreetMap is community-maintained and can omit businesses, contain duplicates, or have inconsistent tagging.\n- A downloaded sample is **not** a complete business census and should not be treated as official economic statistics.\n- A qualitative signature profile is a research-informed description of corridor character; it is not computed from the raw business counts.\n- The dashboard describes mapped records and spatial patterns; it does not establish causation or guarantee that one corridor is commercially better than another.")
    status = pd.DataFrame([{"Corridor": name, "CSV path": str(corridor_csv(name).relative_to(BASE_DIR)) if corridor_csv(name).is_relative_to(BASE_DIR) else str(corridor_csv(name)), "File exists": corridor_csv(name).exists(), "Loaded records": len(all_data[name])} for name in CORRIDORS])
    st.markdown("### Local dataset status")
    st.dataframe(status, width="stretch", hide_index=True)
    st.download_button("Download dataset status (CSV)", status.to_csv(index=False).encode("utf-8"), "dataset_status.csv", "text/csv")



elif page == "Austin Corridor Map":
    st.markdown("## 🗺️ Austin Corridor Map")
    st.write("Explore the approximate locations and study boundaries of all eleven commercial corridors. Select a corridor shape or marker to see its details.")
    try:
        import folium
        import geopandas as gpd
        from streamlit_folium import st_folium

        map_col, legend_col = st.columns([4.6, 1.05], gap="medium")
        with map_col:
            overview_map = folium.Map(
                location=(30.285, -97.755), zoom_start=12, tiles=None,
                control_scale=True, prefer_canvas=True, fade_animation=False,
                zoom_animation=False, marker_zoom_animation=False,
            )
            folium.TileLayer(
                tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
                name="OpenStreetMap", overlay=False, control=True,
                max_zoom=19, update_when_idle=True, keep_buffer=4,
            ).add_to(overview_map)
            all_bounds = []
            legend_rows = []

            for corridor_name, info in CORRIDORS.items():
                lat, lon = info["center"]
                color = info["color"]
                short = info["short"]
                file_key = short.lower().replace(" ", "_").replace("/", "_")
                boundary_path = DATA_DIR / ("boundary_" + file_key + ".geojson")
                roads_path = DATA_DIR / ("roads_" + file_key + ".geojson")

                # Prefer the reviewed SoCo study polygon and road line when present.
                if corridor_name == "South Congress Avenue (SoCo)":
                    if (DATA_DIR / "soco_corridor_area.geojson").exists():
                        boundary_path = DATA_DIR / "soco_corridor_area.geojson"
                    if (DATA_DIR / "soco_corridor_roads.geojson").exists():
                        roads_path = DATA_DIR / "soco_corridor_roads.geojson"

                boundary_gdf = None
                if boundary_path.exists():
                    try:
                        boundary_gdf = gpd.read_file(boundary_path)
                        if boundary_gdf.crs is None:
                            boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
                        boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
                        boundary_gdf = boundary_gdf[boundary_gdf.geometry.notna() & ~boundary_gdf.geometry.is_empty].copy()
                        if boundary_gdf.empty:
                            boundary_gdf = None
                    except Exception:
                        boundary_gdf = None

                # If no saved polygon exists, derive an approximate study polygon from saved road geometry.
                if boundary_gdf is None and roads_path.exists():
                    try:
                        roads_gdf = gpd.read_file(roads_path)
                        if roads_gdf.crs is None:
                            roads_gdf = roads_gdf.set_crs("EPSG:4326")
                        roads_gdf = roads_gdf[roads_gdf.geometry.notna() & ~roads_gdf.geometry.is_empty].copy()
                        if not roads_gdf.empty:
                            projected = roads_gdf.to_crs(roads_gdf.estimate_utm_crs())
                            polygon = projected.geometry.union_all().buffer(int(info["buffer_m"]))
                            boundary_gdf = gpd.GeoDataFrame({"corridor": [corridor_name]}, geometry=[polygon], crs=projected.crs).to_crs("EPSG:4326")
                    except Exception:
                        boundary_gdf = None

                if boundary_gdf is not None:
                    folium.GeoJson(
                        boundary_gdf.to_json(), name=f"{short} study boundary",
                        style_function=lambda feature, c=color: {
                            "color": c, "weight": 3, "fillColor": c,
                            "fillOpacity": 0.32, "opacity": 0.95,
                        },
                        highlight_function=lambda feature, c=color: {
                            "color": c, "weight": 5, "fillOpacity": 0.42,
                        },
                        tooltip=f"{short} — study boundary",
                        popup=folium.Popup(f"<b>{corridor_name}</b><br>Approximate study boundary<br>Buffer: {info['buffer_m']} m", max_width=300),
                    ).add_to(overview_map)
                    minx, miny, maxx, maxy = boundary_gdf.total_bounds
                    all_bounds.extend([[miny, minx], [maxy, maxx]])

                # Draw saved corridor road centreline on top of its polygon.
                if roads_path.exists():
                    try:
                        folium.GeoJson(
                            str(roads_path), name=f"{short} road centreline",
                            style_function=lambda feature, c=color: {"color": c, "weight": 4, "opacity": 1.0},
                            tooltip=f"{short} matched road centreline",
                        ).add_to(overview_map)
                    except Exception:
                        pass

                all_bounds.append([lat, lon])
                popup_html = (
                    f"<div style='min-width:220px'><b>{corridor_name}</b>"
                    f"<br>Reference centre: {lat:.4f}, {lon:.4f}"
                    f"<br>Search radius: {info['search_radius_m']:,} m"
                    f"<br>Study buffer: {info['buffer_m']:,} m"
                    f"<p>{info['description']}</p></div>"
                )
                folium.Marker(
                    location=[lat, lon], tooltip=short,
                    popup=folium.Popup(popup_html, max_width=340),
                    icon=folium.Icon(color="darkgreen", icon="map-marker", prefix="fa"),
                ).add_to(overview_map)
                label_html = (
                    "<div style='font-size:11px;font-weight:800;color:white;background:" + color +
                    ";padding:3px 7px;border:1px solid white;border-radius:6px;white-space:nowrap;"
                    "box-shadow:0 1px 4px #555'>" + short + "</div>"
                )
                folium.Marker(location=[lat, lon], icon=folium.DivIcon(html=label_html, icon_size=(150, 24), icon_anchor=(0, 12))).add_to(overview_map)
                legend_rows.append({"Corridor": corridor_name, "Map label": short, "Reference centre": f"{lat:.4f}, {lon:.4f}", "Boundary file loaded": "Yes" if boundary_gdf is not None else "No saved boundary/road", "Study buffer (m)": info["buffer_m"]})

            if all_bounds:
                overview_map.fit_bounds(all_bounds, padding=(35, 35), max_zoom=13)
            folium.LayerControl(collapsed=True).add_to(overview_map)
            st_folium(overview_map, height=650, width="stretch", key="austin_all_corridors_map_v2")

        with legend_col:
            st.markdown("<div style='background:#ffffff;color:#17352f;border:1px solid #dce7e2;border-radius:12px;padding:14px'>", unsafe_allow_html=True)
            st.markdown("#### Corridor legend")
            for corridor_name, info in CORRIDORS.items():
                st.markdown(
                    f"<div style='display:flex;gap:8px;align-items:flex-start;margin:0 0 13px 0;color:#17352f'>"
                    f"<span style='display:inline-block;min-width:13px;height:13px;margin-top:3px;border-radius:50%;background:{info['color']};border:1px solid #fff;box-shadow:0 0 0 1px {info['color']}'></span>"
                    f"<div><b>{info['short']}</b><br><span style='font-size:11px;color:#52645f'>{corridor_name}</span></div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            st.markdown("**Map details**")
            st.caption("📍 Pins show configured reference centres.")
            st.caption("⬟ Colored shapes show saved study boundaries where available.")
            st.caption("━ Colored lines show matched road centre lines where saved.")
            st.caption("Base map: OpenStreetMap.")
            st.info("Boundaries are analytical approximations created from matched roads and configured buffers, not official zoning or legal boundaries.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Corridor boundary status")
        st.dataframe(pd.DataFrame(legend_rows), width="stretch", hide_index=True)
        st.caption("If a corridor has no saved boundary or road GeoJSON file, its reference marker will still appear. Collect or restore that corridor's road/boundary files to display its shape.")
    except ImportError:
        st.warning("The interactive map needs folium, geopandas, and streamlit-folium. Install them with: pip install folium geopandas streamlit-folium")
    except Exception as exc:
        st.error(f"Could not render the Austin corridor map: {exc}")
