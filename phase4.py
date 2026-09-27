# ============================================================
# PHASE 4
# AI-DRIVEN AQUIFER RECHARGE POTENTIAL MAPPING
# USING GIS + XGBOOST
#
# STUDY AREA: THANE DISTRICT
# ============================================================


# ============================================================
# STEP 1 — IMPORT LIBRARIES AND PROJECT SETUP
# ============================================================

import os
import math
import re
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import box

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# FOLDERS
# ------------------------------------------------------------

DATA = "data"
OUTPUT = "outputs"

os.makedirs(OUTPUT, exist_ok=True)


print("\n")
print("============================================================")
print(" PHASE 4 - XGBOOST RECHARGE POTENTIAL MAPPING")
print(" STUDY AREA: THANE DISTRICT")
print("============================================================")


# ============================================================
# STEP 2 — READ ADMINISTRATIVE BOUNDARY
# ============================================================

print("\n")
print("============================================================")
print("STEP 2 - READING ADMINISTRATIVE BOUNDARY")
print("============================================================")


admin = gpd.read_file(
    os.path.join(
        DATA,
        "admin_boundary.geojson"
    )
)

print("\nColumns:")
print(admin.columns.tolist())

print("\nAvailable districts:")
print(admin["DTENAME"].unique())


# ============================================================
# STEP 3 — SELECT THANE DISTRICT
# ============================================================

print("\n")
print("============================================================")
print("STEP 3 - SELECTING THANE DISTRICT")
print("============================================================")


thane = admin[
    admin["DTENAME"].astype(str).str.lower() == "thane"
].copy()


if len(thane) == 0:
    raise ValueError(
        "THANE DISTRICT WAS NOT FOUND."
    )


print(
    "\nThane features:",
    len(thane)
)


print("\nOriginal CRS:")
print(thane.crs)


# Save original Thane boundary

thane.to_file(
    os.path.join(
        OUTPUT,
        "thane_boundary_original.geojson"
    ),
    driver="GeoJSON"
)


# ============================================================
# STEP 4 — CONVERT CRS
# ============================================================

print("\n")
print("============================================================")
print("STEP 4 - CONVERTING TO UTM CRS")
print("============================================================")


# EPSG 32643 = WGS84 / UTM Zone 43N

thane = thane.to_crs(
    epsg=32643
)


print("\nNew CRS:")
print(thane.crs)


# ============================================================
# STEP 5 — CREATE 1 KM × 1 KM GRID
# ============================================================

print("\n")
print("============================================================")
print("STEP 5 - CREATING 1 KM GRID")
print("============================================================")


minx, miny, maxx, maxy = thane.total_bounds


print("\nBoundary extent:")
print("minx =", minx)
print("miny =", miny)
print("maxx =", maxx)
print("maxy =", maxy)


# X coordinates

xs = np.arange(
    math.floor(minx / 1000) * 1000,
    math.ceil(maxx / 1000) * 1000 + 1000,
    1000
)


# Y coordinates

ys = np.arange(
    math.floor(miny / 1000) * 1000,
    math.ceil(maxy / 1000) * 1000 + 1000,
    1000
)


# Create grid cells

cells = []

for x in xs[:-1]:

    for y in ys[:-1]:

        cells.append(
            box(
                x,
                y,
                x + 1000,
                y + 1000
            )
        )


# Convert to GeoDataFrame

grid = gpd.GeoDataFrame(
    {
        "cell_id": range(len(cells))
    },
    geometry=cells,
    crs="EPSG:32643"
)


# Keep only Thane

grid = gpd.clip(
    grid,
    thane
)


# Calculate area

grid["area_m2"] = (
    grid.geometry.area
)


print(
    "\nNumber of grid cells:",
    len(grid)
)


print("\nArea statistics:")
print(
    grid["area_m2"].describe()
)


# Save grid

grid.to_file(
    os.path.join(
        OUTPUT,
        "thane_1km_grid.geojson"
    ),
    driver="GeoJSON"
)


# ============================================================
# STEP 6 — ASSIGN RECHARGE PRIORITY TARGET
# ============================================================

print("\n")
print("============================================================")
print("STEP 6 - RECHARGE PRIORITY TARGET")
print("============================================================")


priority = gpd.read_file(
    os.path.join(
        DATA,
        "recharge_priority.geojson"
    )
)


print("\nRecharge priority columns:")
print(
    priority.columns.tolist()
)


# Convert CRS

priority = priority.to_crs(
    epsg=32643
)


# Select Thane

priority = priority[
    priority["DTENAME"].astype(str).str.lower() == "thane"
].copy()


print(
    "\nThane recharge priority features:",
    len(priority)
)


# Check classes

print("\nRecharge priority classes:")

print(
    priority[
        "Recharge_priority"
    ].value_counts(
        dropna=False
    )
)


# Spatial intersection

intersection = gpd.overlay(
    grid[
        [
            "cell_id",
            "geometry"
        ]
    ],
    priority[
        [
            "Recharge_priority",
            "geometry"
        ]
    ],
    how="intersection",
    keep_geom_type=True
)


# Calculate area

intersection["intersection_area"] = (
    intersection.geometry.area
)


# Select dominant class

# Select the recharge class covering
# the largest area of each grid cell

dominant_priority = (
    intersection
    .sort_values(
        ["cell_id", "intersection_area"],
        ascending=[True, False]
    )
    .drop_duplicates(
        subset="cell_id"
    )
)

dominant_priority = dominant_priority[
    [
        "cell_id",
        "Recharge_priority"
    ]
]

dominant_priority = dominant_priority[
    [
        "cell_id",
        "Recharge_priority"
    ]
]


# Merge target into grid

grid = grid.merge(
    dominant_priority,
    on="cell_id",
    how="left"
)


print("\nTarget distribution:")

print(
    grid[
        "Recharge_priority"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nCells without target:",
    grid["Recharge_priority"].isna().sum()
)


# ============================================================
# STEP 7 — EXTRACT SLOPE
# ============================================================

print("\n")
print("============================================================")
print("STEP 7 - SLOPE")
print("============================================================")


slope = gpd.read_file(
    os.path.join(
        DATA,
        "slope.geojson"
    )
).to_crs(32643)


slope = gpd.clip(
    slope,
    thane
)


# Use grid centroids

centroids = grid[
    [
        "cell_id",
        "geometry"
    ]
].copy()


centroids["geometry"] = (
    centroids.geometry.centroid
)


slope_join = gpd.sjoin(
    centroids,
    slope,
    how="left",
    predicate="within"
)


# Remove duplicate cell matches

slope_join = (
    slope_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


# Add slope attributes

grid["slope"] = grid["cell_id"].map(
    slope_join.set_index("cell_id")["SLOPE"]
)

grid["slope_zone"] = grid["cell_id"].map(
    slope_join.set_index("cell_id")["SLP_ZONE"]
)


print("Slope extraction completed.")


# ============================================================
# STEP 8 — SOIL
# ============================================================

print("\n")
print("============================================================")
print("STEP 8 - SOIL")
print("============================================================")


soil = gpd.read_file(
    os.path.join(
        DATA,
        "soil.geojson"
    )
).to_crs(32643)


soil = gpd.clip(
    soil,
    thane
)


soil_join = gpd.sjoin(
    centroids,
    soil,
    how="left",
    predicate="within"
)


soil_join = (
    soil_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


soil_index = soil_join.set_index(
    "cell_id"
)


# Extract soil attributes

soil_columns = [
    "DEPTH",
    "TEXTURE",
    "EROSION",
    "DRAINAGE",
    "SOIL_ZONE"
]


for col in soil_columns:

    if col in soil_index.columns:

        grid[
            "soil_" + col.lower()
        ] = grid["cell_id"].map(
            soil_index[col]
        )


print("Soil extraction completed.")


# ============================================================
# STEP 9 — LITHOLOGY
# ============================================================

print("\n")
print("============================================================")
print("STEP 9 - LITHOLOGY")
print("============================================================")


lithology = gpd.read_file(
    os.path.join(
        DATA,
        "lithology.geojson"
    )
).to_crs(32643)


lithology = gpd.clip(
    lithology,
    thane
)


lith_join = gpd.sjoin(
    centroids,
    lithology,
    how="left",
    predicate="within"
)


lith_join = (
    lith_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


lith_index = lith_join.set_index(
    "cell_id"
)


for col in [
    "Rock_Group",
    "Rock_Type"
]:

    if col in lith_index.columns:

        grid[
            "lithology_" + col.lower()
        ] = grid["cell_id"].map(
            lith_index[col]
        )


print("Lithology extraction completed.")


# ============================================================
# STEP 10 — GEOMORPHOLOGY
# ============================================================

print("\n")
print("============================================================")
print("STEP 10 - GEOMORPHOLOGY")
print("============================================================")


geomorphology = gpd.read_file(
    os.path.join(
        DATA,
        "geomorphology.geojson"
    )
).to_crs(32643)


geomorphology = gpd.clip(
    geomorphology,
    thane
)


geom_join = gpd.sjoin(
    centroids,
    geomorphology,
    how="left",
    predicate="within"
)


geom_join = (
    geom_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


geom_index = geom_join.set_index(
    "cell_id"
)


for col in [
    "GEOM_L1_",
    "GEOM_L3",
    "GEOM_L5_"
]:

    if col in geom_index.columns:

        grid[
            "geom_" + col.lower()
        ] = grid["cell_id"].map(
            geom_index[col]
        )


print(
    "Geomorphology extraction completed."
)


# ============================================================
# STEP 11 — LULC
# ============================================================

print("\n")
print("============================================================")
print("STEP 11 - LAND USE / LAND COVER")
print("============================================================")


lulc = gpd.read_file(
    os.path.join(
        DATA,
        "lulc.geojson"
    )
).to_crs(32643)


lulc = gpd.clip(
    lulc,
    thane
)


lulc_join = gpd.sjoin(
    centroids,
    lulc,
    how="left",
    predicate="within"
)


lulc_join = (
    lulc_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


lulc_index = lulc_join.set_index(
    "cell_id"
)


for col in [
    "LU_15_16_L1",
    "LU_15_16_L2",
    "LU_15_16_L3"
]:

    if col in lulc_index.columns:

        grid[
            "lulc_" + col.lower()
        ] = grid["cell_id"].map(
            lulc_index[col]
        )


print(
    "LULC extraction completed."
)


# ============================================================
# STEP 12 — GROUNDWATER PROSPECT
# ============================================================

print("\n")
print("============================================================")
print("STEP 12 - GROUNDWATER PROSPECT")
print("============================================================")


gwp = gpd.read_file(
    os.path.join(
        DATA,
        "groundwater_prospect.geojson"
    )
).to_crs(32643)


gwp = gpd.clip(
    gwp,
    thane
)


gwp_join = gpd.sjoin(
    centroids,
    gwp,
    how="left",
    predicate="within"
)


gwp_join = (
    gwp_join
    .sort_values("cell_id")
    .drop_duplicates("cell_id")
)


gwp_index = gwp_join.set_index(
    "cell_id"
)


if "GWP_Zones" in gwp_index.columns:

    grid[
        "groundwater_prospect"
    ] = grid["cell_id"].map(
        gwp_index["GWP_Zones"]
    )


print(
    "Groundwater prospect extraction completed."
)


# ============================================================
# STEP 13 — LINEAMENT DENSITY
# ============================================================

print("\n")
print("============================================================")
print("STEP 13 - LINEAMENT DENSITY")
print("============================================================")


lineaments = gpd.read_file(
    os.path.join(
        DATA,
        "lineaments.geojson"
    )
).to_crs(32643)


lineaments = gpd.clip(
    lineaments,
    thane
)


line_intersection = gpd.overlay(
    grid[
        [
            "cell_id",
            "geometry"
        ]
    ],
    lineaments[
        [
            "geometry"
        ]
    ],
    how="intersection",
    keep_geom_type=False
)


# Keep line geometries only

line_intersection = (
    line_intersection[
        line_intersection.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString"
            ]
        )
    ]
)


line_intersection["length_m"] = (
    line_intersection.geometry.length
)


line_length = (
    line_intersection
    .groupby("cell_id")["length_m"]
    .sum()
)


grid["lineament_length_m"] = (
    grid["cell_id"]
    .map(line_length)
    .fillna(0)
)


# Density = km of lineament / km²

grid["lineament_density"] = (
    (
        grid["lineament_length_m"] / 1000
    )
    /
    (
        grid["area_m2"] / 1_000_000
    )
)


print(
    "Lineament density calculated."
)


# ============================================================
# STEP 14 — DRAINAGE DENSITY
# ============================================================

print("\n")
print("============================================================")
print("STEP 14 - DRAINAGE DENSITY")
print("============================================================")


drainage = gpd.read_file(
    os.path.join(
        DATA,
        "drainage_soi.geojson"
    )
).to_crs(32643)


drainage = gpd.clip(
    drainage,
    thane
)


drain_intersection = gpd.overlay(
    grid[
        [
            "cell_id",
            "geometry"
        ]
    ],
    drainage[
        [
            "geometry"
        ]
    ],
    how="intersection",
    keep_geom_type=False
)


drain_intersection = (
    drain_intersection[
        drain_intersection.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString"
            ]
        )
    ]
)


drain_intersection["length_m"] = (
    drain_intersection.geometry.length
)


drain_length = (
    drain_intersection
    .groupby("cell_id")["length_m"]
    .sum()
)


grid["drainage_length_m"] = (
    grid["cell_id"]
    .map(drain_length)
    .fillna(0)
)


grid["drainage_density"] = (
    (
        grid["drainage_length_m"] / 1000
    )
    /
    (
        grid["area_m2"] / 1_000_000
    )
)


print(
    "Drainage density calculated."
)


# ============================================================
# STEP 15 — RECHARGE STRUCTURE COVERAGE
# ============================================================

print("\n")
print("============================================================")
print("STEP 15 - RECHARGE STRUCTURES")
print("============================================================")


structures = gpd.read_file(
    os.path.join(
        DATA,
        "recharge_structure.geojson"
    )
).to_crs(32643)


structures = gpd.clip(
    structures,
    thane
)


structure_intersection = gpd.overlay(
    grid[
        [
            "cell_id",
            "geometry"
        ]
    ],
    structures[
        [
            "geometry"
        ]
    ],
    how="intersection",
    keep_geom_type=True
)


structure_intersection[
    "intersection_area"
] = structure_intersection.geometry.area


structure_area = (
    structure_intersection
    .groupby("cell_id")[
        "intersection_area"
    ]
    .sum()
)


grid["recharge_structure_area"] = (
    grid["cell_id"]
    .map(structure_area)
    .fillna(0)
)


grid["recharge_structure_pct"] = (
    grid["recharge_structure_area"]
    /
    grid["area_m2"]
    *
    100
)


print(
    "Recharge structure coverage calculated."
)


# ============================================================
# STEP 16 — CREATE MACHINE LEARNING DATASET
# ============================================================

print("\n")
print("============================================================")
print("STEP 16 - CREATE ML DATASET")
print("============================================================")


# ------------------------------------------------------------
# Remove cells without target
# ------------------------------------------------------------

ml_grid = grid[
    grid["Recharge_priority"].notna()
].copy()


print(
    "\nCells used for ML:",
    len(ml_grid)
)


# ------------------------------------------------------------
# Convert slope category into numerical value
# ------------------------------------------------------------

def convert_slope(value):

    if pd.isna(value):
        return np.nan

    text = str(value)

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )

    if len(numbers) == 0:
        return np.nan

    numbers = [
        float(x)
        for x in numbers
    ]

    if len(numbers) >= 2:

        return (
            numbers[0] +
            numbers[1]
        ) / 2

    return numbers[0]


# ------------------------------------------------------------
# IMPORTANT:
# Create slope_numeric in BOTH grid and ml_grid
# ------------------------------------------------------------

if "slope" in grid.columns:

    grid["slope_numeric"] = (
        grid["slope"].apply(
            convert_slope
        )
    )

    ml_grid["slope_numeric"] = (
        ml_grid["slope"].apply(
            convert_slope
        )
    )


# ------------------------------------------------------------
# Groundwater prospect numerical score
# ------------------------------------------------------------

gwp_mapping = {

    "Very High": 5,
    "High": 4,
    "Moderate": 3,
    "Low": 2,
    "Very Low": 1

}


# ------------------------------------------------------------
# IMPORTANT:
# Create groundwater_score in BOTH grid and ml_grid
# ------------------------------------------------------------

if "groundwater_prospect" in grid.columns:

    grid["groundwater_score"] = (
        grid[
            "groundwater_prospect"
        ].map(
            gwp_mapping
        )
    )

    ml_grid["groundwater_score"] = (
        ml_grid[
            "groundwater_prospect"
        ].map(
            gwp_mapping
        )
    )


print("\nNumerical feature conversion completed.")

print("\nChecking required columns:")

print(
    "slope_numeric in grid:",
    "slope_numeric" in grid.columns
)

print(
    "groundwater_score in grid:",
    "groundwater_score" in grid.columns
)

print(
    "slope_numeric in ML grid:",
    "slope_numeric" in ml_grid.columns
)

print(
    "groundwater_score in ML grid:",
    "groundwater_score" in ml_grid.columns
)

# ============================================================
# STEP 17 — SELECT FEATURES
# ============================================================

print("\n")
print("============================================================")
print("STEP 17 - SELECTING FEATURES")
print("============================================================")


candidate_features = [

    # Slope
    "slope_numeric",
    "slope_zone",

    # Soil
    "soil_depth",
    "soil_texture",
    "soil_erosion",
    "soil_drainage",
    "soil_soil_zone",

    # Lithology
    "lithology_rock_group",
    "lithology_rock_type",

    # Geomorphology
    "geom_geom_l1_",
    "geom_geom_l3",
    "geom_geom_l5_",

    # LULC
    "lulc_lu_15_16_l1",
    "lulc_lu_15_16_l2",
    "lulc_lu_15_16_l3",

    # Groundwater
    "groundwater_prospect",
    "groundwater_score",

    # Density
    "lineament_density",
    "drainage_density",

    # Recharge structures
    "recharge_structure_pct"

]


feature_columns = [

    col
    for col in candidate_features
    if col in ml_grid.columns

]


print("\nFeatures being used:")

for feature in feature_columns:

    print(
        " -",
        feature
    )


# ============================================================
# STEP 18 — BUILD X AND Y
# ============================================================

print("\n")
print("============================================================")
print("STEP 18 - BUILD X AND Y")
print("============================================================")


X = ml_grid[
    feature_columns
].copy()


y = ml_grid[
    "Recharge_priority"
].astype(str)


print("\nX shape:")
print(X.shape)


print("\nTarget classes:")
print(
    y.value_counts()
)


# ============================================================
# STEP 19 — CLEAN MISSING VALUES
# ============================================================

print("\n")
print("============================================================")
print("STEP 19 - CLEANING DATA")
print("============================================================")


categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


numeric_columns = [
    col
    for col in X.columns
    if col not in categorical_columns
]


# Fill categorical missing values

for col in categorical_columns:

    X[col] = (
        X[col]
        .astype(str)
        .replace(
            "nan",
            "Unknown"
        )
        .fillna("Unknown")
    )


# Fill numerical missing values

for col in numeric_columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )


# Save clean dataset

clean_dataset = X.copy()

clean_dataset[
    "Recharge_priority"
] = y.values


clean_dataset[
    "cell_id"
] = ml_grid[
    "cell_id"
].values


clean_dataset.to_csv(
    os.path.join(
        OUTPUT,
        "phase4_training_dataset.csv"
    ),
    index=False
)


print(
    "\nTraining dataset saved."
)


# ============================================================
# STEP 20 — ENCODE CATEGORICAL VARIABLES
# ============================================================

print("\n")
print("============================================================")
print("STEP 20 - ENCODING FEATURES")
print("============================================================")


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            encoder,
            categorical_columns
        )

    ],
    remainder="passthrough"
)


X_encoded = preprocessor.fit_transform(
    X
)


print(
    "\nEncoded X shape:",
    X_encoded.shape
)


# ============================================================
# STEP 21 — ENCODE TARGET
# ============================================================

print("\n")
print("============================================================")
print("STEP 21 - ENCODING TARGET")
print("============================================================")


classes = sorted(
    y.unique()
)


print(
    "\nClasses:",
    classes
)


label_to_number = {

    label: number

    for number, label
    in enumerate(classes)

}


number_to_label = {

    number: label

    for label, number
    in label_to_number.items()

}


y_encoded = y.map(
    label_to_number
).astype(int)


# ============================================================
# STEP 22 — TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("============================================================")
print("STEP 22 - TRAIN / TEST SPLIT")
print("============================================================")


X_train, X_test, y_train, y_test = train_test_split(

    X_encoded,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# STEP 23 — BALANCE TRAINING CLASSES
# ============================================================

print("\n")
print("============================================================")
print("STEP 23 - CLASS BALANCING")
print("============================================================")


sample_weights = compute_sample_weight(

    class_weight="balanced",

    y=y_train

)


# ============================================================
# STEP 24 — TRAIN XGBOOST
# ============================================================

print("\n")
print("============================================================")
print("STEP 24 - TRAINING XGBOOST")
print("============================================================")


model = XGBClassifier(

    n_estimators=350,

    max_depth=4,

    learning_rate=0.04,

    subsample=0.90,

    colsample_bytree=0.90,

    objective="multi:softprob",

    num_class=len(classes),

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=4

)


model.fit(

    X_train,

    y_train,

    sample_weight=sample_weights

)


print(
    "\nXGBoost training completed!"
)


# ============================================================
# STEP 25 — MODEL PREDICTION
# ============================================================

print("\n")
print("============================================================")
print("STEP 25 - MODEL PREDICTION")
print("============================================================")


test_prediction = model.predict(
    X_test
)


# ============================================================
# STEP 26 — MODEL EVALUATION
# ============================================================

print("\n")
print("============================================================")
print("STEP 26 - MODEL EVALUATION")
print("============================================================")


accuracy = accuracy_score(
    y_test,
    test_prediction
)


precision = precision_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)


print(
    "\nAccuracy:",
    round(accuracy, 4)
)


print(
    "Precision:",
    round(precision, 4)
)


print(
    "Recall:",
    round(recall, 4)
)


print(
    "F1 Score:",
    round(f1, 4)
)


# Detailed report

report = classification_report(

    y_test,

    test_prediction,

    labels=range(len(classes)),

    target_names=classes,

    zero_division=0

)


print("\nClassification Report:")
print(report)


# Save report

with open(
    os.path.join(
        OUTPUT,
        "model_metrics.txt"
    ),
    "w"
) as file:

    file.write(
        "XGBoost Recharge Prediction - Thane District\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# STEP 27 — CONFUSION MATRIX
# ============================================================

print("\n")
print("============================================================")
print("STEP 27 - CONFUSION MATRIX")
print("============================================================")


cm = confusion_matrix(

    y_test,

    test_prediction,

    labels=range(len(classes))

)


cm_df = pd.DataFrame(

    cm,

    index=classes,

    columns=classes

)


print("\nConfusion Matrix:")
print(cm_df)


cm_df.to_csv(
    os.path.join(
        OUTPUT,
        "confusion_matrix.csv"
    )
)


# Plot confusion matrix

fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.imshow(cm)


ax.set_xticks(
    range(len(classes))
)

ax.set_yticks(
    range(len(classes))
)


ax.set_xticklabels(
    classes
)

ax.set_yticklabels(
    classes
)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "XGBoost Confusion Matrix - Thane"
)


for i in range(len(classes)):

    for j in range(len(classes)):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT,
        "confusion_matrix.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# STEP 28 — PREDICT ENTIRE THANE GRID
# ============================================================

print("\n")
print("============================================================")
print("STEP 28 - FULL THANE XGBOOST PREDICTION")
print("============================================================")


# Prepare ALL grid features

X_all = grid[
    feature_columns
].copy()


# Fill categorical values

for col in categorical_columns:

    X_all[col] = (
        X_all[col]
        .astype(str)
        .replace(
            "nan",
            "Unknown"
        )
        .fillna(
            "Unknown"
        )
    )


# Fill numerical values using
# training medians

for col in numeric_columns:

    X_all[col] = pd.to_numeric(
        X_all[col],
        errors="coerce"
    )

    X_all[col] = X_all[col].fillna(
        X[col].median()
    )


# Encode

X_all_encoded = preprocessor.transform(
    X_all
)


# Predict

all_predictions = model.predict(
    X_all_encoded
)


all_probabilities = model.predict_proba(
    X_all_encoded
)


# Convert numbers back to labels

grid[
    "predicted_recharge_priority"
] = [

    number_to_label[
        int(value)
    ]

    for value in all_predictions

]


# Confidence

grid[
    "prediction_confidence"
] = (
    all_probabilities.max(
        axis=1
    )
)


# ============================================================
# STEP 29 — SAVE FINAL PREDICTION GEOJSON
# ============================================================

print("\n")
print("============================================================")
print("STEP 29 - SAVING FINAL PREDICTION")
print("============================================================")


# Convert to WGS84 for web/GIS use

prediction_map = grid.to_crs(
    epsg=4326
)


prediction_file = os.path.join(
    OUTPUT,
    "xgboost_prediction.geojson"
)


prediction_map.to_file(
    prediction_file,
    driver="GeoJSON"
)


print(
    "\nFinal prediction saved:"
)

print(
    prediction_file
)


# ============================================================
# STEP 30 — PREDICTION DISTRIBUTION
# ============================================================

print("\n")
print("============================================================")
print("STEP 30 - PREDICTION DISTRIBUTION")
print("============================================================")


prediction_counts = (
    grid[
        "predicted_recharge_priority"
    ]
    .value_counts()
)


print(
    prediction_counts
)


# Save CSV

prediction_summary = (
    prediction_counts
    .rename_axis(
        "Recharge Class"
    )
    .reset_index(
        name="Grid Cells"
    )
)


prediction_summary[
    "Percentage"
] = (

    prediction_summary[
        "Grid Cells"
    ]
    /
    len(grid)
    *
    100

).round(2)


prediction_summary.to_csv(
    os.path.join(
        OUTPUT,
        "prediction_summary.csv"
    ),
    index=False
)


# ============================================================
# STEP 31 — XGBOOST PREDICTION MAP
# ============================================================

print("\n")
print("============================================================")
print("STEP 31 - CREATING PREDICTION MAP")
print("============================================================")


fig, ax = plt.subplots(
    figsize=(12, 10)
)


grid.plot(

    ax=ax,

    column="predicted_recharge_priority",

    categorical=True,

    legend=True,

    edgecolor="black",

    linewidth=0.15

)


ax.set_title(
    "XGBoost Recharge Potential Prediction - Thane District",
    fontsize=16
)


ax.set_axis_off()


plt.tight_layout()


prediction_image = os.path.join(
    OUTPUT,
    "xgboost_prediction_map.png"
)


plt.savefig(
    prediction_image,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nPrediction map saved:"
)

print(
    prediction_image
)


# ============================================================
# STEP 32 — FEATURE IMPORTANCE
# ============================================================

print("\n")
print("============================================================")
print("STEP 32 - FEATURE IMPORTANCE")
print("============================================================")


feature_names = (
    preprocessor
    .get_feature_names_out()
)


importance = pd.DataFrame({

    "feature":
    feature_names,

    "importance":
    model.feature_importances_

})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    "\nTop 20 important features:"
)


print(
    importance.head(20)
)


importance.to_csv(
    os.path.join(
        OUTPUT,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# STEP 33 — SAVE MODEL
# ============================================================

print("\n")
print("============================================================")
print("STEP 33 - SAVING MODEL")
print("============================================================")


import joblib


joblib.dump(
    model,
    os.path.join(
        OUTPUT,
        "xgboost_model.pkl"
    )
)


joblib.dump(
    preprocessor,
    os.path.join(
        OUTPUT,
        "xgboost_preprocessor.pkl"
    )
)


print(
    "\nModel saved successfully."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("============================================================")
print("                 PHASE 4 COMPLETED")
print("============================================================")


print(
    "\nStudy Area: Thane District"
)


print(
    "Grid Cells:",
    len(grid)
)


print(
    "Training Cells:",
    len(X_train)
)


print(
    "Testing Cells:",
    len(X_test)
)


print(
    "XGBoost Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "\nPrediction distribution:"
)


print(
    prediction_summary
)


print("\n")
print("OUTPUT FILES:")
print("----------------------------------------")


for filename in sorted(
    os.listdir(OUTPUT)
):

    print(
        filename
    )


print("\n")
print("============================================================")
print(" PHASE 4 SUCCESSFULLY COMPLETED")
print("============================================================")