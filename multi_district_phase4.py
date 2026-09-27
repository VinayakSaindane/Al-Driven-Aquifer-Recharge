import os, re, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

BASE = r'D:\Thane_Recharge_Project'
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

DISTRICTS = ['Thane', 'Palghar', 'Raigarh', 'Nashik']
CRS = 'EPSG:32643'
CELL_SIZE = 1000  # 1 km, same scale used for the Thane workflow


def load(name):
    return gpd.read_file(os.path.join(DATA, name))


def district_col(g):
    for c in ['DTENAME', 'dtename']:
        if c in g.columns:
            return c
    return None


def subset_district(g, district):
    c = district_col(g)
    if c:
        return g[g[c].astype(str).str.strip().str.lower() == district.lower()].copy()
    return g.copy()


def make_grid(boundary, size=CELL_SIZE):
    minx, miny, maxx, maxy = boundary.total_bounds
    xs = np.arange(np.floor(minx/size)*size, np.ceil(maxx/size)*size + size, size)
    ys = np.arange(np.floor(miny/size)*size, np.ceil(maxy/size)*size + size, size)
    cells = []
    for x in xs[:-1]:
        for y in ys[:-1]:
            cells.append(box(x, y, x+size, y+size))
    g = gpd.GeoDataFrame({'geometry': cells}, crs=boundary.crs)
    g = gpd.overlay(g, boundary[['geometry']], how='intersection')
    g = g[g.geometry.area > 0].copy()
    g['cell_id'] = np.arange(len(g))
    return g


def dominant_attribute(grid, layer, attribute, prefix):
    if layer.empty:
        grid[prefix] = np.nan
        return grid
    cols = ['cell_id', 'geometry']
    lyr = layer[[attribute, 'geometry']].copy()
    lyr = lyr[lyr.geometry.notna() & ~lyr.geometry.is_empty]
    candidates = gpd.sjoin(grid[cols], lyr, how='inner', predicate='intersects')
    if candidates.empty:
        grid[prefix] = np.nan
        return grid
    left = candidates['cell_id'].to_numpy()
    right = candidates['index_right'].to_numpy()
    geoms = grid.set_index('cell_id').geometry
    rgeoms = lyr.geometry
    vals = []
    areas = []
    for cid, ridx in zip(left, right):
        try:
            inter = geoms.loc[cid].intersection(rgeoms.loc[ridx])
            a = inter.area
        except Exception:
            a = 0
        vals.append((cid, lyr.loc[ridx, attribute], a))
    d = pd.DataFrame(vals, columns=['cell_id', 'value', 'area'])
    d = d.sort_values(['cell_id', 'area'], ascending=[True, False]).drop_duplicates('cell_id')
    grid = grid.merge(d[['cell_id', 'value']], on='cell_id', how='left')
    grid.rename(columns={'value': prefix}, inplace=True)
    return grid


def length_density(grid, layer, prefix):
    if layer.empty:
        grid[prefix] = 0.0
        return grid
    lyr = layer[['geometry']].copy()
    candidates = gpd.sjoin(grid[['cell_id','geometry']], lyr, how='inner', predicate='intersects')
    totals = {}
    if not candidates.empty:
        geoms = grid.set_index('cell_id').geometry
        rgeoms = lyr.geometry
        for cid, ridx in zip(candidates['cell_id'], candidates['index_right']):
            try:
                inter = geoms.loc[cid].intersection(rgeoms.loc[ridx])
                length = inter.length
            except Exception:
                length = 0
            totals[cid] = totals.get(cid, 0) + length
    grid[prefix] = grid['cell_id'].map(totals).fillna(0) / 1000.0  # km per km² cell
    return grid


def area_coverage(grid, layer, prefix):
    if layer.empty:
        grid[prefix] = 0.0
        return grid
    lyr = layer[['geometry']].copy()
    candidates = gpd.sjoin(grid[['cell_id','geometry']], lyr, how='inner', predicate='intersects')
    totals = {}
    if not candidates.empty:
        geoms = grid.set_index('cell_id').geometry
        rgeoms = lyr.geometry
        for cid, ridx in zip(candidates['cell_id'], candidates['index_right']):
            try:
                inter = geoms.loc[cid].intersection(rgeoms.loc[ridx])
                a = inter.area
            except Exception:
                a = 0
            totals[cid] = totals.get(cid, 0) + a
    cell_area = grid.geometry.area
    grid[prefix] = grid['cell_id'].map(totals).fillna(0) / cell_area
    return grid


def slope_to_numeric(v):
    if pd.isna(v): return np.nan
    nums = re.findall(r'\d+(?:\.\d+)?', str(v))
    if not nums: return np.nan
    nums = [float(x) for x in nums]
    return sum(nums[:2])/len(nums[:2])

print('\n=== MULTI-DISTRICT AQUIFER RECHARGE PIPELINE ===')
print('Districts:', ', '.join(DISTRICTS))

admin = load('admin_boundary.geojson').to_crs(CRS)
admin4 = admin[admin['DTENAME'].isin(DISTRICTS)].copy()
if admin4.empty:
    raise RuntimeError('Requested districts were not found in admin_boundary.geojson')

# Keep district boundaries separate for website filtering and per-district grid IDs.
admin4 = admin4[['DTENAME','geometry']].dissolve('DTENAME', as_index=False)

layers = {name: load(name).to_crs(CRS) for name in [
    'recharge_priority.geojson','slope.geojson','soil.geojson','lithology.geojson',
    'geomorphology.geojson','lulc.geojson','lineaments.geojson','drainage_soi.geojson',
    'recharge_structure.geojson','groundwater_prospect.geojson'
]}

all_grids = []

for district in DISTRICTS:
    print(f'\nProcessing {district}...')
    bnd = admin4[admin4['DTENAME'] == district]
    grid = make_grid(bnd)
    grid['district'] = district
    grid['district_cell_id'] = np.arange(len(grid))
    print('  grid cells:', len(grid))

    # Filter layers using their district attribute where available; soil is spatially clipped.
    rp = subset_district(layers['recharge_priority.geojson'], district)
    sl = subset_district(layers['slope.geojson'], district)
    li = subset_district(layers['lithology.geojson'], district)
    gm = subset_district(layers['geomorphology.geojson'], district)
    lu = subset_district(layers['lulc.geojson'], district)
    ln = subset_district(layers['lineaments.geojson'], district)
    dr = subset_district(layers['drainage_soi.geojson'], district)
    rs = subset_district(layers['recharge_structure.geojson'], district)
    gw = subset_district(layers['groundwater_prospect.geojson'], district)
    so = gpd.clip(layers['soil.geojson'], bnd)

    grid = dominant_attribute(grid, rp, 'Recharge_priority', 'Recharge_priority')
    grid = dominant_attribute(grid, sl, 'SLP_ZONE', 'slope_zone')
    grid = dominant_attribute(grid, so, 'TEXTURE', 'soil')
    grid = dominant_attribute(grid, li, 'Rock_Type', 'lithology')
    grid = dominant_attribute(grid, gm, 'GEOM_L2_', 'geomorphology')
    grid = dominant_attribute(grid, lu, 'LU_15_16_L2', 'lulc')
    grid = dominant_attribute(grid, gw, 'GWP_Zones', 'groundwater_prospect')
    grid = length_density(grid, ln, 'lineament_density')
    grid = length_density(grid, dr, 'drainage_density')
    grid = area_coverage(grid, rs, 'recharge_structure_coverage')

    grid['slope_numeric'] = grid['slope_zone'].apply(slope_to_numeric)
    grid['cell_area_km2'] = grid.geometry.area / 1_000_000
    all_grids.append(grid)

master = pd.concat(all_grids, ignore_index=True)
master = gpd.GeoDataFrame(master, geometry='geometry', crs=CRS)
master['global_cell_id'] = np.arange(len(master))

print('\nTotal grid cells:', len(master))
print('\nRecharge labels by district:')
print(pd.crosstab(master['district'], master['Recharge_priority'], dropna=False))

# Save GIS/ML-ready master dataset.
master.to_crs('EPSG:4326').to_file(os.path.join(OUT,'multi_district_master_grid.geojson'), driver='GeoJSON')
master.drop(columns='geometry').to_csv(os.path.join(OUT,'multi_district_ml_dataset.csv'), index=False)

# ML dataset: only cells with a recharge-priority target.
ml = master[master['Recharge_priority'].notna()].copy()
features = [
    'slope_numeric','groundwater_prospect','soil','lithology','geomorphology','lulc',
    'lineament_density','drainage_density','recharge_structure_coverage'
]
features = [f for f in features if f in ml.columns]
target = 'Recharge_priority'

X = ml[features].copy()
y_raw = ml[target].astype(str)

num_features = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
cat_features = [f for f in features if f not in num_features]

pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

le = LabelEncoder()
y = le.fit_transform(y_raw)

if len(np.unique(y)) < 2:
    raise RuntimeError('Not enough target classes for XGBoost training.')

# Stratification only when every class has at least 2 samples.
counts = pd.Series(y).value_counts()
stratify = y if counts.min() >= 2 else None
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, ml.index, test_size=0.20, random_state=42, stratify=stratify
)

model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist'
)

pipe = Pipeline([('preprocess', pre), ('model', model)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='weighted')
print('\nAccuracy:', round(acc*100,2), '%')
print('Weighted F1:', round(f1*100,2), '%')
print('\nClassification report:')
print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_test, pred)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(os.path.join(OUT,'multi_district_confusion_matrix.csv'))
with open(os.path.join(OUT,'multi_district_metrics.json'),'w') as f:
    json.dump({'districts':DISTRICTS,'grid_cells':int(len(master)),'labeled_cells':int(len(ml)),
               'accuracy':float(acc),'weighted_f1':float(f1),'classes':le.classes_.tolist()}, f, indent=2)

# Predict all cells.
master_X = master[features].copy()
master['predicted_recharge_priority'] = le.inverse_transform(pipe.predict(master_X))
master['prediction_confidence'] = pipe.predict_proba(master_X).max(axis=1)

pred_out = master.to_crs('EPSG:4326')
pred_out.to_file(os.path.join(OUT,'multi_district_recharge_predictions.geojson'), driver='GeoJSON')

# Prediction summary for website cards/charts.
summary = pred_out.groupby(['district','predicted_recharge_priority']).size().reset_index(name='cells')
summary.to_csv(os.path.join(OUT,'prediction_summary_by_district.csv'), index=False)

# Save model bundle.
joblib.dump({'pipeline':pipe,'label_encoder':le,'features':features,'districts':DISTRICTS},
            os.path.join(OUT,'multi_district_xgboost_model.joblib'))

# Quick map.
fig, ax = plt.subplots(figsize=(10,9))
pred_out.plot(column='predicted_recharge_priority', ax=ax, legend=True, categorical=True)
admin4.to_crs('EPSG:4326').boundary.plot(ax=ax, linewidth=1)
ax.set_title('AI-Driven Aquifer Recharge Potential - Thane, Palghar, Raigad, Nashik')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'multi_district_prediction_map.png'), dpi=220)
plt.close()

print('\nPrediction distribution:')
print(pred_out.groupby(['district','predicted_recharge_priority']).size())
print('\nDONE. Outputs saved in:', OUT)
