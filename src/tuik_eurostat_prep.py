"""
tuik_energy_pipeline.py
Cleans TUIK population Excel files, gets province features,
and prepares demand values for the optimization.
"""

import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Sector name mapping from raw TUIK labels to short names
SECTOR_MAP = {
    '1. (Mesken)': 'Mesken',
    '2. (Ticaret Ve Kamu Hizmetleri)': 'Ticaret_Kamu',
    '3. (Sanayi)': 'Sanayi',
    '4. (Tarımsal Faaliyetler)': 'Tarimsal',
    '5. (Aydınlatma)': 'Aydinlatma',
}

# =====================================================================
# 1. TUIK ELECTRICITY USAGE (REQUIRED)
# =====================================================================

def parse_tuik_electricity(filepath):
    """Read and clean the pipe-separated TUIK electricity usage CSV file."""
    print(f"  Reading: {os.path.basename(filepath)}")
    # Read the raw file without guessing headers
    df_raw_all = pd.read_csv(filepath, sep='|', encoding='utf-8-sig', header=None, dtype=str)

    # Find the row that has province names by looking for 'Adana'
    header_row = None
    for i in range(min(10, len(df_raw_all))):
        row_str = ' '.join([str(v) for v in df_raw_all.iloc[i].values if pd.notna(v)])
        if 'Adana' in row_str:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not locate province header row in {filepath}")

    # Set that row as column names and remove rows above it
    df_raw = df_raw_all.iloc[header_row + 1:].copy()
    df_raw.columns = df_raw_all.iloc[header_row].values
    df_raw = df_raw.reset_index(drop=True)

    raw_cols = df_raw.columns.tolist()
    province_cols_raw = raw_cols[3:]

    province_names = []
    for col in province_cols_raw:
        col = str(col).strip()
        # Skip empty or unnamed columns
        if not col or col.lower() in ('unnamed', 'nan', 'none') or col.lower().startswith('unnamed'):
            continue
        # Real province names look like 'Adana-1' with a number after the dash
        if '-' not in col:
            continue
        name_part, plate_part = col.rsplit('-', 1)
        if not plate_part.strip().isdigit():
            continue
        province_names.append(name_part.strip())

    if len(province_names) != 81:
        print(f"  [WARNING] Found {len(province_names)} provinces (expected 81)")

    new_cols = ['Metrik', 'Sektor', 'Yil'] + province_names
    while len(new_cols) < len(raw_cols):
        new_cols.append(f'_drop_{len(new_cols)}')
    df_raw.columns = new_cols[:len(raw_cols)]

    df = df_raw.copy()
    df['Yil_num'] = pd.to_numeric(df['Yil'], errors='coerce')
    df = df[df['Yil_num'].notna()].copy()
    df['Yil'] = df['Yil_num'].astype(int)
    df = df.drop(columns=['Yil_num'])
    df['Sektor'] = df['Sektor'].ffill()
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith('_drop_') or c == 'Metrik'])

    df_long = df.melt(
        id_vars=['Sektor', 'Yil'],
        value_vars=province_names,
        var_name='Il',
        value_name='Tuketim_MWh',
    )
    df_long['Tuketim_MWh'] = pd.to_numeric(df_long['Tuketim_MWh'], errors='coerce')
    df_long = df_long.dropna(subset=['Tuketim_MWh'])
    df_long['Sektor'] = df_long['Sektor'].map(SECTOR_MAP).fillna(df_long['Sektor'])

    print(f"  [OK] Parsed {len(province_names)} provinces, "
          f"{df_long['Yil'].nunique()} years, "
          f"{df_long['Sektor'].nunique()} sectors")
    return df_long


def build_demand_tables(df_long):
    """Get total demand for each province and year, plus sector breakdown."""
    df_total = (df_long.groupby(['Yil', 'Il'])['Tuketim_MWh']
                .sum()
                .reset_index()
                .rename(columns={'Tuketim_MWh': 'Toplam_Tuketim_MWh'}))
    return df_total, df_long


# =====================================================================
# 2. TUIK POPULATION AND DENSITY (REQUIRED for upper bounds)
# =====================================================================

def parse_tuik_population_xls(filepath):
    """
    Read population and density data from the TUIK Excel workbook.

    The file has metadata in the first rows, then province data below.
    We skip the total row and get the latest year's numbers.

    Returns: province, population, density for the most recent year.
    """
    engine = 'xlrd' if filepath.lower().endswith('.xls') else 'openpyxl'
    df_raw = pd.read_excel(filepath, engine=engine, sheet_name=0, header=None)

    df = df_raw.iloc[4:, [0, 1, 2, 13]].copy()
    df.columns = ['year', 'province', 'population', 'density']

    df['year'] = df['year'].ffill()
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year

    df = df.dropna(subset=['province'])
    df['province'] = df['province'].astype(str).str.strip()
    df = df[df['province'] != '']
    df = df[df['province'] != 'Toplam-Total']

    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df['density'] = pd.to_numeric(df['density'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['population', 'density', 'year'])
    df['year'] = df['year'].astype(int)

    latest = int(df['year'].max())
    df_latest = df[df['year'] == latest][['province', 'population', 'density']]
    print(f"  [OK] Population and density for {len(df_latest)} provinces (year {latest})")
    return df_latest.reset_index(drop=True)


# =====================================================================
# 3. EUROSTAT (OPTIONAL)
# =====================================================================

def parse_eurostat(filepath):
    """Read Eurostat renewable energy CSV and get Country, Year, Share columns."""
    df = pd.read_csv(filepath)

    # Eurostat CSV files have both a country code and a full name column.
    # We keep the readable name and the year/value.
    keep_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == 'geopolitical entity (reporting)' or cl == 'geo':
            keep_map[c] = 'Country'
        elif cl == 'time_period' or cl == 'time':
            keep_map[c] = 'Yil'
        elif cl == 'obs_value':
            keep_map[c] = 'RenewableShare_Pct'
        elif cl == 'nrg_bal':
            keep_map[c] = 'Indicator'

    # If there are two country columns, keep the readable one
    cols_found = list(keep_map.values())
    if cols_found.count('Country') > 1:
        # Remove the short code, keep the full country name
        for c, target in list(keep_map.items()):
            if target == 'Country' and c.strip().lower() == 'geo':
                del keep_map[c]
    if cols_found.count('Yil') > 1:
        for c, target in list(keep_map.items()):
            if target == 'Yil' and c.strip().lower() == 'time':
                del keep_map[c]
    if cols_found.count('RenewableShare_Pct') > 1:
    # If both got mapped, keep the numeric value column
        for c, target in list(keep_map.items()):
            if target == 'RenewableShare_Pct' and c.strip().lower() != 'obs_value':
                del keep_map[c]
    if cols_found.count('Indicator') > 1:
        for c, target in list(keep_map.items()):
            if target == 'Indicator' and c.strip().lower() != 'nrg_bal':
                del keep_map[c]

    df = df[list(keep_map.keys())].rename(columns=keep_map)
    df['Yil'] = pd.to_numeric(df['Yil'], errors='coerce').astype('Int64')
    df['RenewableShare_Pct'] = pd.to_numeric(df['RenewableShare_Pct'], errors='coerce')
    df = df.dropna(subset=['Yil', 'RenewableShare_Pct'])
    df = df.sort_values(['Country', 'Yil']).reset_index(drop=True)
    print(f"  [OK] Eurostat: {df['Country'].nunique()} countries, "
          f"{df['Yil'].min()}-{df['Yil'].max()}")
    return df

# =====================================================================
# 4. PROVINCE FEATURES
# =====================================================================

def build_province_features(df_total, df_pop, reference_year=2024):
    """Build one row per province with demand and density class."""
    available_years = df_total['Yil'].unique()
    if reference_year not in available_years:
        reference_year = int(max(available_years))
        print(f"  Using fallback reference year: {reference_year}")

    df_ref = df_total[df_total['Yil'] == reference_year].copy()
    df_ref = df_ref.rename(columns={
        'Toplam_Tuketim_MWh': 'total_demand_mwh',
        'Il': 'province'
    })
    df_ref['demand_share'] = df_ref['total_demand_mwh'] / df_ref['total_demand_mwh'].sum()
    df_ref['reference_year'] = reference_year
    df_ref = df_ref.merge(df_pop, on='province', how='left')

    # Split provinces into density groups (used for capacity limits)
    signal = df_ref['density']
    q33 = signal.quantile(0.33)
    q67 = signal.quantile(0.67)

    def classify(v):
        if pd.isna(v):
            return 'medium'
        if v > q67:
            return 'high'    # crowded, lower capacity limit
        elif v > q33:
            return 'medium'
        else:
            return 'low'     # more space, higher capacity limit

    df_ref['intensity_class'] = signal.apply(classify)
    print(f"  [OK] Intensity classification from density")
    return df_ref


def add_geographic_regions(df_features):
    """Add a 'region' column for the 7 regions of Turkey."""
    region_map = {
        'Marmara': ['İstanbul', 'Tekirdağ', 'Edirne', 'Kırklareli', 'Bursa',
                    'Balıkesir', 'Çanakkale', 'Yalova', 'Kocaeli', 'Sakarya',
                    'Bolu', 'Bilecik', 'Düzce'],
        'Ege': ['İzmir', 'Manisa', 'Aydın', 'Denizli', 'Muğla', 'Uşak',
                'Afyonkarahisar', 'Kütahya'],
        'Akdeniz': ['Antalya', 'Isparta', 'Burdur', 'Mersin', 'Adana',
                    'Osmaniye', 'Hatay', 'Kahramanmaraş'],
        'IcAnadolu': ['Ankara', 'Eskişehir', 'Kayseri', 'Konya', 'Karaman',
                      'Aksaray', 'Niğde', 'Nevşehir', 'Kırıkkale', 'Kırşehir',
                      'Çankırı', 'Yozgat', 'Sivas'],
        'Karadeniz': ['Samsun', 'Trabzon', 'Rize', 'Artvin', 'Giresun', 'Ordu',
                      'Sinop', 'Kastamonu', 'Bartın', 'Zonguldak', 'Karabük',
                      'Amasya', 'Tokat', 'Çorum', 'Gümüşhane', 'Bayburt'],
        'DoguAnadolu': ['Erzurum', 'Erzincan', 'Kars', 'Ağrı', 'Iğdır',
                        'Ardahan', 'Elazığ', 'Malatya', 'Bingöl', 'Muş',
                        'Bitlis', 'Van', 'Hakkari', 'Tunceli'],
        'GuneydoguAnadolu': ['Diyarbakır', 'Gaziantep', 'Şanlıurfa', 'Adıyaman',
                             'Mardin', 'Siirt', 'Batman', 'Şırnak', 'Kilis'],
    }
    il_to_region = {il: r for r, iller in region_map.items() for il in iller}
    df_features['region'] = df_features['province'].map(il_to_region)
    unmapped = df_features[df_features['region'].isna()]['province'].tolist()
    if unmapped:
        print(f"  [WARNING] Unmapped provinces: {unmapped}")
    return df_features

def prepare_tuik_data(reference_year=2024):
    """Main function called from main.py. Builds all processed data tables."""
    print("=" * 60)
    print("TUIK + EUROSTAT DATA PIPELINE")
    print("=" * 60)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # [1/3] Electricity data (required)
    print("\n[1/3] TUIK electricity usage...")
    elec_candidates = [
        os.path.join(DATA_DIR, "raw", "tuik_elektrik_tuketim.csv"),
    ]
    elec_path = next((p for p in elec_candidates if os.path.exists(p)), None)
    if elec_path is None:
        print(f"[ERROR] TUIK electricity CSV not found in data/raw/")
        return None
    df_long = parse_tuik_electricity(elec_path)
    df_total, df_sectoral = build_demand_tables(df_long)
    df_total.to_csv(os.path.join(PROCESSED_DIR, "demand_by_province.csv"), index=False)
    df_sectoral.to_csv(os.path.join(PROCESSED_DIR, "demand_by_province_sector.csv"), index=False)

    # [2/3] Population and density (required)
    print("\n[2/3] TUIK population and density...")
    pop_candidates = [
        os.path.join(DATA_DIR, "raw", "tuik_nufus_yogunluk.xls"),
        os.path.join(DATA_DIR, "raw", "tuik_nufus_yogunluk.xlsx"),
    ]
    pop_path = next((p for p in pop_candidates if os.path.exists(p)), None)
    if pop_path is None:
        print(f"[ERROR] Population .xls file not found in data/raw/")
        return None
    df_pop = parse_tuik_population_xls(pop_path)

    # Build province features by combining electricity and density data
    df_features = build_province_features(df_total, df_pop, reference_year=reference_year)
    df_features = add_geographic_regions(df_features)
    df_features.to_csv(os.path.join(PROCESSED_DIR, "province_features.csv"), index=False)

    # [3/3] Eurostat data (optional)
    print("\n[3/3] Eurostat renewable share...")
    eurostat_candidates = [
    os.path.join(DATA_DIR, "raw", "eurostat_ren.csv"),
    os.path.join(DATA_DIR, "raw", "nrg_ind_ren_linear_2_0.csv"),
    os.path.join(DATA_DIR, "raw", "nrg_ind_ren.csv"),
    os.path.join(DATA_DIR, "external", "eurostat_ren.csv"),
    os.path.join(DATA_DIR, "external", "nrg_ind_ren_linear_2_0.csv"),
    os.path.join(DATA_DIR, "external", "nrg_ind_ren.csv"),
    ]
    eurostat_path = next((p for p in eurostat_candidates if os.path.exists(p)), None)
    if eurostat_path is not None:
        df_eur = parse_eurostat(eurostat_path)
        df_eur.to_csv(os.path.join(PROCESSED_DIR, "eurostat_renewable_share.csv"), index=False)
    else:
        print("  Skipped (no Eurostat file)")

    print(f"\nSaved to {PROCESSED_DIR}/")
    print(f"   - province_features.csv         ({df_features.shape})")
    print(f"   - demand_by_province.csv        ({df_total.shape})")
    print("=" * 60)
    return df_features


if __name__ == "__main__":
    prepare_tuik_data()
    