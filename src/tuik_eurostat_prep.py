"""
tuik_energy_pipeline.py
Pipeline step: Cleans messy TUIK population Excel tables (.xlsx/.xls), 
extracts province features, and prepares dynamic demand scaling values.
"""

import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# YENİ EKLE
SECTOR_MAP = {
    '1. (Mesken)': 'Mesken',
    '2. (Ticaret Ve Kamu Hizmetleri)': 'Ticaret_Kamu',
    '3. (Sanayi)': 'Sanayi',
    '4. (Tarımsal Faaliyetler)': 'Tarimsal',
    '5. (Aydınlatma)': 'Aydinlatma',
}

# =====================================================================
# 1. TÜİK ELECTRICITY CONSUMPTION (REQUIRED)
# =====================================================================

def parse_tuik_electricity(filepath):
    """Parse messy pipe-separated TÜİK electricity consumption CSV."""
    print(f"  Reading: {os.path.basename(filepath)}")
    # Read raw without auto-detecting headers
    df_raw_all = pd.read_csv(filepath, sep='|', encoding='utf-8-sig', header=None, dtype=str)

    # Find which row contains the province names (look for 'Satırlar' marker or 'Adana')
    header_row = None
    for i in range(min(10, len(df_raw_all))):
        row_str = ' '.join([str(v) for v in df_raw_all.iloc[i].values if pd.notna(v)])
        if 'Adana' in row_str:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not locate province header row in {filepath}")

    # Use that row as the column names, drop everything above it
    df_raw = df_raw_all.iloc[header_row + 1:].copy()
    df_raw.columns = df_raw_all.iloc[header_row].values
    df_raw = df_raw.reset_index(drop=True)

    raw_cols = df_raw.columns.tolist()
    province_cols_raw = raw_cols[3:]

    province_names = []
    for col in province_cols_raw:
        col = str(col).strip()
        # Skip empty, unnamed, NaN, and entries without a plate-number suffix
        if not col or col.lower() in ('unnamed', 'nan', 'none') or col.lower().startswith('unnamed'):
            continue
        # Real province entries have format 'Adana-1' — must contain a hyphen with digits
        if '-' not in col:
            continue
        name_part, plate_part = col.rsplit('-', 1)
        if not plate_part.strip().isdigit():
            continue
        province_names.append(name_part.strip())

    if len(province_names) != 81:
        print(f"  ⚠ Warning: Found {len(province_names)} provinces (expected 81)")

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

    print(f"  ✓ Parsed {len(province_names)} provinces, "
          f"{df_long['Yil'].nunique()} years, "
          f"{df_long['Sektor'].nunique()} sectors")
    return df_long


def build_demand_tables(df_long):
    """Total demand per province×year, and the sectoral breakdown."""
    df_total = (df_long.groupby(['Yil', 'Il'])['Tuketim_MWh']
                .sum()
                .reset_index()
                .rename(columns={'Tuketim_MWh': 'Toplam_Tuketim_MWh'}))
    return df_total, df_long


# =====================================================================
# 2. TÜİK POPULATION & DENSITY (REQUIRED for upper bounds)
# =====================================================================

def parse_tuik_population_xls(filepath):
    """
    Parse the TÜİK 'İl ve Cinsiyete Göre İl/İlçe Merkezi, Belde/Köy Nüfusu
    ve Nüfus Yoğunluğu' workbook (.xls format from veriportali.tuik.gov.tr).

    Structure:
      Rows 0-3 : metadata + multi-row header
      Row 4    : 'Toplam-Total' (Turkey-wide aggregate — skipped)
      Rows 5+  : per-province data, alphabetic order, with year forward-fill
      Columns  : A=Year, B=Province, C=Total Pop, ..., N=Density (people/km²)

    Returns latest year's snapshot: province, population, density.
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
    print(f"  ✓ Population & density for {len(df_latest)} provinces (year {latest})")
    return df_latest.reset_index(drop=True)


# =====================================================================
# 3. EUROSTAT (OPTIONAL)
# =====================================================================

def parse_eurostat(filepath):
    """Parse Eurostat nrg_ind_ren CSV → Country, Yil, RenewableShare_Pct."""
    df = pd.read_csv(filepath)

    # Eurostat now ships dual-column CSVs (both 'geo' code and 'Geopolitical entity'
    # name). We keep the human-readable name and the year/value.
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

    # Prefer human-readable Country column if both exist
    cols_found = list(keep_map.values())
    if cols_found.count('Country') > 1:
        # Drop the 'geo' code, keep 'Geopolitical entity (reporting)'
        for c, target in list(keep_map.items()):
            if target == 'Country' and c.strip().lower() == 'geo':
                del keep_map[c]
    if cols_found.count('Yil') > 1:
        for c, target in list(keep_map.items()):
            if target == 'Yil' and c.strip().lower() == 'time':
                del keep_map[c]
    if cols_found.count('RenewableShare_Pct') > 1:
    # If somehow both got mapped, prefer the OBS_VALUE (numeric) over the description
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
    print(f"  ✓ Eurostat: {df['Country'].nunique()} countries, "
          f"{df['Yil'].min()}–{df['Yil'].max()}")
    return df

# =====================================================================
# 4. PROVINCE FEATURES
# =====================================================================

def build_province_features(df_total, df_pop, reference_year=2024):
    """One-row-per-province feature table with intensity_class from density."""
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

    # Intensity class from density (used by optimization upper bounds)
    signal = df_ref['density']
    q33 = signal.quantile(0.33)
    q67 = signal.quantile(0.67)

    def classify(v):
        if pd.isna(v):
            return 'medium'
        if v > q67:
            return 'high'    # crowded → tight cap
        elif v > q33:
            return 'medium'
        else:
            return 'low'     # spacious → loose cap

    df_ref['intensity_class'] = signal.apply(classify)
    print(f"  ✓ Intensity classification from density")
    return df_ref


def add_geographic_regions(df_features):
    """Add 'region' column (7 geographic regions of Turkey)."""
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
        print(f"  ⚠ Unmapped provinces: {unmapped}")
    return df_features

def prepare_tuik_data(reference_year=2024):
    """Called from main.py. Builds all processed feature tables."""
    print("=" * 60)
    print("🚀 TÜİK + EUROSTAT DATA PIPELINE")
    print("=" * 60)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # [1/3] Electricity (REQUIRED)
    print("\n[1/3] TÜİK electricity consumption…")
    elec_candidates = [
        os.path.join(DATA_DIR, "raw", "tuik_elektrik_tuketim.csv"),
    ]
    elec_path = next((p for p in elec_candidates if os.path.exists(p)), None)
    if elec_path is None:
        print(f"❌ ERROR: TÜİK electricity CSV not found in data/raw/")
        return None
    df_long = parse_tuik_electricity(elec_path)
    df_total, df_sectoral = build_demand_tables(df_long)
    df_total.to_csv(os.path.join(PROCESSED_DIR, "demand_by_province.csv"), index=False)
    df_sectoral.to_csv(os.path.join(PROCESSED_DIR, "demand_by_province_sector.csv"), index=False)

    # [2/3] Population & density (REQUIRED)
    print("\n[2/3] TÜİK population & density…")
    pop_candidates = [
        os.path.join(DATA_DIR, "raw", "tuik_nufus_yogunluk.xls"),
        os.path.join(DATA_DIR, "raw", "tuik_nufus_yogunluk.xlsx"),
    ]
    pop_path = next((p for p in pop_candidates if os.path.exists(p)), None)
    if pop_path is None:
        print(f"❌ ERROR: Population .xls file not found in data/raw/")
        return None
    df_pop = parse_tuik_population_xls(pop_path)

    # Province features (combines electricity + density)
    df_features = build_province_features(df_total, df_pop, reference_year=reference_year)
    df_features = add_geographic_regions(df_features)
    df_features.to_csv(os.path.join(PROCESSED_DIR, "province_features.csv"), index=False)

    # [3/3] Eurostat (OPTIONAL)
    print("\n[3/3] Eurostat renewable share…")
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

    print(f"\n💾 Saved to {PROCESSED_DIR}/")
    print(f"   - province_features.csv         ({df_features.shape})")
    print(f"   - demand_by_province.csv        ({df_total.shape})")
    print("=" * 60)
    return df_features


if __name__ == "__main__":
    prepare_tuik_data()
    