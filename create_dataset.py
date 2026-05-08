from folktables import ACSDataSource
import geopandas as gpd
import time
import pandas as pd

STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
    'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
    'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
    'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
    'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56'
}

survey_year = "2022"
data_source = ACSDataSource(survey_year=survey_year, horizon="1-Year", survey="person")

all_states_data = []

for state, fips in STATE_FIPS.items():
    print(f"\n--- Processing {state} (FIPS: {fips}) ---")

    try:
        df_acs = data_source.get_data(states=[state], download=True)
        print(f"  Loaded ACS Data: {len(df_acs)} rows")

        shapefile_url = f"https://www2.census.gov/geo/tiger/TIGER2022/PUMA/tl_2022_{fips}_puma20.zip"
        print(f"  Downloading shapefile...")
        gdf_puma = gpd.read_file(shapefile_url)

        gdf_puma_projected = gdf_puma.to_crs(epsg=3857)
        centroids = gdf_puma_projected.geometry.centroid.to_crs(epsg=4326)

        gdf_puma['Longitude'] = centroids.x
        gdf_puma['Latitude'] = centroids.y

        gdf_puma['PUMA_ID'] = gdf_puma['PUMACE20'].astype(int)
        df_acs['PUMA_ID'] = df_acs['PUMA'].astype(int)

        puma_coords = gdf_puma[['PUMA_ID', 'Latitude', 'Longitude']]

        df_merged = df_acs.merge(puma_coords, on='PUMA_ID', how='inner')

        df_merged['RACE'] = df_merged['RAC1P']

        all_states_data.append(df_merged)
        print(f"  Successfully merged {len(df_merged)} rows for {state}.")

        time.sleep(1)

    except Exception as e:
        print(f"  Error processing {state}: {e}")

print("\nConcatenating all states...")
final_us_dataset = pd.concat(all_states_data, ignore_index=True)

columns_to_show = ['AGEP', 'SEX', 'RACE', 'PINCP', 'PUMA_ID', 'Latitude', 'Longitude']
print(f"\n=== FINAL DATASET READY: {len(final_us_dataset)} total rows ===")
print(final_us_dataset[columns_to_show].head())
print("\nDataset Info:")
print(final_us_dataset.info())

final_us_dataset.to_csv("us_census_puma_data.csv", index=False)
