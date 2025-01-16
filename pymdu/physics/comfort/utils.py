import os

from datetime import datetime
import geopandas as gpd
import pandas as pd
import psycopg2
import psycopg2.extras
import rioxarray as riox
from rasterio.io import MemoryFile
from shapely import box
from sqlalchemy import create_engine, text as sql_text
from pythermalcomfort.models import utci
import rasterio
from osgeo import gdal
import rioxarray
import numpy as np


def build_connection(
    dbname='Qapeosud', host='10.17.36.192', user='postgres', password='postgres'
):
    conn = psycopg2.connect(
        f'dbname={dbname} host={host} user={user} password={password}'
    )
    return conn


def get_geodata(
    uri='postgresql+psycopg2://postgres:postgres@10.17.36.192:5432/Qapeosud',
):
    df = pd.read_sql_query(
        sql=""" SELECT g.table_schema, g.table_name, g.column_name, g.data_type,g.udt_name,f.type,f.srid
                                    FROM information_schema.columns as g JOIN geometry_columns AS f ON (g.table_schema = f.f_table_schema and g.table_name = f.f_table_name )
                                    WHERE g.udt_name = 'geometry' """,
        con=create_engine(uri, echo=False),
    )
    df['key'] = [x + '.' + y for (x, y) in zip(df['table_schema'], df['table_name'])]
    return df


def get_tables(connection, print_to_console=False):
    """
    Create and return a list of dictionaries with the
    schemas and names of tables in the database
    connected to by the connection argument.
    """

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(
        """SELECT *
                      FROM information_schema.tables
                      WHERE table_schema != 'pg_catalog'
                      AND table_schema != 'information_schema'
                      AND table_type='BASE TABLE'
                      ORDER BY table_schema, table_name"""
    )

    tables = cursor.fetchall()
    cursor.close()
    if print_to_console:
        for row in tables:
            print('{}.{}'.format(row['table_schema'], row['table_name']))
    return tables


def collect_data(
    df,
    uri='postgresql+psycopg2://postgres:postgres@10.17.36.192:5432/Qapeosud',
    schema='Qapeosud_Atlantech_2020',
    table='Building',
):
    uri = uri
    engine = create_engine(uri, echo=False)
    test = f'{schema}.{table}'
    if (df.key == test).any():
        query = sql_text(f'''SELECT * FROM "{schema}"."{table}"''')
        gdf = gpd.read_postgis(sql=query, con=engine.connect(), geom_col='geom')
        gdf = gdf.explode(ignore_index=True)
        try:
            gdf.rename_geometry('geometry', inplace=True)
        except:
            pass
        return gdf
    else:
        df = pd.read_sql_query(
            sql=sql_text(f"""SELECT * FROM {schema}.{table}"""), con=engine.connect()
        )
        df = df.explode(ignore_index=True)
        return df


def get_raster(connection, table_name='public.atlantech_mnt'):
    curs = connection.cursor()
    # Read raster from postgis using ST_AsGDALRaster function
    curs.execute(
        f"""select ST_AsGDALRaster(st_union(rast), 'GTIFF') from {table_name}"""
    )
    # Fetch result
    result = curs.fetchone()
    # Load raster in memory file using MemoryFile module of rasterio
    inMemoryRaster = MemoryFile(bytes(result[0]))
    # Read in memoery raster using rioxarray
    raster_dataset = riox.open_rasterio(inMemoryRaster)

    return raster_dataset


def recuperation_schema_donnees_sql(
    uri='postgresql+psycopg2://postgres:postgres@10.17.36.192:5432/Qapeosud',
    schema='Qapeosud_Atlantech_2020',
):
    uri = uri
    engine = create_engine(uri, echo=False)
    query = sql_text(
        f"""SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = {schema}"""
    )
    gdf = gpd.read_postgis(sql=query, con=engine.connect(), geom_col='geom')
    gdf = gdf.explode(ignore_index=True)
    return gdf


def analyze_footprint(gdf_project, delta_mask=100):
    # creation du mask
    # TODO : en discuter avec BB
    # ================================
    envelope_polygon = gdf_project.envelope.bounds
    bbox = envelope_polygon.values[0]
    delta_mask = 100
    bbox_final = box(
        bbox[0] + delta_mask,
        bbox[1] + delta_mask,
        bbox[2] - delta_mask,
        bbox[3] - delta_mask,
    )

    gdf_project_4326 = gdf_project.to_crs(4326)

    envelope_polygon_4326 = gdf_project_4326.envelope.bounds
    bbox_4326 = envelope_polygon_4326.values[0]
    bbox_final_4326 = box(bbox[0], bbox[1], bbox[2], bbox[3])
    # =======================
    return bbox_final, bbox_4326


def classe_confort(
    data_as_array, interval=[9, 26, 32, 38, 46, 60], values=[0, 1, 2, 3, 4]
):
    """
    ===
    Determination de la classe confort, pour les plots
    ===
    """
    size_x = data_as_array.shape[0]
    size_y = data_as_array.shape[1]
    output = np.zeros(shape=(size_x, size_y))
    for x in range(0, size_x):
        for y in range(0, size_y):
            utci = data_as_array[x][y]
            if utci < interval[1]:
                output[x][y] = values[0] + (utci - interval[0]) / (
                    interval[1] - interval[0]
                )
            elif utci < interval[2]:
                output[x][y] = values[1] + (utci - interval[1]) / (
                    interval[2] - interval[1]
                )
            elif utci < interval[3]:
                output[x][y] = values[2] + (utci - interval[2]) / (
                    interval[3] - interval[2]
                )
            elif utci < interval[4]:
                output[x][y] = values[3] + (utci - interval[3]) / (
                    interval[4] - interval[3]
                )
            else:
                output[x][y] = values[4] + (utci - interval[4]) / (
                    interval[5] - interval[4]
                )
    return output


def calculate_confort_from_solweig_and_urock(
    result_path='./data.results/',
    solweig_tmr_file='Tmrt_1997_157_1000D.tif',
    urock_file='urock_outputWS.Gtiff',
):
    date = solweig_tmr_file.split('Tmrt_')[-1].split('.tif')[0]
    # img_object = GeoTIFF()
    # img_object.reproject_resample_cropped_raster(
    #                 solweig_tmr_file,
    #                 urock_file,
    #                 output_file = urock_file_reproj)

    dataset = rasterio.open(fp=solweig_tmr_file)
    dataset_rio = rioxarray.open_rasterio(filename=solweig_tmr_file)
    new_data = dataset_rio.copy()
    image = dataset.read()

    step1 = gdal.Open(solweig_tmr_file, gdal.GA_ReadOnly)
    GT_input = step1.GetGeoTransform()
    step2 = step1.GetRasterBand(1)
    tmr_as_array = step2.ReadAsArray()
    size1, size2 = tmr_as_array.shape
    output = np.zeros(shape=(size1, size2))

    step1 = gdal.Open(urock_file, gdal.GA_ReadOnly)
    GT_input = step1.GetGeoTransform()
    step2 = step1.GetRasterBand(1)
    wind_as_array = step2.ReadAsArray()

    for i in range(0, size1):
        output[i, :] = utci(
            tdb=25,
            tr=tmr_as_array[i, :],
            v=wind_as_array[i, :],
            rh=75,
            limit_inputs=False,
        )
    new_data.data[0] = output
    new_data.rio.to_raster(os.path.join(result_path, f'UTCI_{date}.tif'))

    dataset = rasterio.open(fp=os.path.join(result_path, f'UTCI_{date}.tif'))
    dataset_rio = rioxarray.open_rasterio(
        filename=os.path.join(result_path, f'UTCI_{date}.tif')
    )
    new_data = dataset_rio.copy()

    step1 = gdal.Open(os.path.join(result_path, f'UTCI_{date}.tif'), gdal.GA_ReadOnly)
    GT_input = step1.GetGeoTransform()
    step2 = step1.GetRasterBand(1)
    utci_as_array = step2.ReadAsArray()
    size1, size2 = tmr_as_array.shape
    output = np.zeros(shape=(size1, size2))

    output = classe_confort(utci_as_array)
    new_data.data[0] = output
    new_data.rio.to_raster(os.path.join(result_path, f'CLASSE_CONFORT_{date}.tif'))


def calculate_confort_from_solweig_and_urock_dataarray(final, pas_de_temps=0, tdb=25):
    tmr_as_array = final['Tmrt'][pas_de_temps, :, :]
    wind_as_array = final['Wind'][pas_de_temps, :, :]
    size1, size2 = tmr_as_array.shape
    output = np.zeros(shape=(size1, size2))
    for i in range(0, size1):
        output[i, :] = utci(
            tdb=tdb,
            tr=tmr_as_array[i, :],
            v=wind_as_array[i, :],
            rh=75,
            limit_inputs=False,
        )

    return output


def time_index_from_filenames(filenames, year='2022', name='Tmrt'):
    liste_of_day = [x.split(f'{name}_1997_')[-1].split('_')[0] for x in filenames]
    liste_of_hour = [
        x.split('.tif')[0].split('_')[-1].split('00')[0].split('00N')[0].split('00D')[0]
        for x in filenames
    ]
    liste_of_hour = ['00' if len(x) == 0 else x for x in liste_of_hour]
    liste_of_hour = [x + '0' if len(x) == 1 else x for x in liste_of_hour]
    liste_of_date = [
        datetime.strptime(year + '-' + day, '%Y-%j') for day in liste_of_day
    ]
    date = [
        res.strftime(f'%Y-%m-%d {hour}:00:00')
        for (res, hour) in zip(liste_of_date, liste_of_hour)
    ]
    return [pd.to_datetime(x).strftime('%Y-%m-%d %X') for x in date]


def trouver_nombre_plus_proche(liste, nombre):
    nombre_plus_proche = None
    difference_min = float('inf')
    for element in liste:
        difference = abs(element - nombre)
        if difference < difference_min:
            difference_min = difference
            nombre_plus_proche = element
    return nombre_plus_proche
