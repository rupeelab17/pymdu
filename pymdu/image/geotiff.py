# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  Copyright                                                                   *
#                                                                              *
#  pymdu is free software: you can redistribute it and/or modify               *
#  it under the terms of the GNU General Public License as published by        *
#  the Free Software Foundation, either version 3 of the License, or           *
#  (at your option) any later version.                                         *
#                                                                              *
#  pymdu is distributed in the hope that it will be useful,                    *
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              *
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
#  GNU General Public License for more details.                                *
#                                                                              *
#  You should have received a copy of the GNU General Public License           *
#  along with pymdu.  If not, see <https://www.gnu.org/licenses/>.             *
# ******************************************************************************
import os
import os.path
import subprocess
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray
from geocube.api.core import make_geocube
from osgeo import ogr, gdal, gdalconst
from osgeo import osr
from osgeo.osr import SpatialReference


def crop_from_image(src_tif: str = None, dst_tif: str = None, crs: int = 2154):
    dataset = rasterio.open(src_tif)
    minx = dataset.bounds[0]
    miny = dataset.bounds[3]
    maxx = dataset.bounds[2]
    maxy = dataset.bounds[1]
    # print([minx, miny, maxx, maxy])

    destination = gdal.Translate(
        src_tif, dst_tif, projWin=[minx, miny, maxx, maxy], outputSRS='EPSG:%s' % crs
    )
    # close the destination image
    destination = None
    dataset = None


def resample_resolution(
    src_tif: str = None, dst_tif: str = None, new_xres: int = 1, new_yres: int = -1
):
    """
    resample resolution the pixel size of the raster.

    Args:
        new_xres (int): desired resolution in x-direction
        new_yres (int): desired resolution in y-direction
        save_path (str): filepath to where the output file should be stored

    Returns: Nothing, it writes a raster file with decreased resolution.
    :param new_yres:
    :param new_xres:

    """
    options = gdal.WarpOptions(options=['tr'], xRes=new_xres, yRes=new_yres)
    newfile = gdal.Warp(
        destNameOrDestDS=src_tif,
        srcDSOrSrcDSTab=dst_tif,
        options=options,
        overwrite=True,
    )
    # newprops = newfile.GetGeoTransform()
    newfile = None


def reproject(
    src_tif: str = None,
    dst_tif: str = None,
    src_epsg: int = 3857,
    dist_epsg: int = 2154,
):
    """
    resample resolution the pixel size of the raster.

    Args:
        new_xres (int): desired resolution in x-direction
        new_yres (int): desired resolution in y-direction
        save_path (str): filepath to where the output file should be stored

    Returns: Nothing, it writes a raster file with decreased resolution.
    :param new_yres:
    :param new_xres:

    """
    options = gdal.WarpOptions(options=['tr'], srcSRS=src_epsg, dstSRS=dist_epsg)
    newfile = gdal.Warp(
        destNameOrDestDS=src_tif,
        srcDSOrSrcDSTab=dst_tif,
        options=options,
        overwrite=True,
    )
    newfile = None


def resample_size(
    src_tif: str = None, dst_tif: str = None, new_width: int = 1, new_height: int = -1
):
    """
    resample size of the raster.

    Args:
        new_width (int): desired width
        new_height (int): desired height

    Returns: Nothing, it writes a raster file with decreased resolution.
    :param save_path:
    :param new_height:
    :param new_width:

    """
    options = gdal.WarpOptions(options=['ts'], width=new_width, height=new_height)
    newfile = gdal.Warp(
        destNameOrDestDS=src_tif,
        srcDSOrSrcDSTab=dst_tif,
        options=options,
        overwrite=True,
    )
    newfile = None


def resample_projection(src_tif: str = None, dst_tif: str = None, new_epsg: int = 2154):
    """
    resample projection of the raster.

    Args:
        new_epsg (int): desired epsg
        save_path (str): filepath to where the output file should be stored

    Returns: Nothing, it writes a raster file with decreased resolution.
    :param save_path:
    :param new_epsg:

    """
    options = gdal.WarpOptions(options=['ts'], dstSRS='EPSG:' + str(new_epsg))
    newfile = gdal.Warp(
        destNameOrDestDS=src_tif,
        srcDSOrSrcDSTab=dst_tif,
        options=options,
        overwrite=True,
    )
    newfile = None


def toto(src_tif, dst_tif='output.tif'):
    """

    Returns:
        object:
    """
    import rioxarray as rxr

    dataarray = rxr.open_rasterio(src_tif)
    df = dataarray[0].to_pandas()
    print(df.head())
    print(df.columns[0])
    miny = df[df.columns[0]][df[df.columns[0]] > 0].index[-1]
    maxy = df[df.columns[-1]][df[df.columns[-1]] > 0].index[0]
    minx = (
        df[df.index == df.index[0]]
        .T[df[df.index == df.index[0]].T > 0]
        .dropna()
        .index[0]
    )
    maxx = (
        df[df.index == df.index[-1]]
        .T[df[df.index == df.index[-1]].T > 0]
        .dropna()
        .index[-1]
    )
    data = gdal.Open(dst_tif, gdal.GA_ReadOnly)  # Your data the one you want to clip
    gdal.Translate(
        dst_tif,
        data,
        format='GTiff',
        projWin=[minx, maxy, maxx, miny],
        outputSRS=data.GetProjectionRef(),
    )


def raster_to_gdf(src_tif, new_field_name: str = 'elevation') -> gpd.GeoDataFrame:
    """
    Returns:
        object:
    """
    source_raster = gdal.Open(src_tif)
    myarray = source_raster.GetRasterBand(1).ReadAsArray()

    # modify numpy array to mask values
    myarray[(0 <= myarray) & (myarray < 1)] = 2
    myarray[myarray == 1.0] = 0

    driver = gdal.GetDriverByName('GTiff')
    raster_tmp = os.path.join(tempfile.gettempdir(), 'raster_tmp.tif')
    ds_out = driver.CreateCopy(raster_tmp, source_raster)
    band = ds_out.GetRasterBand(1)
    # write the modified array to the raster
    band.WriteArray(myarray)
    # set the NoData metadata flag
    band.SetNoDataValue(-1)
    # clear the buffer, and ensure file is written
    ds_out.FlushCache()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_tmp = os.path.join(tempfile.gettempdir(), 'shp_tmp.shp')
    out_data = driver.CreateDataSource(shp_tmp)
    # getting projection from source raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_raster.GetProjectionRef())
    # create layer with projection
    out_layer = out_data.CreateLayer('polygons', srs)
    new_field = ogr.FieldDefn(new_field_name, ogr.OFTReal)
    out_layer.CreateField(new_field)
    gdal.Polygonize(band, band, out_layer, -1, [], callback=None)

    out_data.SyncToDisk()
    out_data.Destroy()
    out_data = None
    ds_out = None

    gdf = gpd.read_file(shp_tmp)
    os.remove(shp_tmp)
    os.remove(raster_tmp)

    return gdf


def gdf_to_raster(
    gdf: gpd.GeoDataFrame,
    dst_tif,
    measurement: str,
    categorical: bool = False,
    resolution: tuple = (-0.03, 0.03),
    raster_file_like: str = None,
    fill_value=None,
    dtype='float64',
):
    """

    Args:
        dst_tif:
        measurement:
        dtype:
        fill_value:
        gdf:
        categorical:
        resolution:
        raster_file_like:
    """
    if categorical:
        categorical_enums = {
            measurement: gdf[measurement].drop_duplicates().values.tolist()
        }

    else:
        categorical_enums = None
        # for col in bands:
        # gdf[col] = gdf[col].fillna(-999)
        gdf[measurement] = gdf[measurement].astype(int)

    if raster_file_like:
        raster_file = rioxarray.open_rasterio(raster_file_like)
        like = raster_file.to_dataset(name='data')
        resolution = None
    else:
        like = None

    geo_grid = make_geocube(
        vector_data=gdf,
        measurements=[measurement],
        categorical_enums=categorical_enums,
        resolution=resolution,
        fill=fill_value,
        like=like,
    )
    # geo_grid[geo_grid['type'] == geo_grid.type.rio.nodata] = 1
    # geo_grid.type.where(geo_grid.type == geo_grid.type.rio.nodata)
    # trest_string = geo_grid["type_categories"][geo_grid['type'].astype(int)].drop('type_categories')
    # trest_string.fillna(0)

    # geo_grid['type'] = trest_string
    # print(trest_string)
    # geo_grid = geo_grid.rio.reproject(geo_grid.rio.crs, resolution=0.5, resampling=Resampling.average)
    geo_grid[measurement].rio.to_raster(
        dst_tif, compress='lzw', bigtiff='YES', dtype=dtype
    )

    return geo_grid


def tif_to_geojson(tif_path, output_geojson_path):
    # Open the TIFF file
    tif_dataset = gdal.Open(tif_path)

    # Get the spatial reference system (SRS) of the TIFF file
    srs = SpatialReference()
    srs.ImportFromWkt(tif_dataset.GetProjection())

    # Create a new GeoJSON datasource
    driver = ogr.GetDriverByName('GeoJSON')
    geojson_dataset = driver.CreateDataSource(output_geojson_path)

    # Create a new layer in the GeoJSON datasource
    layer = geojson_dataset.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Get the raster band from the TIFF file
    band = tif_dataset.GetRasterBand(1)

    # Convert the raster band to a polygon feature
    gdal.Polygonize(band, None, layer, 0, [], callback=None)

    # Close the datasets
    tif_dataset = None
    geojson_dataset = None


def clip_raster(
    dst_tif=r'./clipped.tif',
    src_tif=r'./utci_1700.tif',
    format='GTiff',
    width=0,
    height=0,
    cut_shp=r'./mask.shp',
    cut_name='mask',
):
    warp_options = gdal.WarpOptions(
        format=format,
        outputType=gdalconst.GDT_Float32,
        width=width,
        height=height,
        dstNodata=None,
        dstAlpha=False,
        dstSRS='EPSG:2154',
        cropToCutline=True,
        cutlineDSName=cut_shp,
        cutlineLayer=cut_name,
        resampleAlg='cubic',
    )

    gdal.Warp(dst_tif, src_tif, options=warp_options)
    return


def clip_raster_processing(
    input_dir: str = None, dst_tif: str = None, cut_shp: str = None, list_files=None
):
    from pymdu.physics.umep.UmepCore import UmepCore

    if list_files is None:
        list_files = ['DSM', 'DEM', 'TDSM', 'CDSM', 'landcover', 'HEIGHT', 'ASPECT']

    for file in list_files:
        umep_core = UmepCore(output_dir=input_dir)
        (
            umep_core.run_processing(
                name='gdal:cliprasterbymasklayer',
                options={
                    'INPUT': os.path.join(input_dir, f'{file}.tif'),
                    'MASK': os.path.join(input_dir, cut_shp),
                    # 'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
                    # 'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
                    'SOURCE_CRS': 'EPSG:2154',
                    'TARGET_CRS': 'EPSG:2154',
                    'TARGET_EXTENT': None,
                    'NODATA': None,
                    'ALPHA_BAND': False,
                    'CROP_TO_CUTLINE': True,
                    'KEEP_RESOLUTION': False,
                    'SET_RESOLUTION': False,
                    'X_RESOLUTION': None,
                    'Y_RESOLUTION': None,
                    'MULTITHREADING': False,
                    'OPTIONS': '',
                    'DATA_TYPE': 0,
                    'EXTRA': '',
                    'OUTPUT': os.path.join(input_dir, dst_tif),
                },
            ),
        )


def raster_file_like(
    src_tif='./BASE/RESHAPE/landcover.tif',
    like_path='./BASE/RESHAPE/DEM.tif',
    dst_tif='./BASE/RESHAPE/landcover_ok.tif',
    remove_nan: bool = False,
):
    raster = rioxarray.open_rasterio(
        src_tif,
        masked=True,
    )

    if remove_nan:
        raster.data[0] = np.nan_to_num(raster.data[0], nan=1.0)

    like = rioxarray.open_rasterio(
        like_path,
        masked=True,
    )

    raster_temp = like.copy()
    try:
        raster_temp.data[0] = raster.data[0][1:, 1:]
    except Exception:
        print('Pas besoin de re-découper')
        raster_temp.data[0] = raster.data[0]

    raster_temp.rio.write_crs('epsg:2154', inplace=True)
    raster_temp.rio.write_transform(like.rio.transform(), inplace=True)
    raster_temp.rio.to_raster(dst_tif)

    return raster


def reproject_resample_cropped_raster(model_file, src_tif, dst_tif='optionnal'):
    """

    Args:
        model_file: le raster .tif dont les caractéristiques doivent être copiées
        input_file: le fichier raster .tif d'entrée
        output_file: le du raster .tif nom que l'on souhaite donner à la sortie
    """
    # model_file = 'Tmrt_1997_157_1000D.tif'
    # input_file = 'DIR_350_VENT_1__UROCK_OUTPUTWS.GTiff'
    # output_file = 'DIR_350_VENT_1__UROCK_OUTPUTWS_ok.GTiff'

    model_ds = gdal.Open(model_file)
    input_ds = gdal.Open(src_tif)
    model_cols = model_ds.RasterXSize
    model_rows = model_ds.RasterYSize
    model_proj = model_ds.GetProjection()
    model_geo = model_ds.GetGeoTransform()
    input_cols = input_ds.RasterXSize
    input_rows = input_ds.RasterYSize  # # Get the geotransform of the raster
    # geotransform = gdal_dem.GetGeoTransform()# print("geotransform", geotransform)# Get the size of the raster in pixels
    xsize = model_ds.RasterXSize
    ysize = model_ds.RasterYSize  # print("xsize", xsize)
    # print("ysize", ysize)# Calculate the coordinates of the upper-left and lower-right corners

    ulx = model_geo[0]
    uly = model_geo[3]
    lrx = ulx + xsize * model_geo[1]
    lry = uly + ysize * model_geo[5]
    minx = float(ulx)
    miny = float(uly)
    maxx = float(lrx)
    maxy = float(lry)
    input_proj = input_ds.GetProjection()
    input_geo = input_ds.GetGeoTransform()
    input_data = input_ds.GetRasterBand(1).ReadAsArray()

    model_ds = None
    input_ds = None

    # projection du raster dans le CRS 2154
    step0 = '/vsimem/step_0.tif'

    print(gdal.__file__)
    ds = gdal.Warp(
        destNameOrDestDS=step0,
        srcDSOrSrcDSTab=src_tif,
        options=gdal.WarpOptions(dstSRS='EPSG:2154', format='GTiff'),
    )

    ds = None

    # resample du raster pour avoir des pixels de -1, 1
    step1 = '/vsimem/step_1.tif'
    ds = gdal.Warp(
        destNameOrDestDS=step1,
        srcDSOrSrcDSTab=step0,
        options=gdal.WarpOptions(xRes=model_geo[1], yRes=model_geo[5]),
    )
    ds = None

    # clip du raster
    dataset = rasterio.open(model_file)
    minx = dataset.bounds[0]
    miny = dataset.bounds[3]
    maxx = dataset.bounds[2]
    maxy = dataset.bounds[1]

    destination = gdal.Translate(
        dst_tif,
        step1,
        options=gdal.TranslateOptions(
            format='GTiff', projWin=[minx, miny, maxx, maxy], outputSRS='EPSG:2154'
        ),
    )

    ds = None

    if os.path.exists(step0):
        os.remove(step0)
    if os.path.exists(step1):
        os.remove(step1)


def add_pixel_width_height(
    dst_tif=r'./utci_1700.tif',
    src_tif=r'./utci_1700_resized.tif',
    n_width=400,
    n_height=400,
):
    # Modifier tif_files[0] pour la rendre plus large
    # Ouvrir l'image
    ds = gdal.Open(src_tif)

    # Obtenir les dimensions de l'image
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Ajouter n1 pixels à la hauteur et n2 pixels à la largeur
    new_width = width + n_width
    new_height = height + n_height

    # Obtenir le nombre de bandes de l'image
    bands = ds.RasterCount

    # Obtenir le format de l'image
    driver = ds.GetDriver()

    # Obtenir les informations de géoréférencement de l'image originale
    geoTransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # Obtenir le système de coordonnées de l'image originale
    srs = osr.SpatialReference(wkt=projection)

    # Calculer la différence de pixels à ajouter pour chaque côté de l'image
    dx = int(n_width / 2)
    dy = int(n_height / 2)

    # Recalculer les informations de géoréférencement pour la nouvelle image
    new_geoTransform = (
        geoTransform[0] - (dx * geoTransform[1]),
        geoTransform[1],
        geoTransform[2],
        geoTransform[3] - (dy * geoTransform[5]),
        geoTransform[4],
        geoTransform[5],
    )

    # Créer une nouvelle image avec les nouvelles dimensions
    new_ds = driver.Create(dst_tif, new_width, new_height, bands, gdal.GDT_Float32)

    # Copier les métadonnées de l'image originale dans la nouvelle image
    new_ds.SetGeoTransform(new_geoTransform)
    new_ds.SetProjection(projection)

    # Copier les bandes de l'image originale dans la nouvelle image en répartissant les pixels ajoutés par rapport au centre de l'image
    for i in range(bands):
        band = ds.GetRasterBand(i + 1)
        new_band = new_ds.GetRasterBand(i + 1)
        data = band.ReadAsArray()
        new_data = np.zeros((new_height, new_width), dtype=np.float32)
        new_data[dy : height + dy, dx : width + dx] = data
        new_band.WriteArray(new_data)

    # Enregistrer les modifications
    new_ds = None


def shp_to_tif(
    InputShp=r'C:\Users\simon\python-scripts\pymdu\Tests\umep\ATLANTECH\pedestrian.shp',
    OutputImage=r'T:\_QAPEOSUD_SIMULATIONS\ATLANTECH_2020\indicator\ZonePietonne.tif',
    RefImage=r'T:\_QAPEOSUD_SIMULATIONS\ATLANTECH_2020\solweig\results\Tmrt_1997_171_2300N.tif',
):
    gdalformat = 'GTiff'
    datatype = gdal.GDT_Byte
    burnVal = 1  # value for the output image pixels
    ##########################################################
    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputShp)
    Shapefile_layer = Shapefile.GetLayer()

    # Rasterise
    Output = gdal.GetDriverByName(gdalformat).Create(
        OutputImage,
        Image.RasterXSize,
        Image.RasterYSize,
        1,
        datatype,
        options=['COMPRESS=DEFLATE'],
    )
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

    # Close datasets
    Band = None
    Output = None
    Image = None
    Shapefile = None

    # Build image overviews
    subprocess.call(
        'gdaladdo --config COMPRESS_OVERVIEW DEFLATE '
        + OutputImage
        + ' 2 4 8 16 32 64',
        shell=True,
    )


def mask_raster_from_tif(
    InputMaskTif=r'cours_de_recreation.tif',
    InputRaster=r'aout_mois_14_heure_10_vent_10_direction_90_32.5_Tair_33_vmin_42_vmax.tif',
):
    data = rioxarray.open_rasterio(InputRaster).copy()
    ds_data = data.to_dataset(name='Indicateur')
    zone_interet = rioxarray.open_rasterio(InputMaskTif).copy()
    ds_zone_interet = zone_interet.to_dataset(name='ZoneInteret')

    merge_xr = xarray.merge([ds_data['Indicateur'], ds_zone_interet['ZoneInteret']])
    data_masked = merge_xr.where(merge_xr.ZoneInteret > 0.5)
    data_masked['Indicateur'][0].rio.to_raster(
        InputRaster.replace('.tif', '_masked.tif')
    )

    return data_masked


def tiff_to_jp2(
    src_tif,
    output='output.jp2',
    openjpeg_base_path='/Users/Boris/anaconda3/envs/pymdu/bin',
):
    """

    Returns:
        object:
    """

    dataset = gdal.Open(src_tif, gdal.GA_ReadOnly)

    # Get the raster band
    band = dataset.GetRasterBand(1)

    # Read the raster data as a NumPy array
    raster_data = band.ReadAsArray()

    # Round the values in the array
    # rounded_data = raster_data.round()
    rounded_data = np.round(raster_data, decimals=3)

    # Multiply the values by 1000
    multiplied_data = rounded_data * 1000

    # Convert the data type to int16
    rounded_multiplied_data_int16 = multiplied_data.astype('int16')

    # Create a new raster dataset for the rounded values
    output_path = os.path.join(tempfile.gettempdir(), 'output.tif')

    driver = gdal.GetDriverByName('GTiff')
    # output_dataset = driver.CreateCopy(output_path, dataset)

    # Create a new raster dataset for the multiplied values
    # output_path = 'output.jp2'
    # driver = gdal.GetDriverByName('JP2OpenJPEG')
    output_dataset = driver.Create(
        output_path,
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount,
        gdal.GDT_Int16,
    )

    # Set the georeferencing information
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())

    # Write the rounded data to the output raster band
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(rounded_multiplied_data_int16)

    # Set the data type of the output band
    output_band.SetMetadata({'PIXELTYPE': 'INT16'})

    # Close the datasets
    band = None
    dataset = None
    output_band = None
    output_dataset = None

    from image_processing import openjpeg

    opj = openjpeg.OpenJpeg(openjpeg_base_path=openjpeg_base_path)
    opj.opj_compress(
        output_path, output, openjpeg_options=openjpeg.DEFAULT_LOSSLESS_COMPRESS_OPTIONS
    )
