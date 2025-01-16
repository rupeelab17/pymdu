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
from __future__ import absolute_import

import glob
import os
import pathlib

import numpy as np
from osgeo import gdal, osr, ogr

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.physics.solar.UtilitiesSolar import saveraster


class DsmGenerator(GeoCore):
    def __init__(
        self,
        filepath_dem: str,
        filepath_mask: str,
        osm_tool: bool = False,
        building_shp_path: str = None,
        output_path: str = None,
    ):
        self.filepath_dem = filepath_dem
        self.filepath_mask = filepath_mask
        self.osm_tool = osm_tool
        self.building_shp_path = building_shp_path
        self.output_path = output_path if output_path else TEMP_PATH

    def run(self):
        gdal_dem = gdal.Open(self.filepath_dem)
        # dem = gdal_dem.ReadAsArray().astype(float)
        dem_wkt = gdal_dem.GetProjection()
        dem_crs = osr.SpatialReference()
        dem_crs.ImportFromWkt(dem_wkt)
        # dem_epsg = dem_crs.GetAttrValue("PROJCS|AUTHORITY", 1)
        # dem_unit = dem_crs.GetAttrValue('UNIT')

        # print(dem_crs)
        # print(dem_epsg)
        # print(dem_unit)
        # print(dem_wkt)

        # Get the geotransform of the raster
        geotransform = gdal_dem.GetGeoTransform()

        # print("geotransform", geotransform)

        # Get the size of the raster in pixels
        xsize = gdal_dem.RasterXSize
        ysize = gdal_dem.RasterYSize

        # print("xsize", xsize)
        # print("ysize", ysize)

        # Calculate the coordinates of the upper-left and lower-right corners
        ulx = geotransform[0]
        uly = geotransform[3]
        lrx = ulx + xsize * geotransform[1]
        lry = uly + ysize * geotransform[5]

        minx = float(ulx)
        miny = float(uly)
        maxx = float(lrx)
        maxy = float(lry)

        # print("minx", minx)
        # print("miny", miny)
        # print("maxx", maxx)
        # print("maxy", maxy)

        # bbox = box(minx, miny, maxx, maxy)
        # self.mask = gpd.GeoDataFrame(index=[0], crs='epsg:2154', geometry=[bbox])
        # self.mask.to_file(self.output_folder + 'mask.shp')
        # print("bbox", bbox)

        # stats = gdal_dem.GetRasterBand(1).GetStatistics(0, 1)
        # print("hauteur stats MIN", stats[0])
        # print("hauteur stats MAX", stats[1])
        # print("hauteur stats MEAN", stats[2])
        # print("hauteur stats STDDEV", stats[3])

        if self.osm_tool:
            pass

        if self.building_shp_path:
            # Open a Shapefile, and get field names
            source_building_shp = ogr.Open(self.building_shp_path, update=True)
            # driver = ogr.Open(self.building_shp_path, update=True).GetDriver()
            # source = driver.Open(self.building_shp_path, update=True)

            layer_building_shp = source_building_shp.GetLayer()
            layer_defn = layer_building_shp.GetLayerDefn()
            field_names = [
                layer_defn.GetFieldDefn(i).GetName()
                for i in range(layer_defn.GetFieldCount())
            ]
            # print(field_names)
            # print(len(field_names))
            if 'height_asl' not in field_names:
                # Add a new field
                new_field = ogr.FieldDefn('height_asl', ogr.OFTReal)
                layer_building_shp.CreateField(new_field)

            # https://opensourceoptions.com/blog/zonal-statistics-algorithm-with-python-in-4-steps/

            def boundingBoxToOffsets(bbox, geot):
                col1 = int((bbox[0] - geot[0]) / geot[1])
                col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
                row1 = int((bbox[3] - geot[3]) / geot[5])
                row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
                return [row1, row2, col1, col2]

            def geotFromOffsets(row_offset, col_offset, geot):
                new_geot = [
                    geot[0] + (col_offset * geot[1]),
                    geot[1],
                    0.0,
                    geot[3] + (row_offset * geot[5]),
                    0.0,
                    geot[5],
                ]
                return new_geot

            mem_driver_gdal = gdal.GetDriverByName('MEM')
            mem_driver = ogr.GetDriverByName('Memory')

            shp_name = 'temp.shp'
            nodata = gdal_dem.GetRasterBand(1).GetNoDataValue()

            p_feat = layer_building_shp.GetNextFeature()

            while p_feat:
                if p_feat.GetGeometryRef() is not None:
                    if os.path.exists(shp_name):
                        mem_driver.DeleteDataSource(shp_name)

                    dest_srs = osr.SpatialReference()
                    dest_srs.ImportFromEPSG(2154)
                    tp_ds = mem_driver.CreateDataSource(shp_name)
                    tp_lyr = tp_ds.CreateLayer('polygons', dest_srs, ogr.wkbPolygon)
                    tp_lyr.CreateFeature(p_feat.Clone())

                    offsets = boundingBoxToOffsets(
                        p_feat.GetGeometryRef().GetEnvelope(), geotransform
                    )
                    # print("offsets", offsets)
                    new_geot = geotFromOffsets(offsets[0], offsets[2], geotransform)
                    # print("new_geot", new_geot)

                    tr_ds = mem_driver_gdal.Create(
                        '',
                        offsets[3] - offsets[2],
                        offsets[1] - offsets[0],
                        1,
                        gdal.GDT_Byte,
                    )

                    tr_ds.SetGeoTransform(new_geot)
                    gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
                    tr_array = tr_ds.ReadAsArray()

                    try:
                        r_array = gdal_dem.GetRasterBand(1).ReadAsArray(
                            offsets[2],
                            offsets[0],
                            offsets[3] - offsets[2],
                            offsets[1] - offsets[0],
                        )
                    except Exception as e:
                        print('ERROR DsmGenerator', e)
                        r_array = None

                    id = p_feat.GetFID()
                    # print(id)

                    if r_array is not None:
                        dem_maskarray = np.ma.MaskedArray(
                            r_array,
                            mask=np.logical_or(
                                r_array == nodata, np.logical_not(tr_array)
                            ),
                        )
                        if dem_maskarray is not None:
                            # print("elevation mean", dem_maskarray.mean())
                            field_hauteur = p_feat.GetField('hauteur')
                            # print(f"hauteur batiment : {field_hauteur}\n")

                            if (
                                isinstance(dem_maskarray.mean(), float)
                                and dem_maskarray.mean()
                                and field_hauteur
                            ):
                                p_feat.SetField(
                                    'height_asl', dem_maskarray.mean() + field_hauteur
                                )
                                # p_feat.SetField('height_asl', 0)
                            elif (
                                isinstance(dem_maskarray.mean(), float)
                                and dem_maskarray.mean()
                                and not field_hauteur
                            ):
                                p_feat.SetField('height_asl', dem_maskarray.mean())
                                # p_feat.SetField('height_asl', 0)
                            else:
                                pass
                                # p_feat.SetField('height_asl', nodata)

                            layer_building_shp.SetFeature(p_feat)

                            # print(maskarray.min(),
                            # maskarray.max(),
                            # maskarray.mean(),
                            # np.ma.median(maskarray),
                            # maskarray.std(),
                            # maskarray.sum(),
                            # maskarray.count())

                        else:
                            # p_feat.SetField('height_asl', nodata)
                            layer_building_shp.SetFeature(p_feat)
                            pass
                    else:
                        # p_feat.SetField('height_asl', nodata)
                        layer_building_shp.SetFeature(p_feat)
                        pass

                    tp_ds = None
                    tp_lyr = None
                    tr_ds = None

                p_feat = layer_building_shp.GetNextFeature()

            # Close the Shapefile
            source = None

            # print("dem_wkt", dem_crs)
            inlayer = pathlib.Path(self.building_shp_path).stem
            # print("inlayer", inlayer)

            sort_options = gdal.VectorTranslateOptions(
                options=[
                    '-select',
                    'height_asl',
                    '-t_srs',
                    'EPSG:2154',
                    '-sql',
                    f'SELECT * FROM  {inlayer} ORDER BY height_asl ASC',
                ]
            )

            gdal.UseExceptions()
            # Reads example file with sorted polygons
            sort_ln = 'sortPoly'

            # Sort layer ascending to prevent lower buildings from overwriting higher buildings in some complexes
            path_sortPoly = os.path.join(self.output_path, sort_ln + '.shp')
            # print("path_sortPoly", path_sortPoly)

            # if os.path.exists(path_sortPoly):
            #     os.remove(path_sortPoly)

            gdal.VectorTranslate(
                destNameOrDestDS=path_sortPoly,
                srcDS=self.building_shp_path,
                # SQLStatement=f'SELECT * FROM  {inlayer} ORDER BY hauteur ASC',
                options=sort_options,
                format='ESRI Shapefile',
            )

            # Convert polygon layer to raster
            source_sortPoly = gdal.OpenEx(path_sortPoly)

            rasterize_options = gdal.RasterizeOptions(
                xRes=1,
                yRes=-1,
                attribute='height_asl',
                format='GTiff',
                layers=[str(sort_ln)],
                creationOptions=['COMPRESS=DEFLATE', 'TILED=YES'],
            )

            if os.path.exists(os.path.join(self.output_path, 'clipdsm.tif')):
                os.remove(os.path.join(self.output_path, 'clipdsm.tif'))

            ras = gdal.Rasterize(
                destNameOrDestDS=os.path.join(self.output_path, 'clipdsm.tif'),
                srcDS=source_sortPoly,
                options=rasterize_options,
            )

            del ras

            warp_options = gdal.WarpOptions(
                format='GTiff',
                xRes=1,
                yRes=1,
                dstSRS='EPSG:2154',
                cutlineDSName=self.filepath_mask,
                cutlineLayer='mask',
                cropToCutline=True,
            )

            if os.path.exists(os.path.join(self.output_path, 'clipdsm_clipped.tif')):
                os.remove(os.path.join(self.output_path, 'clipdsm_clipped.tif'))

            ds = gdal.Warp(
                destNameOrDestDS=os.path.join(self.output_path, 'clipdsm_clipped.tif'),
                srcDSOrSrcDSTab=os.path.join(self.output_path, 'clipdsm.tif'),
                options=warp_options,
            )
            del ds

            warp_options = gdal.WarpOptions(
                # HYPER IMPORTANT overwrite == True
                # Remplacez le jeu de données cible s'il existe déjà. L'écrasement doit être compris ici comme la suppression et la recréation du fichier à partir de zéro. Notez que si cette option n'est pas spécifiée et que le fichier de sortie existe déjà, il sera mis à jour sur place.
                options='overwrite',
                format='GTiff',
                outputBounds=[str(minx), str(miny), str(maxx), str(maxy)],
                outputBoundsSRS=dem_crs,
                dstSRS=dem_crs,
                xRes=1,
                yRes=-1,
                dstNodata='-9999',
                cropToCutline=True,
            )

            if os.path.exists(os.path.join(self.output_path, 'clipdem.tif')):
                os.remove(os.path.join(self.output_path, 'clipdem.tif'))

            gdal.Warp(
                destNameOrDestDS=os.path.join(self.output_path, r'clipdem.tif'),
                srcDSOrSrcDSTab=self.filepath_dem,
                options=warp_options,
            )

            # Adding DSM to DEM
            # Read DEM
            dem_raster = gdal.Open(os.path.join(self.output_path, 'clipdem.tif'))
            dem_array = np.array(dem_raster.ReadAsArray().astype(np.float64))
            dem_raster = None

            dsm_raster = gdal.Open(
                os.path.join(self.output_path, 'clipdsm_clipped.tif')
            )
            dsm_array = np.array(dsm_raster.ReadAsArray().astype(np.float64))

            indx = dsm_array.shape
            indx_dem = dem_array.shape

            try:
                for ix in range(0, int(indx[0])):
                    for iy in range(0, int(indx[1])):
                        if int(dsm_array[ix, iy]) == 0:
                            dsm_array[ix, iy] = dem_array[ix, iy]
            except:
                # erreurs d'arrondis des pixels ?
                for ix in range(0, int(indx_dem[0])):
                    for iy in range(0, int(indx_dem[1])):
                        if int(dsm_array[ix, iy]) == 0:
                            dsm_array[ix, iy] = dem_array[ix, iy]

            saveraster(
                gdal_data=dsm_raster,
                filename=os.path.join(self.output_path, 'DSM.tif'),
                raster=dsm_array,
            )
            dsm_raster = None

            os.remove(os.path.join(self.output_path, 'clipdem.tif'))
            os.remove(os.path.join(self.output_path, 'clipdsm.tif'))
            os.remove(os.path.join(self.output_path, 'clipdsm_clipped.tif'))
            os.remove(path_sortPoly)
            for file in glob.glob('sortPoly*'):
                os.remove(file)


if __name__ == '__main__':
    from pymdu.geometric.Building import Building
    from pymdu.geometric import Dem

    geocore = GeoCore()
    geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    building = Building(output_path='./')
    building.run().to_shp(name='buildings')
    print(building.output_path_shp)

    dem = Dem(output_path='./')
    dem.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    ign_dem = dem.run()

    print(ign_dem.path_save_tiff)
    print(ign_dem.path_save_mask)
    print(building.output_path_shp)

    dsm = DsmGenerator(
        output_path='./',
        filepath_dem=ign_dem.path_save_tiff,
        filepath_mask=ign_dem.path_save_mask,
        osm_tool=False,
        building_shp_path=building.output_path_shp,
    )
    dsm.run()
