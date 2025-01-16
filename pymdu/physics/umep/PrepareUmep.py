import glob
import os
import shutil
from typing import Iterable

import geopandas as gpd

from osgeo import gdal, gdalconst
from shapely import box

import pymdu
from pymdu.GeoCore import GeoCore
from pymdu.geometric import LandCover
from pymdu.image import geotiff
from pymdu.meteo import Meteo
from pymdu.physics.umep.DsmModelGenerator import DsmModelGenerator
from pymdu.physics.umep.HeightAspectModelGenerator import HeightAspectModelGenerator
from pymdu.physics.umep.SVFModelGenerator import SVFModelGenerator
from pymdu.physics.umep.Solweig import Solweig
from pymdu.physics.umep.SurfaceModelGenerator import SurfaceModelGenerator
from pymdu.physics.umep.Urock import Urock


class PrepareUmep(GeoCore):
    def __init__(
        self,
        working_directory: os.path,
        java_home_path: os.path = os.environ.get('JAVA_HOME'),
        input_dir: os.path = './Ressources/Inputs',
        output_dir: os.path = './Ressources/Outputs',
    ):
        self.working_directory = working_directory
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.java_home_path = java_home_path

    def run_solweig(self):
        bbox = self.bbox
        gdf_project = gpd.GeoDataFrame(
            gpd.GeoSeries(box(bbox[0], bbox[1], bbox[2], bbox[3])),
            columns=['geometry'],
            crs='epsg:4126',
        )
        # gdf_project = gdf_project.to_crs(epsg=2154)
        gdf_project = gdf_project.scale(xfact=1.15, yfact=1.15)

        envelope_polygon = gdf_project.envelope.bounds
        bbox = envelope_polygon.values[0]
        bbox_final = [bbox[0], bbox[1], bbox[2], bbox[3]]

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        geo_core = GeoCore()
        geo_core.bbox = bbox_final

        building = pymdu.geometric.Building(output_path=self.output_dir)
        building.bbox = bbox_final
        buildings_gdf = building.run().to_gdf()
        building.to_shp(name='buildings')

        water = pymdu.geometric.Water(output_path=self.output_dir)
        water.bbox = bbox_final
        water_gdf = water.run().to_gdf()
        water.to_shp(name='waters')

        pedestrian = pymdu.geometric.Pedestrian(output_path=self.output_dir)
        pedestrian.bbox = bbox_final
        pedestrian_gdf = pedestrian.run().to_gdf()
        pedestrian.to_shp(name='pedestrians')

        vegetation = pymdu.geometric.Vegetation(
            output_path=self.output_dir, min_area=100
        )
        vegetation.bbox = bbox_final
        vegetation_gdf = vegetation.run().to_gdf()
        vegetation.to_shp(name='vegetations')

        landcover = LandCover(
            output_path=self.output_dir,
            building_gdf=buildings_gdf,
            vegetation_gdf=vegetation_gdf,
            water_gdf=water_gdf,
            pedestrian_gdf=pedestrian_gdf,
            write_file=False,
        )

        landcover_gdf = landcover.run(keep_geom_type=True).to_gdf()
        landcover.bbox = bbox_final
        landcover.to_shp(name='landcover')

        geotiff.gdf_to_raster(
            dst_tif=os.path.join(self.input_dir, 'landcover.tif'),
            gdf=landcover_gdf,
            measurement='type',
            resolution=(-1, 1),
            raster_file_like=None,
            fill_value=None,
            dtype='float64',
        )

        dem = pymdu.geometric.Dem(output_path=self.input_dir)
        dem.bbox = bbox_final
        dem.run()

        warp_options = gdal.WarpOptions(
            format='GTiff',  # width=width,
            # height=height,
            xRes=1,
            yRes=1,
            outputType=gdalconst.GDT_Float32,
            dstNodata=None,
            dstSRS='EPSG:2154',
            cropToCutline=True,
            cutlineDSName=os.path.join(self.input_dir, 'mask.shp'),
            cutlineLayer='mask',
        )

        gdal.Warp(
            destNameOrDestDS=os.path.join(self.output_dir, 'DEM.tif'),
            srcDSOrSrcDSTab=os.path.join(self.input_dir, 'DEM.tif'),
            options=warp_options,
        )

        gdal.Warp(
            destNameOrDestDS=os.path.join(self.input_dir, 'landcover_clip.tif'),
            srcDSOrSrcDSTab=os.path.join(self.input_dir, 'landcover.tif'),
            options=warp_options,
        )

        geotiff.raster_file_like(
            dst_tif=os.path.join(self.output_dir, 'landcover.tif'),
            src_tif=os.path.join(self.input_dir, 'landcover_clip.tif'),
            like_path=os.path.join(self.output_dir, 'DEM.tif'),
            remove_nan=True,
        )

        dsm = DsmModelGenerator(
            working_directory=self.output_dir,
            output_filepath_dsm=os.path.join(self.output_dir, 'DSM.tif'),
            input_filepath_dem=os.path.join(self.output_dir, 'DEM.tif'),
            input_building_shp_path=os.path.join(self.output_dir, 'buildings.shp'),
            input_mask_shp_path=os.path.join(self.input_dir, 'mask.shp'),
        )
        dsm.run(pixel_resolution=2)

        # gdal.Warp(destNameOrDestDS=os.path.join(os.path.join(self.output_dir, 'DSM.tif'),
        #           srcDSOrSrcDSTab=os.path.join(self.input_dir, 'DSM.tif'),
        #           options=warp_options)

        surface_model = SurfaceModelGenerator(
            working_directory=self.input_dir,
            input_filepath_dsm=os.path.join(self.output_dir, 'DSM.tif'),
            input_filepath_dem=os.path.join(self.output_dir, 'DEM.tif'),
            input_filepath_tree_shp=os.path.join(self.output_dir, 'trees.shp'),
            output_filepath_cdsm=os.path.join(self.input_dir, 'CDSM_clip.tif'),
            output_filepath_tdsm=os.path.join(self.input_dir, 'TDSM_clip.tif'),
        )
        surface_model.run()

        list_files = ['CDSM', 'TDSM']

        for file in list_files:
            gdal.Warp(
                destNameOrDestDS=os.path.join(self.output_dir, f'{file}.tif'),
                srcDSOrSrcDSTab=os.path.join(self.input_dir, f'{file}_clip.tif'),
                options=warp_options,
            )

        SVFModelGenerator(
            working_directory=self.output_dir,
            input_filepath_dsm=os.path.join(self.output_dir, 'DSM.tif'),
            input_filepath_cdsm=os.path.join(self.output_dir, 'CDSM.tif'),
            input_filepath_tdsm=os.path.join(self.output_dir, 'TDSM.tif'),
            ouptut_filepath_svf=os.path.join(self.output_dir, 'SVF.tif'),
        ).run()

        HeightAspectModelGenerator(
            working_directory=self.input_dir,
            output_filepath_height=os.path.join(self.input_dir, 'HEIGHT_clip.tif'),
            output_filepath_aspect=os.path.join(self.input_dir, 'ASPECT_clip.tif'),
            input_filepath_dsm=os.path.join(self.output_dir, 'DSM.tif'),
        ).run()

        list_files = ['HEIGHT', 'ASPECT']

        for file in list_files:
            gdal.Warp(
                destNameOrDestDS=os.path.join(self.output_dir, f'{file}.tif'),
                srcDSOrSrcDSTab=os.path.join(self.input_dir, f'{file}_clip.tif'),
                options=warp_options,
            )

        meteo_path = os.path.join(self.output_dir, 'meteo')
        if not os.path.exists(meteo_path):
            os.makedirs(meteo_path, exist_ok=True)

        meteo_test = Meteo(output_path=meteo_path)
        meteo_test.bbox = bbox_final
        meteo_test.run(begin='2022-06-30 08:00:00', end='2022-06-30 12:00:00')
        meteo_path = glob.glob(os.path.join(meteo_path, '*.txt'))

        d = Solweig(
            meteo_path=meteo_path[0],
            output_dir=self.output_dir,
            working_directory=self.output_dir,
            input_filepath_landcover=os.path.join(self.output_dir, 'landcover.tif'),
            input_filepath_dsm=os.path.join(self.output_dir, 'DSM.tif'),
            input_filepath_dem=os.path.join(self.output_dir, 'DEM.tif'),
            input_filepath_cdsm=os.path.join(self.output_dir, 'CDSM.tif'),
            input_filepath_tdsm=os.path.join(self.output_dir, 'TDSM.tif'),
            input_filepath_height=os.path.join(self.output_dir, 'HEIGHT.tif'),
            input_filepath_aspect=os.path.join(self.output_dir, 'ASPECT.tif'),
            input_filepath_shadowmats_npz=os.path.join(
                self.output_dir, 'shadowmats.npz'
            ),
            input_filepath_svf_zip=os.path.join(self.output_dir, 'svfs.zip'),
        )
        d.run()

    def run_urock(
        self,
        buildings_path_shp='./Ressources/Inputs/buildings.shp',
        trees_path_shp='./Ressources/Inputs/trees.shp',
        winds_speed: list = [1],
        winds_direction: Iterable = range(0, 10, 10),
    ):
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # try_urock_gen = UrockFiles(
        #     output_path=self.input_dir,
        #     buildings_gdf=gpd.read_file(buildings_path_shp),
        #     trees_gdf=gpd.read_file(trees_path_shp),
        # )

        # urock_buildings_gdf = try_urock_gen.generate_urock_buildings(
        #     filename_shp="urock_bld.shp"
        # )
        # urock_trees_gdf = try_urock_gen.generate_urock_trees(
        #     filename_shp="urock_trees.shp"
        # )

        urock = Urock(
            output_dir=self.output_dir,
            working_directory=self.working_directory,
            input_filepath_buildings_shp=os.path.join(self.input_dir, 'urock_bld.shp'),
            input_filepath_trees_shp=os.path.join(self.input_dir, 'urock_trees.shp'),
            java_home_path=self.java_home_path,
        )

        # urock.run_with_umep(output_dir=None,
        #                     height_field="hauteur",
        #                     wind_speed=5,
        #                     wind_direction=180,
        #                     max_height_field="MAX_HEIGHT",
        #                     min_height_field="MIN_HEIGHT",
        #                     wind_height=1.5,
        #                     horizontal_resolution=5,
        #                     vertical_resolution=5,
        #                     save_netcdf=True, save_raster=True, save_vector=True)

        # urock.run(output_dir=outputs_results,
        #           height_field="hauteur",
        #           wind_speed=1,
        #           wind_direction=0,
        #           wind_height=1.5,
        #           horizontal_resolution=2,
        #           vertical_resolution=5,
        #           save_netcdf=False, save_raster=True, save_vector=False)

        urock.run_parallel(
            winds_speed=winds_speed,
            winds_direction=winds_direction,
            height_field='hauteur',
            wind_height=1.5,
            horizontal_resolution=5,
            vertical_resolution=5,
            save_netcdf=True,
            save_raster=True,
            save_vector=True,
        )


if __name__ == '__main__':
    from pymdu.GeoCore import GeoCore

    output_dir = './Ressources/Outputs/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    geo_core = GeoCore()
    geo_core.bbox = [
        -1.1493508673864028,
        46.181799599701606,
        -1.1457294027794944,
        46.18377074425578,
    ]

    prepare_umep = PrepareUmep(
        working_directory='./Ressources/Outputs/',
        output_dir=output_dir,
    )
    prepare_umep.bbox = [
        -1.1493508673864028,
        46.181799599701606,
        -1.1457294027794944,
        46.18377074425578,
    ]

    prepare_umep.run_solweig()  # prepare_umep.run_urock(buildings_path_shp="./Ressources/Inputs/buildings.shp",  #                        trees_path_shp="./Ressources/Inputs/trees.shp")
