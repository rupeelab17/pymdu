import os

import geopandas as gpd
from pymdu.physics.umep.UmepCore import UmepCore


class DsmModelGenerator(UmepCore):
    def __init__(
        self,
        working_directory: os.path,
        output_filepath_dsm: os.path = 'DSM.tif',
        input_filepath_dem: os.path = 'DEM.tif',
        input_building_shp_path: os.path = 'buildings.shp',
        input_mask_shp_path: os.path = 'mask.shp',
    ):
        """
        Initializes an instance of the class.

        Args:
            input_solweig_path (os.path): The path to the input Solweig file.

        Returns:
            None
        """
        super().__init__(output_dir=working_directory)
        self.output_filepath_dsm = output_filepath_dsm
        self.input_filepath_dem = input_filepath_dem
        self.input_building_shp_path = input_building_shp_path
        self.input_mask_shp_path = input_mask_shp_path

    def run(
        self,
        building_level: float = 3.1,
        use_osm: bool = False,
        input_field: str = 'hauteur',
        pixel_resolution: int = 1,
    ):
        gdf = gpd.read_file(self.input_mask_shp_path, driver='ESRI Shapefile')
        gdf = gdf.to_crs(epsg=2154)
        envelope_polygon = gdf.envelope.bounds
        bbox = envelope_polygon.values[0]
        extent = f'{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]} [EPSG:2154]'
        print('extent', str(extent))
        self.run_processing(
            name='umep:Spatial Data: DSM Generator',
            options={
                'INPUT_DEM': self.input_filepath_dem,
                'INPUT_POLYGONLAYER': self.input_building_shp_path,
                'INPUT_FIELD': input_field,
                'USE_OSM': use_osm,
                'BUILDING_LEVEL': building_level,
                'EXTENT': str(extent),
                'PIXEL_RESOLUTION': pixel_resolution,
                'OUTPUT_DSM': self.output_filepath_dsm,
            },
        )
        return None


if __name__ == '__main__':
    from pymdu.GeoCore import GeoCore
    import pymdu
    from osgeo import gdal, gdalconst

    geo_core = GeoCore()
    geo_core.bbox = [
        -1.1610335199999986,
        46.18881213999999,
        -1.1541192199999983,
        46.19434632999999,
    ]

    building = pymdu.geometric.Building(output_path='./')
    building.bbox = [
        -1.1610335199999986,
        46.18881213999999,
        -1.1541192199999983,
        46.19434632999999,
    ]
    buildings_gdf = building.run().to_gdf()
    building.to_shp(name='buildings')

    dem = pymdu.geometric.Dem(output_path='./')
    dem.bbox = [
        -1.1610335199999986,
        46.18881213999999,
        -1.1541192199999983,
        46.19434632999999,
    ]
    # dem.run(shape=(width, height))
    dem.run()

    warp_options = gdal.WarpOptions(
        format='GTiff',
        # width=width,
        # height=height,
        xRes=1,
        yRes=1,
        outputType=gdalconst.GDT_Float32,
        dstNodata=None,
        # dstAlpha=False,
        dstSRS='EPSG:2154',
        cropToCutline=True,
        cutlineDSName='mask.shp',
        cutlineLayer='mask',
        # resampleAlg='cubic'
    )

    gdal.Warp(
        destNameOrDestDS='DEM_clip.tif', srcDSOrSrcDSTab='DEM.tif', options=warp_options
    )

    dsm = DsmModelGenerator(
        working_directory='./',
        output_filepath_dsm='DSM.tif',
        input_filepath_dem='DEM.tif',
        input_building_shp_path='buildings.shp',
        input_mask_shp_path='mask.shp',
    )
    dsm.run()
