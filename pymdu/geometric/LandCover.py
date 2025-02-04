# ******************************************************************************
#  This file is part of pymdu.                                                 *
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
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.commons.BasicFunctions import BasicFunctions
from pymdu.image.geotiff import gdf_to_raster

try:
    from osgeo import gdal, ogr, gdalconst
except ImportError:
    pass


class LandCover(GeoCore, BasicFunctions):
    """
    classdocs
    rappel de la définition initiale des classes dans Umep
    Name              Code Alb  Emis Ts_deg Tstart TmaxLST
    Roofs(buildings)   2   0.18 0.95 0.58   -9.78  15.0
    Dark_asphalt       1   0.18 0.95 0.58   -9.78  15.0
    Cobble_stone_2014a 0   0.20 0.95 0.37   -3.41  15.0
    Water              7   0.05 0.98 0.00    0.00  12.0
    Grass_unmanaged    5   0.16 0.94 0.21   -3.38  14.0
    bare_soil          6   0.25 0.94 0.33   -3.01  14.0
    Walls             99   0.20 0.90 0.37   -3.41  15.0
    """

    def __init__(
        self,
        building_gdf: gpd.GeoDataFrame = None,
        vegetation_gdf: gpd.GeoDataFrame = None,
        water_gdf: gpd.GeoDataFrame = None,
        pedestrian_gdf: gpd.GeoDataFrame = None,
        cosia_gdf: gpd.GeoDataFrame = None,
        dxf_gdf: gpd.GeoDataFrame = None,
        output_path: str = None,
        write_file: bool = True,
    ):
        """
        Initializes the object with the given parameters.

        Args:
             building_gdf: gpd.GeoDataFrame = None,
            vegetation_gdf: gpd.GeoDataFrame = None,
            water_gdf: gpd.GeoDataFrame = None,
            pedestrian_gdf: gpd.GeoDataFrame = None,
            cosia_gdf: gpd.GeoDataFrame = None,
            dxf_gdf: gpd.GeoDataFrame = None,
            output_path: str = None,
            write_file: bool = True,

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            plt.clf()  # markdown-exec: hide
            from pymdu.image import geotiff
            from pymdu.geometric import Vegetation, Pedestrian, Water, Building, LandCover
            from pymdu.geometric.Dem import Dem
            from pymdu.commons.BasicFunctions import plot_sol_occupancy
            from pymdu.GeoCore import GeoCore

            geocore = GeoCore()
            geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]

            building = Building(output_path="./")
            buildings_gdf = building.run().to_gdf()
            # building.to_shp(name='buildings')

            water = Water(output_path="./")
            water_gdf = water.run().to_gdf()
            # water.to_shp(name='water')

            pedestrian = Pedestrian(output_path="./")
            pedestrian_gdf = pedestrian.run().to_gdf()
            # pedestrian.to_shp(name='pedestrian')

            vegetation = Vegetation(output_path="./", min_area=100)
            vegetation_gdf = vegetation.run().to_gdf()
            # vegetation.to_shp(name='vegetation')

            # cosia_gdf = gpd.read_file("../../demos/demo_cosia_gdf.shp")
            # dxf_gdf = gpd.read_f  #
            cosia_gdf = None
            dxf_gdf = None

            landcover = LandCover(
            output_path="./",
            building_gdf=buildings_gdf,
            vegetation_gdf=vegetation_gdf,
            water_gdf=water_gdf,
            cosia_gdf=cosia_gdf,
            dxf_gdf=dxf_gdf,
            pedestrian_gdf=pedestrian_gdf,
            write_file=False,
            )

            landcover.run()
            # landcover.to_shp(name="landcover")
            landcover_gdf = landcover.to_gdf()

            fig, ax = plt.subplots(figsize=(10, 10))

            if cosia_gdf is not None:
                landcover_gdf.plot(color=landcover_gdf["color"])
                fig_hist = plot_sol_occupancy(cosia_gdf, landcover_gdf)
                fig_hist.show()
            else:
                landcover_gdf.plot(ax=plt.gca(), edgecolor="black", column="type")

            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
            * For module TODOs
        """
        self.listGDF = []
        self.output_path = output_path if output_path else TEMP_PATH
        if isinstance(building_gdf, gpd.geodataframe.GeoDataFrame):
            self.building = building_gdf[["geometry"]].copy()
            self.building["type"] = 2
            self.listGDF.append(self.building)
        if isinstance(vegetation_gdf, gpd.geodataframe.GeoDataFrame):
            self.vegetation = vegetation_gdf[["geometry"]].copy()
            self.vegetation["type"] = 5
            self.listGDF.append(self.vegetation)

        self.cosia_gdf = cosia_gdf
        self.dxf_gdf = dxf_gdf

        if isinstance(water_gdf, gpd.geodataframe.GeoDataFrame):
            # water_gdf.to_file('./water.shp', driver='ESRI Shapefile')
            self.water = water_gdf[["geometry"]].copy()
            self.water["type"] = 7
            self.listGDF.append(self.water)

        if isinstance(pedestrian_gdf, gpd.geodataframe.GeoDataFrame):
            self.pedestrian = pedestrian_gdf[["geometry"]].copy()
            self.pedestrian["type"] = 6
            # self.pedestrian['type']= self.pedestrian[self.pedestrian['type']] = 6
            # pedestrian_gdf.to_file('./pedestrian.shp', driver='ESRI Shapefile')
            self.listGDF.append(self.pedestrian)

        # ajout des types

        # self.building['type'] = [2 for x in self.building.index]
        # self.vegetation['type'] = [5 for x in self.vegetation.index]
        # self.water['type'] = [7 for x in self.water.index]

        self.write_file = write_file

    def run(self, mask=None, keep_geom_type=True):
        if self.cosia_gdf is not None and self.dxf_gdf is not None:
            self.gdf = self.unify_cosia_dxf()
        elif self.cosia_gdf is not None:
            self.gdf = self.cosia_gdf
        else:
            if mask is not None:
                mask = mask.to_crs(4326)
                envelope_polygon = mask.envelope.bounds
                bbox = envelope_polygon.values[0]
                bbox = box(bbox[0], bbox[1], bbox[2], bbox[3])
            else:
                bbox = box(self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])

            mask = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[bbox])
            mask = mask.to_crs(self._epsg)

            # print("CONCAT 1")
            no_ground = pd.concat(self.listGDF)

            # print("OVERLAY")
            ground = mask.overlay(
                no_ground,
                how="difference",
                keep_geom_type=keep_geom_type,
                make_valid=True,
            )
            ground = ground.copy()
            ground = ground.to_crs(self._epsg)
            ground = ground.explode(ignore_index=True)

            # ground.insert(len(ground.columns), 'type', 1)
            ground["type"] = 1
            ground.crs = mask.crs

            # print("CONCAT 2")
            self.listGDF.append(ground)

            for k, gdf in enumerate(self.listGDF):
                gdf.crs = self._epsg

            landcover = pd.concat(self.listGDF)
            # print("landcover GDF", landcover['type'].drop_duplicates().values.tolist())

            # print("EXPLODE")
            landcover = landcover.explode(ignore_index=True).copy()
            try:
                # print("CLIP")
                landcover = gpd.clip(landcover, mask, keep_geom_type=keep_geom_type)
            except Exception as e:
                print(f"Buffer is on {e}")
                landcover.geometry = landcover.buffer(0)
                landcover = gpd.clip(landcover, mask, keep_geom_type=keep_geom_type)

            self.gdf = landcover.explode(ignore_index=True)

        if self.write_file:
            self.gdf = self.gdf[self.gdf.geometry.type != "LineString"]
            self.gdf.to_file("landcover.geojson")

        return self

    def to_gdf(self):
        return self.gdf

    def to_gpkg(self, name: str = "landcover"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{self.output_path}/{name}.gpkg", driver="GPKG")

    def unify_cosia_dxf(self):
        from pymdu.geometric.Cosia import Cosia

        if self.dxf_gdf.crs != self.cosia_gdf.crs:
            self.dxf_gdf = self.dxf_gdf.to_crs(self.cosia_gdf.crs)

        self.cosia_gdf = self.cosia_gdf.rename(columns={"classe": "classe_cosia"})
        self.dxf_gdf = self.dxf_gdf.rename(columns={"classe": "classe_dxf"})

        self.dxf_gdf["geometry"] = self.dxf_gdf["geometry"].buffer(0.001)

        final_gdf = gpd.overlay(
            self.cosia_gdf, self.dxf_gdf, how="union", keep_geom_type=False
        )
        final_gdf["geometry"] = final_gdf["geometry"].buffer(0)

        final_gdf["classe"] = [
            classe_cosia if pd.isna(classe_dxf) else classe_dxf
            for (classe_cosia, classe_dxf) in zip(
                final_gdf["classe_cosia"], final_gdf["classe_dxf"]
            )
        ]
        final_gdf["color"] = [Cosia().table_color_cosia[x] for x in final_gdf.classe]

        return final_gdf

    def create_landcover_from_cosia(
        self, dst_tif="landcover.tif", template_raster_path=None
    ):
        """
        crée le fichier tif du gdf de couverture du sol COSIA
        dst_tif: le fichier output
        """
        cosia_keys = {
            "Bâtiment": 2,
            "Zone imperméable": 1,
            "Zone perméable": 6,
            "Piscine": 7,
            "Serre": 1,
            "Sol nu": 6,
            "Surface eau": 7,
            "Neige": 7,
            "Conifère": 6,
            "Feuillu": 6,
            "Coupe": 5,
            "Broussaille": 5,
            "Pelouse": 5,
            "Culture": 5,
            "Terre labourée": 6,
            "Vigne": 5,
            "Autre": 1,
        }
        self.gdf["type"] = [cosia_keys[x] for x in self.gdf.classe]

        gdf_to_raster(
            dst_tif=dst_tif,  # os.path.join(self.output_path, dst_tif),
            gdf=self.gdf,
            measurement="type",
            resolution=(-1, 1),
            raster_file_like=template_raster_path,
            fill_value=None,
            dtype="float32",
        )


if __name__ == "__main__":
    from pymdu.image import geotiff
    from pymdu.geometric import Vegetation, Pedestrian, Water, Building
    from pymdu.geometric.Dem import Dem
    from pymdu.commons.BasicFunctions import plot_sol_occupancy

    import matplotlib.pyplot as plt

    geocore = GeoCore()
    geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]

    building = Building(output_path="./")
    buildings_gdf = building.run().to_gdf()
    # building.to_shp(name='buildings')

    water = Water(output_path="./")
    water_gdf = water.run().to_gdf()
    # water.to_shp(name='water')

    pedestrian = Pedestrian(output_path="./")
    pedestrian_gdf = pedestrian.run().to_gdf()
    # pedestrian.to_shp(name='pedestrian')

    vegetation = Vegetation(output_path="./", min_area=100)
    vegetation_gdf = vegetation.run().to_gdf()
    # vegetation.to_shp(name='vegetation')

    # cosia_gdf = gpd.read_file("../../demos/demo_cosia_gdf.shp")
    # dxf_gdf = gpd.read_f  #
    cosia_gdf = None
    dxf_gdf = None

    landcover = LandCover(
        output_path="./",
        building_gdf=buildings_gdf,
        vegetation_gdf=vegetation_gdf,
        water_gdf=water_gdf,
        cosia_gdf=cosia_gdf,
        dxf_gdf=dxf_gdf,
        pedestrian_gdf=pedestrian_gdf,
        write_file=False,
    )

    landcover.run()
    # landcover.to_shp(name="landcover")
    landcover_gdf = landcover.to_gdf()

    fig, ax = plt.subplots(figsize=(10, 10))

    if cosia_gdf is not None:
        landcover_gdf.plot(color=landcover_gdf["color"])
        fig_hist = plot_sol_occupancy(cosia_gdf, landcover_gdf)
        fig_hist.show()
    else:
        landcover_gdf.plot(ax=plt.gca(), edgecolor="black", column="type")

    plt.show()
    # génération landcover.tif

    dem = Dem(output_path="./")
    dem.run()

    # warp_options = gdal.WarpOptions(
    #     format="GTiff",
    #     outputType=gdalconst.GDT_Float32,
    #     # xRes=1, yRes=1,
    #     dstNodata=None,
    #     dstSRS='EPSG:2154',
    #     cropToCutline=True,
    #     cutlineDSName=r'./mask.shp',
    #     cutlineLayer='mask',
    #     resampleAlg='cubic'
    # )
    #
    # gdal.Warp(destNameOrDestDS=r'./DEM_ok.tif',
    #           srcDSOrSrcDSTab=r'./DEM.tif',
    #           options=warp_options)

    geotiff.clip_raster(
        dst_tif="./DEM_ok.tif",
        src_tif="./DEM.tif",
        format="GTiff",
        cut_shp="./mask.shp",
        cut_name="mask",
    )

    geotiff.gdf_to_raster(
        dst_tif="./landcover.tif",
        gdf=landcover_gdf,
        measurement="type",
        resolution=(-1, 1),
        raster_file_like=None,
        fill_value=None,
        dtype="float32",
    )

    geotiff.clip_raster(
        dst_tif="./landcover_ok.tif",
        src_tif="./landcover.tif",
        format="GTiff",
        cut_shp="./mask.shp",
        cut_name="mask",
    )

    ########## TESTING ##########
    # geotiff.raster_file_like(
    #     src_tif="./landcover_ok.tif",
    #     dst_tif="./landcover3.tif",
    #     like_path="./DEM_ok.tif",
    #     remove_nan=True,
    # )
    #
    # reshape_path = os.path.join("./", "reshape")
    # if not os.path.exists(reshape_path):
    #     shutil.rmtree(reshape_path, ignore_errors=True)
    #     os.makedirs(reshape_path, exist_ok=True)
    #
    # list_files = ["DEM", "landcover"]
    #
    # for file in list_files:
    #     umep_core = UmepCore(output_dir="./")
    #     (
    #         umep_core.run_processing(
    #             name="gdal:cliprasterbymasklayer",
    #             options={
    #                 "INPUT": os.path.join("./", f"{file}.tif"),
    #                 "MASK": os.path.join("./", "mask.shp"),
    #                 # 'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
    #                 # 'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:2154'),
    #                 "SOURCE_CRS": "EPSG:2154",
    #                 "TARGET_CRS": "EPSG:2154",
    #                 "TARGET_EXTENT": None,
    #                 "NODATA": None,
    #                 "ALPHA_BAND": False,
    #                 "CROP_TO_CUTLINE": True,
    #                 "KEEP_RESOLUTION": False,
    #                 "SET_RESOLUTION": False,
    #                 "X_RESOLUTION": None,
    #                 "Y_RESOLUTION": None,
    #                 "MULTITHREADING": False,
    #                 "OPTIONS": "",
    #                 "DATA_TYPE": 0,
    #                 "EXTRA": "",
    #                 "OUTPUT": os.path.join("./", f"{file}_test.tif"),
    #             },
    #         ),
    #     )
