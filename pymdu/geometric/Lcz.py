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
import os.path

import geopandas as gpd
from shapely.geometry.geo import box

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Lcz(GeoCore):
    """
    Class to collect the Building data
    """

    def __init__(
        self,
        filepath_shp: str = None,
        output_path: str = None,
        set_crs: int = None,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            filepath_shp (str): The file path to the shapefile.
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            set_crs (int): The CRS (Coordinate Reference System) to be set.

        Example:
            ```python exec="false" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            plt.clf()  # markdown-exec: hide
            import pymdu.geometric.Lcz as Lcz

            lcz = Lcz()
            lcz_gdf = lcz.run().to_gdf()
            lcz.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            table_color = lcz.table_color
            fig, ax = plt.subplots(figsize=(10, 10))
            lcz_gdf.plot(ax=ax, edgecolor=None, color=lcz_gdf['color'])
            patches = [
                mpatches.Patch(color=info[1], label=info[0])
                for info in table_color.values()
            ]
            plt.legend(
                handles=patches,
                loc='upper right',
                title='LCZ Legend',
                bbox_to_anchor=(1.1, 1.0),
            )
            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
            * For module TODOs
        """
        super().__init__()
        self.filepath_shp = filepath_shp
        self.output_path = output_path if output_path else TEMP_PATH
        self.set_crs = set_crs
        self.table_color = {
            1: ['LCZ 1: Compact high-rise', '#8b0101'],
            2: ['LCZ 2: Compact mid-rise', '#cc0200'],
            3: ['LCZ 3: Compact low-rise', '#fc0001'],
            4: ['LCZ 4: Open high-rise', '#be4c03'],
            5: ['LCZ 5: Open mid-rise', '#ff6602'],
            6: ['LCZ 6: Open low-rise', '#ff9856'],
            7: ['LCZ 7: Lightweight low-rise', '#fbed08'],
            8: ['LCZ 8: Large low-rise', '#bcbcba'],
            9: ['LCZ 9: Sparsely built', '#ffcca7'],
            10: ['LCZ 10: Heavy industry', '#57555a'],
            11: ['LCZ A: Dense trees', '#006700'],
            12: ['LCZ B: Scattered trees', '#05aa05'],
            13: ['LCZ C: Bush,scrub', '#648423'],
            14: ['LCZ D: Low plants', '#bbdb7a'],
            15: ['LCZ E: Bare rock or paved', '#010101'],
            16: ['LCZ F: Bare soil or sand', '#fdf6ae'],
            17: ['LCZ G: Water', '#6d67fd'],
        }

    def run(
        self,
        zipfile_url: str = 'zip+https://static.data.gouv.fr/resources/cartographie-des-zones-climatiques-locales-lcz-de'
        '-83-aires-urbaines-de-plus-de-50-000-habitants-2022/20241011-113952/lcz-spot-2022-la'
        '-rochelle.zip/LCZ_SPOT_2022_La Rochelle.shp',
    ):
        gdf = gpd.read_file(zipfile_url, driver='ESRI Shapefile')
        gdf1 = gdf[['lcz_int', 'geometry']].copy()
        gdf1['color'] = [self.table_color[x][1] for x in gdf1['lcz_int']]
        gdf1 = gdf1.to_crs(self._epsg)
        bbox_final = box(self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])
        gdf_bbox_mask = gpd.GeoDataFrame(
            gpd.GeoSeries(bbox_final), columns=['geometry'], crs='epsg:4326'
        )
        gdf_bbox_mask = gdf_bbox_mask.to_crs(self._epsg)

        self.gdf = gpd.overlay(
            df1=gdf1, df2=gdf_bbox_mask, how='intersection', keep_geom_type=False
        )
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = 'lcz'):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')


if __name__ == '__main__':
    # https://gdal.org/en/latest/user/virtual_file_systems.html
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    lcz = Lcz()
    lcz.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    lcz_gdf = lcz.run().to_gdf()
    table_color = lcz.table_color

    # Tracer le GeoDataFrame
    fig, ax = plt.subplots(figsize=(10, 10))
    lcz_gdf.plot(ax=ax, edgecolor=None, color=lcz_gdf['color'])

    # Créer les patches pour chaque couleur et sa description dans la légende
    patches = [
        mpatches.Patch(color=info[1], label=info[0]) for info in table_color.values()
    ]

    # Ajouter la légende personnalisée
    plt.legend(
        handles=patches,
        loc='upper right',
        title='LCZ Legend',
        bbox_to_anchor=(1.1, 1.0),
    )

    # Afficher la carte avec la légende
    plt.show()
