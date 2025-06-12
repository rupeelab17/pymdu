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
import os

import geopandas as gpd

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.osm.OsmCollect import OsmCollect

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Water(GeoCore):
    """
    ===
    Classe qui permet
    - de construire une reqûete pour interroger l'API de l'IGN
    - enregistre les données dans le dossier ./demo/
    ===
    """

    def __init__(
        self,
        filepath_shp: str | None = None,
        output_path: str | None = None,
        set_crs: int = None,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            filepath_shp: (str) The path to the shapefile to be processed. If not provided, the data will be read from the input file.
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            set_crs: (int) The EPSG code to set the CRS of the output file. If not provided, the CRS of the input file will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.Water as Water

            plt.clf()  # markdown-exec: hide
            water = Water(output_path='./')
            water.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            water = water.run()
            gdf = water.to_gdf()
            gdf.plot(ax=plt.gca(), edgecolor='black', color='blue', alpha=0.5)
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

    def run(self):
        if not self.filepath_shp:
            osm = OsmCollect(key='"natural"="water"')
            self.gdf = osm.run().to_gdf()
        else:
            self.gdf = gpd.read_file(self.filepath_shp, driver="ESRI Shapefile")

        if self.set_crs:
            self.gdf = self.gdf.set_crs(
                crs=self.set_crs, inplace=True, allow_override=True
            )
        else:
            self.gdf = self.gdf.to_crs(epsg=self._epsg)

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "water"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    water = Water(output_path="./")
    water.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    water = water.run()
    # water.to_shp(name="water")
    water_gdf = water.to_gdf()
    # vegetation.to_shp(name="vegetation")
    import matplotlib.pyplot as plt

    water_gdf.plot(edgecolor="k")
    plt.show()
