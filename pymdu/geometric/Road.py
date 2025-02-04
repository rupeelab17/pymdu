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
import io
import os
import os.path

import geopandas as gpd

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.ign.IgnCollect import IgnCollect

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Road(IgnCollect):
    """
    Class to collect the Road data
    """

    def __init__(self, output_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.Road as Road

            plt.clf()  # markdown-exec: hide
            road = Road(output_path='./')
            road.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            road = road.run()
            road_gdf = road.to_gdf()
            ax = road_gdf.plot(ax=plt.gca(), edgecolor='black', color='red')
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
        self.output_path = output_path if output_path else TEMP_PATH

    def run(self):
        self.execute_ign(key="road")
        try:
            file = (
                self.content
                if isinstance(self.content, io.BytesIO)
                else io.BytesIO(self.content)
            )
            gdf = gpd.read_file(file, driver="GeoJSON")
            self.gdf = gdf.to_crs(self._epsg)
        except Exception as e:
            print("ERROR Road Class ==>", e)
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "routes"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    routes = Road(output_path="./")
    routes.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    routes = routes.run()
    # routes.to_shp(name="routes")
    routes_gdf = routes.to_gdf()
    import matplotlib.pyplot as plt

    routes_gdf.plot(edgecolor="k")
    plt.show()
