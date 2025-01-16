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
import numpy as np
from shapely.geometry import Point

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.ign.IgnCollect import IgnCollect

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Iris(IgnCollect):
    """ """

    def __init__(self, output_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.Iris as Iris

            plt.clf()  # markdown-exec: hide
            iris = Iris(output_path='./')
            iris.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            iris = iris.run()
            gdf = iris.to_gdf()
            gdf.plot(ax=plt.gca())
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
        self.execute_ign(key='iris')
        file = (
            self.content
            if isinstance(self.content, io.BytesIO)
            else io.BytesIO(self.content)
        )
        gdf = gpd.read_file(file, driver='GeoJSON')
        gdf = gdf.to_crs(self._epsg)
        self.gdf = gdf
        return self

    def centroid(self):  # rectangle centroid
        coord = self.exterior.coords
        face_p0 = np.array(coord[0])
        face_p2 = np.array(coord[2])
        face_ce = (face_p0 + face_p2) / 2
        return Point(face_ce)

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = 'iris'):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')


if __name__ == '__main__':
    iris = Iris(output_path='./')
    iris.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    iris = iris.run()
    # iris.to_shp(name="iris")
    import matplotlib.pyplot as plt

    iris.to_gdf().plot(figsize=(10, 10))
    plt.show()
