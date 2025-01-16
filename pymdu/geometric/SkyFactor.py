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

import sys

import geopandas as gpd
from shapely.geometry import box
from t4gpd.morph.STExtractOpenSpaces import STExtractOpenSpaces
from t4gpd.morph.STGrid import STGrid
from t4gpd.morph.geoProcesses.STGeoProcess import STGeoProcess
from t4gpd.morph.geoProcesses.SkyViewFactor import SkyViewFactor

from pymdu.GeoCore import GeoCore

sys.setrecursionlimit(10000)


class SkyFactor(GeoCore):
    """
    classdocs
    """

    def __init__(
        self, buildings_gdf: gpd.GeoDataFrame, elevationFieldname: str = 'hauteur'
    ):
        """
        Initializes the object with the given parameters.

        Args:
            buildings_gdf (gpd.GeoDataFrame): GeoDataFrame of the buildings.
            elevationFieldname (str): Name of the elevation field.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.SkyFactor as SkyFactor
            import pymdu.geometric.Building as Building

            plt.clf()  # markdown-exec: hide
            building = Building(output_path='./')
            buildings_gdf = building.run().to_gdf()
            sky_factor = SkyFactor(
                buildings_gdf=buildings_gdf, elevationFieldname='hauteur'
            )
            sky_factor.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            sky_factor = road.run()
            sky_factor_gdf = sky_factor.to_gdf()
            _, basemap = plt.subplots(figsize=(8.26, 8.26))
            basemap.set_title('Sky View Factor', fontsize=16)
            plt.axis('off')
            buildings_gdf.plot(ax=basemap, color='grey')
            sky_factor_gdf.plot(
                ax=basemap, column='svf', markersize=8, legend=True, cmap='viridis'
            )
            plt.legend(loc='upper right', framealpha=0.5)
            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
             * For module TODOs
        """
        self.buildings_gdf = buildings_gdf
        self.sensors = None
        self.characteristicLength = 20
        self.nRays = 10
        self.dx = 30
        self.maxRayLen = 10
        self.elevationFieldname = elevationFieldname

    def run(self):
        roi = box(
            self.buildings_gdf.bounds.minx.min(),
            self.buildings_gdf.bounds.miny.min(),
            self.buildings_gdf.bounds.maxx.max(),
            self.buildings_gdf.bounds.maxy.max(),
        )
        gdf = gpd.GeoDataFrame(index=[0], crs=f'epsg:{self._epsg}', geometry=[roi])
        void = STExtractOpenSpaces(gdf, self.buildings_gdf).run()
        void = void.explode(ignore_index=True)
        void = void[void.area > 100]
        void.geometry = void.simplify(tolerance=1.0, preserve_topology=True)
        buildings = self.buildings_gdf.explode(ignore_index=True)

        # maillage régulier
        sensors = STGrid(
            buildings, dx=self.dx, dy=None, indoor=False, intoPoint=True
        ).run()

        op = SkyViewFactor(
            buildings,
            nRays=self.nRays,
            maxRayLen=self.maxRayLen,
            elevationFieldname=self.elevationFieldname,
            method=2018,
            background=True,
        )
        try:
            self.sensors = STGeoProcess(op, sensors).run()
        except:
            print('Merci de vérifier la géométrie du GeoDataFrame')
            sys.exit(1)

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.sensors


if __name__ == '__main__':
    from pymdu.geometric import Building

    geocore = GeoCore()
    geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]

    building = Building(output_path='./')
    buildings_gdf = building.run().to_gdf()

    sky_factor = SkyFactor(buildings_gdf=buildings_gdf, elevationFieldname='hauteur')
    sky_factor.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    sky_factor = sky_factor.run()
    # routes.to_shp(name="routes")
    sky_factor_gdf = sky_factor.to_gdf()
    import matplotlib.pyplot as plt

    _, basemap = plt.subplots(figsize=(8.26, 8.26))
    basemap.set_title('Sky View Factor', fontsize=16)
    plt.axis('off')
    buildings_gdf.plot(ax=basemap, color='grey')
    sky_factor_gdf.plot(
        ax=basemap, column='svf', markersize=8, legend=True, cmap='viridis'
    )
    plt.legend(loc='upper right', framealpha=0.5)
    plt.show()
