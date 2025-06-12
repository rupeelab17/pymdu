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
import os.path

import geopandas as gpd

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.ign.IgnCollect import IgnCollect
from pymdu.commons.BasicFunctions import process_datetime


class Building(IgnCollect):
    """
    Class to collect the Building data.

    This class provides a method to read and process building data from a shapefile or GeoJSON file.
    It also calculates the mean height of the buildings, as well as their area and centroid.
    """

    def __init__(
        self,
        filepath_shp: str | None = None,
        output_path: str | None = None,
        defaultStoreyHeight: float = 3.0,
        set_crs: int | None = None,
    ):
        """
        Initializes the object with the given parameters.

        Args: filepath_shp (str): The file path to the shapefile. output_path (str): The output path for the
        processed data. If not provided, a default temporary path will be used. defaultStoreyHeight (float): The
        default height of each storey. set_crs (int): The CRS (Coordinate Reference System) to be set.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt

            plt.clf()  # markdown-exec: hide
            import pymdu.geometric.Building as Building

            buildings = Building(output_path='./')
            buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            buildings = buildings.run()
            buildings.to_gdf().plot(
                ax=plt.gca(),
                edgecolor='black',
                column='hauteur',
                legend=True,
                legend_kwds={'label': 'Hauteur', 'orientation': 'vertical'},
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
        self.defaultStoreyHeight = defaultStoreyHeight
        self.set_crs = set_crs

    def run(self):
        if not self.filepath_shp:
            self.execute_ign(key="buildings")
            file = (
                self.content
                if isinstance(self.content, io.BytesIO)
                else io.BytesIO(self.content)
            )

            gdf = gpd.read_file(file, driver="GeoJSON")
        else:
            gdf = gpd.read_file(self.filepath_shp, driver="ESRI Shapefile")

        if self.set_crs:
            gdf = gdf.set_crs(crs=self.set_crs, inplace=True, allow_override=True)
        else:
            gdf = gdf.to_crs(self._epsg)

        gdf["noHauteur"] = gdf["hauteur"].isnull()

        # calcul hauteur moyenne
        # ===========================
        gdf["area"] = gdf.apply(lambda x: x.geometry.area, axis=1)
        gdf["areaHauteur"] = gdf.apply(lambda x: x["area"] * x["hauteur"], axis=1)

        mean_distric_height = gdf["areaHauteur"].sum() / (gdf["area"].sum())

        if "nombre_d_etages" in gdf.columns:
            gdf["etage_nulle"] = gdf["nombre_d_etages"].isnull()
            gdf["hauteur"] = gdf.apply(
                lambda x: (
                    x["nombre_d_etages"] * self.defaultStoreyHeight
                    if x["noHauteur"] and not x["etage_nulle"]
                    else x["hauteur"]
                ),
                axis=1,
            )

        elif "HAUTEUR_2" in gdf.columns:
            gdf["noH2"] = gdf["HAUTEUR_2"].isnull()
            gdf["hauteur"] = gdf.apply(
                lambda x: (
                    x["HAUTEUR_2"] if x["noHauteur"] and not x["noH2"] else x["hauteur"]
                ),
                axis=1,
            )
        else:
            gdf["hauteur"] = gdf.apply(
                lambda x: mean_distric_height if x["noHauteur"] else x["hauteur"],
                axis=1,
            )

        gdf.dropna(subset=["hauteur"], inplace=True)

        self.gdf = gdf
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        """

        Returns:

        """
        gdf = process_datetime(gdf=self.gdf)
        return gdf

    def to_gpkg(self, name: str = "batiments"):
        """

        Args:
            name:
        """
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    buildings = Building(output_path="./")
    buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    buildings = buildings.run()
    print(buildings.to_gdf().head())
