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
import os.path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry.point import Point

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Rnb(GeoCore):
    """ """

    def __init__(self, output_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.Building as Building
            import pymdu.geometric.Rnb as Rnb

            plt.clf()  # markdown-exec: hide
            rnb = Rnb(output_path='./')
            rnb.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            rnb = rnb.run()
            rnb_gdf = rnb.to_gdf()
            buildings = Building(output_path='./')
            buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            buildings_gdf = buildings.run().to_gdf()
            ax = rnb_gdf.plot(ax=plt.gca(), edgecolor='black', color='red')
            buildings_gdf.plot(ax=ax, edgecolor='black', alpha=0.5)
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
        url = "https://rnb-api.beta.gouv.fr/api/alpha/buildings"
        headers = {"Content-type": "application/json"}

        payload = {
            "bb": f"{self._bbox[1]},{self._bbox[0]},{self._bbox[3]},{self._bbox[2]}"
        }
        print(payload["bb"])
        response = requests.get(url=url, headers=headers, params=payload, verify=False)
        content = response.json()

        list_gdf = []
        for item in content["results"]:
            print(item)

            # Extraire les coordonnées pour la géométrie
            coordinates = item["point"]["coordinates"]
            geometry = [Point(coordinates)]
            object = {
                "rnb_id": item["rnb_id"],
                "status": item["status"],
            }

            if len(item["addresses"]) > 0:
                object.update(
                    {
                        "street_number": item["addresses"][0]["street_number"],
                        "city_name": item["addresses"][0]["city_name"],
                        "city_zipcode": item["addresses"][0]["city_zipcode"],
                        "created_at": item["ext_ids"][0]["created_at"],
                    }
                )
            # Créer un DataFrame à partir des données
            df = pd.DataFrame([object])

            # Convertir en GeoDataFrame en utilisant les coordonnées pour la géométrie
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            gdf = gdf.to_crs(self._epsg)
            list_gdf.append(gdf)

        self.gdf = gpd.GeoDataFrame(pd.concat(list_gdf, ignore_index=True))
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "rnb"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    import pymdu.geometric.Building as Building
    import matplotlib.pyplot as plt

    rnb = Rnb(output_path="./")
    rnb.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    rnb = rnb.run()
    rnb_gdf = rnb.to_gdf()
    buildings = Building(output_path="./")
    buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    buildings_gdf = buildings.run().to_gdf()
    ax = rnb_gdf.plot(ax=plt.gca(), edgecolor="black", color="red")
    buildings_gdf.plot(ax=ax, edgecolor="black", alpha=0.5)
    plt.show()
