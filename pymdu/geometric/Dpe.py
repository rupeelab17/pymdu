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
import urllib.parse

import geopandas as gpd
import requests
from shapely.geometry.point import Point
from sqlalchemy.dialects.mssql.information_schema import columns

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Dpe(GeoCore):
    # TODO : déplacer dans collect
    """
    Class to collect the Cadastre data
    """

    def __init__(self, output_path: str = None, columns: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import pymdu.geometric.Dpe as Dpe
            import pymdu.geometric.Building as Building

            plt.clf()  # markdown-exec: hide
            dpe = Dpe()
            dpe.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            dpe_gdf = dpe.run().to_gdf()
            table_color = dpe.table_color
            buildings = Building(output_path='./')
            buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            buildings_gdf = buildings.run().to_gdf()
            # color_mapping = {"category1": "green", "category2": "red"}
            ax = dpe_gdf.plot(
                ax=plt.gca(),
                edgecolor='black',
                categorical=True,
                column='classe_consommation_energie',
                legend=True,
                color=dpe_gdf['color'],
            )
            patches = [
                mpatches.Patch(color=info[1], label=info[0])
                for info in table_color.values()
            ]
            plt.legend(
                handles=patches,
                loc='upper right',
                title='Etiquette DPE',
                bbox_to_anchor=(1.1, 1.0),
            )
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

        if columns is None:
            columns = "consommation_energie,classe_consommation_energie,annee_construction,latitude,longitude,geo_adresse"

        self.columns: str = columns

        self.output_path = output_path if output_path else TEMP_PATH
        self.base_url = (
            r"https://data.ademe.fr/data-fair/api/v1/datasets/dpe-france/geo_agg?"
        )
        self.table_color = {
            "A": ["A: ≤50", "#5ebd46"],  # Green for class A
            "B": ["B: 51 à 90", "#a0d468"],  # Light Green for class B
            "C": ["C: 91 à 150", "#e6e600"],  # Yellow Green for class C
            "D": ["D: 151 à 230", "#ffcc33"],  # Yellow for class D
            "E": ["E: 231 à 330", "#f39c12"],  # Light Orange for class E
            "F": ["F: 331 à 450", "#e67e22"],  # Orange for class F
            "G": ["G: >450", "#e74c3c"],  # Red for class G
            "N": ["N: null", "#2323"],  # 2323 for class N
        }

    def run(self, all_values=False):
        headers = {"Content-type": "application/json"}

        payload = {
            "agg_size": "1000",
            "q_mode": "simple",
            "bbox": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
            "size": "1000",
            "sort": "geo_adresse",
            "select": self.columns,
            "highlight": "nom_methode_dpe",
            "sampling": "neighbors",
            "format": "geojson",
        }
        if all_values:
            payload = {
                "agg_size": "1000",
                "q_mode": "simple",
                "bbox": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                "size": "1000",
                "sort": "geo_adresse",
                "highlight": "nom_methode_dpe",
                "sampling": "neighbors",
                "format": "geojson",
            }
        new_url = self.__get_url_dpe(self.base_url, payload)
        response = requests.get(url=new_url, headers=headers, verify=False)
        geojson_data = response.json()
        try:
            # Initialize lists to store data
            rows = []

            # Iterate through each feature in the collection
            for feature in geojson_data["features"]:
                geometry = Point(feature["geometry"]["coordinates"])
                results = feature["properties"]["results"]
                for result in results:
                    result["geometry"] = geometry
                    rows.append(result)

            # Convert to GeoDataFrame
            self.gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326", geometry="geometry")
            self.gdf = self.gdf.to_crs(self._epsg)
            self.gdf["color"] = [
                self.table_color[x][1] for x in self.gdf["classe_consommation_energie"]
            ]
        except Exception as e:
            print("ERROR Dpe Class ==>", e)
        return self

    @staticmethod
    def __get_url_dpe(url, payload):
        new_url = url + "&" + urllib.parse.urlencode(payload)
        return new_url

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "dpe"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pymdu.geometric.Building as Building

    dpe = Dpe()
    dpe.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    dpe_gdf = dpe.run().to_gdf()

    table_color = dpe.table_color

    buildings = Building(output_path="./")
    buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    buildings_gdf = buildings.run().to_gdf()

    # color_mapping = {"category1": "green", "category2": "red"}
    ax = dpe_gdf.plot(
        ax=plt.gca(),
        edgecolor="black",
        categorical=True,
        column="classe_consommation_energie",
        legend=True,
        color=dpe_gdf["color"],
    )
    patches = [
        mpatches.Patch(color=info[1], label=info[0]) for info in table_color.values()
    ]
    plt.legend(
        handles=patches,
        loc="upper right",
        title="Etiquette DPE",
        bbox_to_anchor=(1.1, 1.0),
    )

    buildings_gdf.plot(ax=ax, edgecolor="black", alpha=0.5)

    buildings_gdf.plot(
        ax=plt.gca(),
    )
    plt.show()

    buildings_gdf = buildings_gdf.to_crs(4326)

    print(buildings_gdf.columns)
    colonnes_mat = [
        col for col in buildings_gdf.columns if col.startswith("materiaux_")
    ]

    # Ajout explicite de la géométrie
    colonnes_mat.append("geometry")

    buildings_gdf[colonnes_mat].to_file("buildings.geojson", driver="GeoJSON")
    print(buildings_gdf[colonnes_mat])
