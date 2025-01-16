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

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class IsochroneIGN(IgnCollect):
    """
    Class to collect the Building data
    """

    def __init__(
        self,
        output_path: str = None,
        resource: str = 'bdtopo-valhalla',
        point: tuple = (2.337306, 48.849319),
        costValue: int = 300,
        costType: str = 'time',
        set_crs: int = None,
    ):
        """
        Obtenir une surface géo-localisée représentant l’ensemble des points atteignables à partir d’un point de départ. Les points de départ et d’arrivée peuvent être inversés: on obtient alors la liste des points de départs possibles permettant d’atteindre un point d’arrivée donné. On peut aussi fournir un critère de distance plutôt que de temps: on parle alors de calcul d’iso-distances.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            resource (str): The resource to use.
            point (tuple): Coordonnées d'une position ponctuelle. C'est le point à partir duquel seront fait les calculs. Il devra être exprimé dans le CRS, par défaut, de la ressource (voir le paramètre 'crs' dans le GetCapabilities).
            costValue (int): The cost value. Valeur du coût utilisé pour le calcul. Les valeurs disponibles et la valeur par défaut utilisées sont présentes dans le GetCapabilities. On pourra, par exemple, préciser une distance ou un temps, selon l'optimisation choisie. L'unité dépendra aussi des paramètres distanceUnit et timeUnit.
            costType (str): The cost type. Type du coût utilisé pour le calcul. Les valeurs disponibles et la valeur par défaut utilisées sont présentes dans le GetCapabilities. On pourra, par exemple, préciser une distance ou un temps, selon l'optimisation choisie. L'unité dépendra aussi des paramètres distanceUnit et timeUnit
            set_crs (int): The CRS (Coordinate Reference System) to be set.


        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt

            plt.clf()  # markdown-exec: hide
            import pymdu.geometric.IsochroneIGN as IsochroneIGN

            isochroneIGN = IsochroneIGN(output_path='./')
            isochroneIGN.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            isochroneIGN = isochroneIGN.run()
            isochroneIGN.to_gdf().plot(ax=plt.gca(), edgecolor='black', legend=True)
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
        self.resource = (resource,)
        self.point = (point,)
        self.costValue = (costValue,)
        self.costType = costType
        self.set_crs = set_crs

    def run(self):
        payload = dict(
            resource=self.resource,
            point=self.point,
            costValue=self.costValue,
            costType=self.costType,
        )

        self.execute_ign(key='isochrone', **payload)
        file = (
            self.content
            if isinstance(self.content, io.BytesIO)
            else io.BytesIO(self.content)
        )

        gdf = gpd.read_file(file, driver='GeoJSON')

        if self.set_crs:
            gdf = gdf.set_crs(crs=self.set_crs, inplace=True, allow_override=True)
        else:
            gdf = gdf.to_crs(self._epsg)

        self.gdf = gdf
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = 'isochrone'):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')


if __name__ == '__main__':
    isochroneIGN = IsochroneIGN(output_path='./')
    isochroneIGN.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    isochroneIGN = isochroneIGN.run()
    import matplotlib.pyplot as plt

    plt.clf()
    isochroneIGN.to_gdf().plot(ax=plt.gca(), edgecolor='black', legend=True)
    plt.show()
