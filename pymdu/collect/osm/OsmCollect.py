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
import shutil

import geopandas as gpd
import orjson as json
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH


class OsmCollect(GeoCore):
    """
    ===
    Classe qui permet
    - de construire une reqûete pour interroger l'API de OpenStreetMap
    - enregistre les données dans le dossier ./demo/
    ===
    """

    def __init__(self, key: str = '"natural"="tree"'):
        self.geojson = None
        self.gdf = None
        self.key = key

    def run(self):
        overpass = Overpass()
        query = overpassQueryBuilder(
            bbox=[self._bbox[1], self._bbox[0], self._bbox[3], self._bbox[2]],
            elementType=['way', 'relation', 'node'],
            selector=self.key,
            includeGeometry=True,
        )

        result = overpass.query(query)
        result.countElements()
        jsonList = result.elements()
        geojsonEmpty = {
            'type': 'FeatureCollection',
            'name': 'OSMPythonTools',
            'features': [],
        }
        for k, item in enumerate(jsonList):
            toAdd = {
                'type': 'Feature',
                'geometry': {
                    'type': item.geometry()['type'],
                    'coordinates': item.geometry()['coordinates'],
                },
                'properties': {'name': '{}{}'.format(self.key, k)},
            }
            geojsonEmpty['features'].append(toAdd)

        self.geojson = json.dumps(geojsonEmpty).decode('utf-8')
        shutil.rmtree('cache')

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        print(self.geojson)
        file = io.StringIO(self.geojson)
        gdf = gpd.read_file(file)
        self.gdf = gdf.to_crs(self._epsg)
        return self.gdf

    def to_shp(self, name_file='example'):
        self.gdf.to_file(os.path.join(TEMP_PATH, f'{name_file}.shp'))
        return
