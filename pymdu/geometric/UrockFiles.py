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

import math
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely import Polygon

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH


class UrockFiles(GeoCore):
    """
    classdocs
    """

    def __init__(
        self,
        output_path: str = None,
        buildings_gdf: gpd.GeoDataFrame = None,
        trees_gdf: gpd.GeoDataFrame = None,
    ):
        self.output_path = output_path if output_path else TEMP_PATH
        self.buildings_gdf = buildings_gdf
        self.trees_gdf = trees_gdf

    def generate_urock_buildings(
        self, filename_shp='urock_bld.shp'
    ) -> gpd.GeoDataFrame:
        """

        Returns:

        """
        buildings = self.buildings_gdf
        buildings_urock = buildings.to_crs(3857)
        buildings_urock.to_file(os.path.join(self.output_path, filename_shp))

        return buildings_urock

    def __flat_hex_coords(self, centre, size, i):
        """Return the point coordinate of a flat-topped regular hexagon.
        points are returned in counter-clockwise order as i increases
        the first coordinate (i=0) will be:
        centre.x + size, centre.y
        """
        angle_deg = 60 * i
        angle_rad = math.pi / 180 * angle_deg
        return (
            centre.x + size * math.cos(angle_rad),
            centre.y + size * math.sin(angle_rad),
        )

    def __flat_hex_polygon(self, centre, size):
        """Return a flat-topped regular hexagonal Polygon, given a centroid Point and side length"""
        return Polygon([self.__flat_hex_coords(centre, size, i) for i in range(6)])

    def generate_urock_trees(
        self,
        filename_shp='urock_trees.shp',
        size: int = 6,
        ID_VEG: int = 5,
        MIN_HEIGHT: float = 2.2,
        MAX_HEIGHT: float = 5.8,
        ATTENUATIO: float = 2.8,
    ) -> gpd.GeoDataFrame:
        """
        Returns:
        """
        trees = self.trees_gdf
        trees = trees.to_crs('3857')
        liste_arbres = []
        for geom in trees['geometry']:
            liste_arbres.append(self.__flat_hex_polygon(geom, size))
        arbres_urock = gpd.GeoDataFrame(crs='epsg:3857', geometry=liste_arbres)
        arbres_urock['ID_VEG'] = [ID_VEG for x in trees['geometry']]
        arbres_urock['MIN_HEIGHT'] = [MIN_HEIGHT for x in trees['geometry']]
        arbres_urock['MAX_HEIGHT'] = [MAX_HEIGHT for x in trees['geometry']]
        arbres_urock['ATTENUATIO'] = [ATTENUATIO for x in trees['geometry']]
        arbres_urock.to_file(os.path.join(self.output_path, filename_shp))

        return arbres_urock

    @staticmethod
    def convert_umep_index_to_date(
        meteo_path='LaRochelle_rcp85_IPSL_bc_type_list_UMEP.txt', year='2022'
    ):
        meteo_path = meteo_path
        meteo = pd.read_csv(meteo_path, sep=' ')
        year = year
        liste_of_date = [
            datetime.strptime(year + '-' + str(day), '%Y-%j') for day in meteo['id']
        ]
        date = [
            res.strftime(f'%Y-%m-%d {hour}:00:00')
            for (res, hour) in zip(liste_of_date, meteo['it'])
        ]
        date = [pd.to_datetime(x) for x in date]
        return date


if __name__ == '__main__':
    # test_simon
    # ================
    path_results = './Ressources'
    path_trees = os.path.join(path_results, 'trees.shp')
    path_buildings = os.path.join(path_results, 'buildings.shp')

    try_urock_gen = UrockFiles(
        output_path=path_results,
        buildings_gdf=gpd.read_file(path_buildings),
        trees_gdf=gpd.read_file(path_trees),
    )
    urock_buildings_gdf = try_urock_gen.generate_urock_buildings(
        filename_shp='urock_bld.shp'
    )
    urock_trees_gdf = try_urock_gen.generate_urock_trees(filename_shp='urock_trees.shp')
