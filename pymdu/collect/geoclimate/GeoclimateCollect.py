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

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH


class GeoclimateCollect(GeoCore):
    """
    ===
    Classe qui permet
    - de lancer geoclimate
    ===
    """

    def __init__(self, output_path: str | None = None):
        super().__init__()
        self.output_path = output_path if output_path else TEMP_PATH
        self.dir_output_path = os.path.join(
            self.output_path, self.geoclimate_temp_directory
        )

    def run(self):
        if os.path.exists(self.dir_output_path):
            shutil.rmtree(self.dir_output_path, ignore_errors=True)

        os.makedirs(self.dir_output_path)
        pathJson = str(
            self.collect_path.joinpath("geoclimate/my_first_config_file_osm.json")
        )
        pathtoGo = str(self.collect_path.joinpath("geoclimate/toGo.json"))
        pathJarGeoclimate = str(
            self.collect_path.joinpath("geoclimate/geoclimate-0.0.2.jar")
        )
        print("pathJarGeoclimate", pathJarGeoclimate)

        with open(pathJson) as f:
            myJson = json.load(f)
            myJson["input"]["locations"] = [
                [self._bbox[1], self._bbox[0], self._bbox[3], self._bbox[2]]
            ]
            # os.makedirs("layers", exist_ok=True)
            myJson["output"]["folder"] = self.output_path

            with open(pathtoGo, "w") as f:
                json.dump(myJson, f)
                f.close()

        proc = subprocess.Popen(
            ["java", "-jar", pathJarGeoclimate, "-w ", "OSM", "-f", pathtoGo],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        (output, error_output) = proc.communicate()

        # if not os.path.exists(os.path.join(TEMP_PATH, self.geoclimate_temp_directory)):
        #     os.mkdir(os.path.join(TEMP_PATH, self.geoclimate_temp_directory))
        # else:
        #     shutil.rmtree(os.path.join(TEMP_PATH, self.geoclimate_temp_directory))

        # src = glob.glob(os.path.join(self.dir_output_path, 'osm*'))[0]
        # os.rename(src, self.dir_output_path)
        return self

    def blocs(self):
        # if os.path.exists(TEMP_PATH + '/geoclimate-data/rsu_indicators.geojson'):
        #
        #     bloc = gpd.read_file(TEMP_PATH + '/geoclimate-data/rsu_indicators.geojson')
        # else:
        #     self.run()

        bloc = gpd.read_file(
            os.path.join(self.dir_output_path, "rsu_indicators.geojson")
        )
        # bloc = bloc[['BUILDING_TOTAL_FRACTION', 'geometry']]
        # bloc = bloc.rename(columns={"BUILDING_TOTAL_FRACTION": "surf_ratio"})
        bloc = bloc.to_crs(self._epsg)

        building_blocks = gpd.read_file(
            os.path.join(self.dir_output_path, "block_indicators.geojson")
        )
        building_blocks = building_blocks.to_crs(self._epsg)
        return bloc, building_blocks

    def to_shp(self, remove_geojson: bool = False) -> None:
        all_files_geojson = glob.glob(os.path.join(self.dir_output_path, "*.geojson"))
        for file_geojson in all_files_geojson:
            filename = Path(file_geojson).stem
            gdf = gpd.read_file(
                os.path.join(self.dir_output_path, f"{filename}.geojson"),
                driver="GeoJSON",
            )
            try:
                from pandas.api.types import is_datetime64_any_dtype as is_datetime

                gdf = gdf[
                    [column for column in gdf.columns if not is_datetime(gdf[column])]
                ]
            except Exception as e:
                print(f"ERROR {filename} to_shp ==>", e)
                pass
            gdf.to_file(
                os.path.join(self.dir_output_path, f"{filename}.shp"), "ESRI Shapefile"
            )
            if remove_geojson:
                os.remove(file_geojson)
        return

    def to_gdf(self, name: str) -> gpd.GeoDataFrame:
        raise NotImplementedError


if __name__ == "__main__":
    geoclimate = GeoclimateCollect(output_path="/Users/Boris/Downloads")
    geoclimate.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    geoclimate.run().to_shp(remove_geojson=True)
