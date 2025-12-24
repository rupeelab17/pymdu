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

from tqdm import tqdm

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.physics.aeraulic.uRockle import MainCalculation
from pymdu.physics.aeraulic.uRockle.GlobalVariables import *


class WindCalculation:
    """
    ===
    Classe qui permet
    - d'estimer la répartition du vent entre les bâtiments
    ===
    """

    def __init__(
        self,
        building_path,
        vegetation_path,
        wind_direction: float = 90,
        wind_speed: float = 2.0,
        wind_height: float = 2,
        mesh_size: int = 5,
        dz: int = 2,
        save_netcdf: bool = True,
        save_vector: bool = False,
        save_raster: bool = False,
        height_field: str = 'hauteur',
        output_path: str = TEMP_PATH,
    ):
        self.test = 'test'
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.wind_height = wind_height
        self.building_path = building_path
        self.vegetation_path = vegetation_path
        self.outputFilePath = output_path
        self.mesh_size = mesh_size
        self.dz = dz
        self.height_field = height_field
        self.save_netcdf = save_netcdf
        self.save_vector = save_vector
        self.save_raster = save_raster
        self.prefix = f'DIR_{self.wind_direction}_VENT_{int(self.wind_speed)}_'

    def run(self):
        temp_dir = tempfile.TemporaryDirectory(prefix=self.prefix)
        MainCalculation.main(
            pluginDirectory=os.getcwd(),
            windDirection=self.wind_direction,
            meshSize=self.mesh_size,
            dz=self.dz,
            v_ref=float(self.wind_speed),
            z_out=[self.wind_height],
            profileType='power',
            outputFilePath=self.outputFilePath,
            vegetationFilePath=self.vegetation_path,
            buildingFilePath=self.building_path,
            buildingHeightField=self.height_field,
            srid=3857,
            tempoDirectory=temp_dir.name,
            prefix=self.prefix,
            saveNetcdf=self.save_netcdf,
            saveVector=self.save_vector,
            saveRaster=self.save_raster,
        )
        temp_dir.cleanup()

    def run_from_windrose(self, path_windrose_analysis='windrose_calculation.csv'):
        data = pd.read_csv(path_windrose_analysis)
        for k, row in tqdm(data.iterrows()):
            print(float(row['vitesse']))
            print(int(row['direction']))
            temp_dir = tempfile.TemporaryDirectory(prefix=self.prefix)
            MainCalculation.main(
                pluginDirectory=os.getcwd(),
                windDirection=int(row['direction']),
                v_ref=float(row['vitesse']),
                z_out=[self.wind_height],
                profileType='power',
                outputFilePath=TEMP_PATH,
                vegetationFilePath=self.vegetation_path,
                buildingFilePath=self.building_path,
                srid=3857,
                tempoDirectory=temp_dir.name,
                prefix=f"DIR_{int(row['direction'])}_VENT_{int(float(row['vitesse']))}_",
                saveNetcdf=False,
                saveVector=True,
            )
            temp_dir.cleanup()

    def run_from_list(self, list='wind_list.csv'):
        data = pd.read_csv(list)
        for k, row in tqdm(data.iterrows()):
            print(float(row['vitesse']))
            print(int(row['direction']))
            temp_dir = tempfile.TemporaryDirectory(prefix=self.prefix)
            MainCalculation.main(
                pluginDirectory=os.getcwd(),
                windDirection=int(row['direction']),
                v_ref=float(row['vitesse']),
                z_out=[self.wind_height],
                profileType='power',
                outputFilePath=TEMP_PATH,
                vegetationFilePath=self.vegetation_path,
                buildingFilePath=self.building_path,
                srid=3857,
                tempoDirectory=temp_dir.name,
                prefix=f"DIR_{int(row['direction'])}_VENT_{int(float(row['vitesse']))}_",
                saveNetcdf=False,
                saveVector=True,
            )
            temp_dir.cleanup()


if __name__ == '__main__':
    test = WindCalculation()
    test.run_from_windrose()
