import os
import shutil
from itertools import product
from typing import Sequence, Iterator, Any, Iterable

import joblib
from joblib import delayed, Parallel
from pymdu.physics.aeraulic import WindCalculation
from pymdu.physics.umep.UmepCore import UmepCore


class Urock(UmepCore):
    def __init__(
        self,
        output_dir: os.path,
        working_directory: os.path,
        input_filepath_buildings_shp: os.path = 'urock_bld.shp',
        input_filepath_trees_shp: os.path = 'urock_trees.shp',
        java_home_path: os.path = None,
    ):
        super().__init__(output_dir=working_directory)

        self.input_filepath_buildings_shp = input_filepath_buildings_shp
        self.java_home_path = java_home_path
        self.input_filepath_trees_shp = input_filepath_trees_shp
        self.working_directory = working_directory
        self.output_dir = output_dir

    def run(
        self,
        output_dir=None,
        wind_speed: float = 1,
        wind_direction: int = 180,
        wind_height=1.5,
        height_field='hauteur',
        horizontal_resolution: int = 5,
        vertical_resolution: int = 2,
        save_netcdf: bool = True,
        save_raster: bool = True,
        save_vector: bool = True,
    ):
        if output_dir:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                # shutil.rmtree(output_dir, ignore_errors=True)

        a = WindCalculation(
            output_path=self.output_dir,
            building_path=self.input_filepath_buildings_shp,
            vegetation_path=self.input_filepath_trees_shp,
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            wind_height=wind_height,
            mesh_size=horizontal_resolution,
            dz=vertical_resolution,
            height_field=height_field,
            save_netcdf=save_netcdf,
            save_raster=save_raster,
            save_vector=save_vector,
        )
        try:
            a.run()
        except Exception as e:
            print('Error: {}'.format(e), flush=True)

        return None

    def run_with_umep(
        self,
        output_dir=None,
        height_field='hauteur',
        wind_speed: float = 5,
        wind_direction: int = 180,
        max_height_field='MAX_HEIGHT',
        min_height_field='MIN_HEIGHT',
        wind_height=1.5,
        horizontal_resolution=5,
        vertical_resolution=5,
        save_netcdf=True,
        save_raster=True,
        save_vector=True,
    ):
        if output_dir:
            self.output_dir = output_dir
            # if os.path.exists(output_dir):
            #     shutil.rmtree(output_dir, ignore_errors=True)
            os.makedirs(output_dir)

        self.run_processing(
            name='umep:Urban Wind Field: URock',
            options={
                'BUILDINGS': self.input_filepath_buildings_shp,
                'HEIGHT_FIELD_BUILD': height_field,
                'VEGETATION': self.input_filepath_trees_shp,
                'VEGETATION_CROWN_TOP_HEIGHT': max_height_field,
                'VEGETATION_CROWN_BASE_HEIGHT': min_height_field,
                'ATTENUATION_FIELD': 'ATTENUATIO',
                'INPUT_PROFILE_FILE': '',
                'INPUT_PROFILE_TYPE': 0,
                'INPUT_WIND_HEIGHT': 10,
                'INPUT_WIND_SPEED': wind_speed,
                'INPUT_WIND_DIRECTION': wind_direction,
                'RASTER_OUTPUT': None,
                'HORIZONTAL_RESOLUTION': horizontal_resolution,
                'VERTICAL_RESOLUTION': vertical_resolution,
                'WIND_HEIGHT': wind_height,
                'UROCK_OUTPUT': self.output_dir,
                'OUTPUT_FILENAME': f'urock_{wind_direction}',
                'SAVE_RASTER': save_raster,
                'SAVE_VECTOR': save_vector,
                'SAVE_NETCDF': save_netcdf,
                'LOAD_OUTPUT': False,
                'JAVA_PATH': self.java_home_path,
            },
        )
        return None

    def processJobs(self, combinaisons, backend, **kwargs):
        self.qgsApp = None
        number_of_cpu = (
            joblib.cpu_count() - 1
        )  # with parallel_backend("loky", inner_max_num_threads=2):
        #     results = Parallel(n_jobs=number_of_cpu)(delayed(urock)(combinaison) for combinaison in combinaisons)
        with Parallel(
            backend=backend, verbose=10, timeout=350, n_jobs=number_of_cpu
        ) as parallel:
            delayed_funcs = [
                delayed(self.run)(
                    wind_direction=combinaison[0],
                    wind_speed=combinaison[1],
                    output_dir=os.path.join(
                        self.output_dir,
                        f'wind_direction_{combinaison[0]}_wind_speed_{combinaison[1]}',
                    ),
                    **kwargs,
                )
                for i, combinaison in enumerate(combinaisons)
            ]
            parallel(delayed_funcs)

    @staticmethod
    def get_chunks(sequence: Sequence[Any], chunk_size: int) -> Iterator[Any]:
        for i in range(0, len(sequence), chunk_size):
            yield sequence[i : i + chunk_size]

    def run_parallel(
        self,
        winds_speed: list,
        winds_direction: Iterable,
        chunk_size: int = 2,
        backend='loky',
        **kwargs,
    ):
        combinaisons = list(product(winds_direction, winds_speed))
        print('combinaisons', combinaisons)

        for combinaison in self.get_chunks(
            sequence=combinaisons, chunk_size=chunk_size
        ):
            # do something with a chunk here
            print('combinaison', combinaison)
            self.processJobs(combinaisons=combinaison, backend=backend, **kwargs)


if __name__ == '__main__':
    from pymdu.geometric.UrockFiles import UrockFiles

    import geopandas as gpd

    inputs_results = './Ressources/Inputs'
    outputs_results = './Ressources/Outputs'
    outputs_results_umep = './Ressources/Outputs_umep'

    if os.path.exists(outputs_results):
        shutil.rmtree(outputs_results, ignore_errors=True)

    path_trees = os.path.join(inputs_results, 'trees.shp')
    path_buildings = os.path.join(inputs_results, 'buildings.shp')

    try_urock_gen = UrockFiles(
        output_path=inputs_results,
        buildings_gdf=gpd.read_file(path_buildings),
        trees_gdf=gpd.read_file(path_trees),
    )

    urock_buildings_gdf = try_urock_gen.generate_urock_buildings(
        filename_shp='urock_bld.shp'
    )
    urock_trees_gdf = try_urock_gen.generate_urock_trees(filename_shp='urock_trees.shp')

    urock = Urock(
        output_dir=outputs_results,
        working_directory='./',
        input_filepath_buildings_shp=os.path.join(inputs_results, 'urock_bld.shp'),
        input_filepath_trees_shp=os.path.join(inputs_results, 'urock_trees.shp'),
        java_home_path=os.environ.get('JAVA_HOME'),
    )

    directions = range(0, 360, 10)
    vitesses_vent = [1]
    urock.run_parallel(
        winds_speed=vitesses_vent,
        winds_direction=directions,
        chunk_size=2,
        backend='multiprocessing',
    )  # urock.run(output_dir=outputs_results,  #           height_field="hauteur",  #           wind_speed=1,  #           wind_direction=0,  #           wind_height=1.5,  #           horizontal_resolution=2,  #           vertical_resolution=5,  #           save_netcdf=False, save_raster=True, save_vector=False)  #  # urock = Urock(output_dir=outputs_results_umep,  #               working_directory="./",  #               input_filepath_buildings_shp=os.path.join(inputs_results, 'urock_bld.shp'),  #               input_filepath_trees_shp=os.path.join(inputs_results, 'urock_trees.shp'),  #               java_home_path=os.environ.get("JAVA_HOME"))  #  # urock.run_with_umep(output_dir=outputs_results_umep,  #                     height_field="hauteur",  #                     wind_speed=1,  #                     wind_direction=0,  #                     wind_height=1.5,  #                     horizontal_resolution=2,  #                     vertical_resolution=5,  #                     save_netcdf=False, save_raster=True, save_vector=False)
