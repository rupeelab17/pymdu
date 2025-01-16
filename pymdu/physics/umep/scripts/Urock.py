import json
import os
import sys
from itertools import product

import joblib
import pandas as pd
from joblib import delayed, Parallel

# parser =  argparse.ArgumentParser()
# parser.add_argument('--root_path', help = 'please give a number', required=True)
# parser.add_argument('--data_path', help = 'please give a number', required=True)
# parser.add_argument('--output_dir', help = 'please give a number', required=True)
# args = parser.parse_args()


# Get user home directory
# user_home = os.environ['~']

# Set up system paths
qspath = 'qgis_sys_paths.csv'
paths = pd.read_csv(qspath).paths.tolist()
sys.path += paths

print('1')

# Set up environment variables
qepath = 'qgis_env.json'
js = json.loads(open(qepath, 'r').read())
for k, v in js.items():
    os.environ[k] = v

print('2')


# Import QGIS plugins in the system paths

# third_party_path = os.path.join(r"C:\Users\simon\AppData\Roaming\QGIS\QGIS3\profiles\default/python/plugins/")
# sys.path.append(r'{0}'.format(third_party_path))
#
# third_party_path = os.path.join(r"C:\Users\simon\AppData\Roaming\QGIS\QGIS3\profiles\default")
# sys.path.append(r'{0}'.format(third_party_path))


def run(combinaison, data_path, output_dir):
    # Import UMEP plugin functions in the "factory"
    from qgis.core import QgsApplication

    # path to your qgis installation
    gui_flag = False
    app = QgsApplication([], gui_flag)
    app.initQgis()

    # Prepare processing framework to access all default QGIS processing functions
    from processing.core.Processing import Processing

    Processing.initialize()
    print('OK')

    from processing_umep.processing_umep_provider import ProcessingUMEPProvider
    import processing

    try:
        os.mkdir(os.path.join(output_dir))
    except FileExistsError:
        print(f'{output_dir} already exists')
        pass
    umep_provider = ProcessingUMEPProvider()
    QgsApplication.processingRegistry().addProvider(umep_provider)
    data_path = data_path
    output_dir = output_dir
    # name_run = meteo_file.split('\\')[-1].split('.txt')[0]
    # subdirectory = os.path.join(output_dir, name_run)
    # os.mkdir(subdirectory)

    processing.run(
        'umep:Urban Wind Field: URock',
        {
            'BUILDINGS': os.path.join(data_path, 'batiments.shp'),
            'HEIGHT_FIELD_BUILD': 'hauteur',
            'VEGETATION': os.path.join(data_path, 'arbres.shp'),
            'VEGETATION_CROWN_TOP_HEIGHT': 'MAX_HEIGHT',
            'VEGETATION_CROWN_BASE_HEIGHT': 'MIN_HEIGHT',
            'ATTENUATION_FIELD': '',
            'INPUT_PROFILE_FILE': '',
            'INPUT_PROFILE_TYPE': 0,
            'INPUT_WIND_HEIGHT': 10,
            'INPUT_WIND_SPEED': combinaison[1],
            'INPUT_WIND_DIRECTION': combinaison[0],
            'RASTER_OUTPUT': None,
            'HORIZONTAL_RESOLUTION': 5,
            'VERTICAL_RESOLUTION': 5,
            'WIND_HEIGHT': '1.5',
            'UROCK_OUTPUT': output_dir,
            'OUTPUT_FILENAME': f'output_{combinaison[0]}',
            'SAVE_RASTER': True,
            'SAVE_VECTOR': True,
            'SAVE_NETCDF': True,
            'LOAD_OUTPUT': True,
            'JAVA_PATH': 'C:/Program Files/Java/jdk-11.0.16.1',
        },
    )

    # app.exitQgis()
    return None


def processJobs(combinaisons, data_path, output_dir):
    number_of_cpu = joblib.cpu_count() - 1
    print(number_of_cpu)
    # with parallel_backend("loky", inner_max_num_threads=2):
    #     results = Parallel(n_jobs=number_of_cpu)(delayed(urock)(combinaison) for combinaison in combinaisons)
    with Parallel(
        verbose=10,
        timeout=350,
        n_jobs=number_of_cpu,
        pre_dispatch='1 * n_jobs',
        prefer='processes',
    ) as parallel:
        delayed_funcs = [
            delayed(run)(combinaison, data_path, output_dir)
            for combinaison in combinaisons
        ]
        parallel_pool = parallel(delayed_funcs)
    return parallel_pool


if __name__ == '__main__':
    data_path = os.path.join(r'C:\Users\simon\python-scripts\Notebooks\DATA\urock')
    output_dir = os.path.join(
        r'C:\Users\simon\python-scripts\Notebooks\RESULTS/WindDirection'
    )
    directions = range(0, 360, 10)
    vitesses_vent = [1]
    combinaisons = list(product(directions, vitesses_vent))
    ####################################################################################################################
    #
    ####################################################################################################################
    combinaisons = list(product(directions, vitesses_vent))
    ####################################################################################################################
    #
    ####################################################################################################################
    from typing import Any, Sequence, Iterator

    def get_chunks(sequence: Sequence[Any], chunk_size: int) -> Iterator[Any]:
        for i in range(0, len(sequence), chunk_size):
            yield sequence[i : i + chunk_size]

    for combinaison in get_chunks(sequence=combinaisons, chunk_size=4):
        # do something with a chunk here
        print(combinaison)
        processJobs(combinaison, data_path, output_dir)
