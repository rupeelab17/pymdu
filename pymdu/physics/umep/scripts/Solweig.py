import glob
import json
import os
import sys

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

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


def run(meteo_file, data_path, output_dir):
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

    umep_provider = ProcessingUMEPProvider()
    QgsApplication.processingRegistry().addProvider(umep_provider)
    data_path = data_path
    output_dir = output_dir
    # name_run = meteo_file.split("\\")[-1].split(".txt")[0]
    # subdirectory = os.path.join(output_dir, name_run)
    # os.mkdir(subdirectory)
    subdirectory = output_dir
    print(os.path.join(meteo_file))

    processing.run(
        'umep:Outdoor Thermal Comfort: SOLWEIG',
        {
            'INPUT_DSM': os.path.join(data_path, 'DSM.tif'),
            'INPUT_SVF': os.path.join(data_path, 'svfs.zip'),
            'INPUT_HEIGHT': os.path.join(data_path, 'HEIGHT.tif'),
            'INPUT_ASPECT': os.path.join(data_path, 'ASPECT.tif'),
            'INPUT_CDSM': os.path.join(data_path, 'CDSM.tif'),
            'TRANS_VEG': 3,
            'INPUT_TDSM': None,
            'INPUT_THEIGHT': 25,
            'INPUT_LC': os.path.join(data_path, 'landcover.tif'),
            'USE_LC_BUILD': False,
            'INPUT_DEM': os.path.join(data_path, 'DEM.tif'),
            'SAVE_BUILD': False,
            'INPUT_ANISO': os.path.join(data_path, 'shadowmats.npz'),
            'ALBEDO_WALLS': 0.2,
            'ALBEDO_GROUND': 0.15,
            'EMIS_WALLS': 0.9,
            'EMIS_GROUND': 0.95,
            'ABS_S': 0.7,
            'ABS_L': 0.95,
            'POSTURE': 0,
            'CYL': True,
            'INPUTMET': os.path.join(meteo_file),
            'ONLYGLOBAL': False,
            'UTC': 0,
            'POI_FILE': None,
            'POI_FIELD': '',
            'AGE': 35,
            'ACTIVITY': 80,
            'CLO': 0.9,
            'WEIGHT': 75,
            'HEIGHT': 180,
            'SEX': 0,
            'SENSOR_HEIGHT': 10,
            'OUTPUT_TMRT': True,
            'OUTPUT_KDOWN': False,
            'OUTPUT_KUP': False,
            'OUTPUT_LDOWN': False,
            'OUTPUT_LUP': False,
            'OUTPUT_SH': True,
            'OUTPUT_TREEPLANTER': False,
            'OUTPUT_DIR': subdirectory,
        },
    )

    # app.exitQgis()
    return None


def processJobs(combinaisons, data_path, output_dir, n_cpu):
    print(n_cpu)
    # with parallel_backend("loky", inner_max_num_threads=2):
    #     results = Parallel(n_jobs=number_of_cpu)(delayed(urock)(combinaison) for combinaison in combinaisons)
    with Parallel(verbose=100, n_jobs=n_cpu) as parallel:
        delayed_funcs = [
            delayed(run)(meteo_path, data_path, output_dir) for meteo_path in liste_path
        ]
        parallel_pool = parallel(delayed_funcs)
    return parallel_pool


if __name__ == '__main__':
    ####################################################################################################################
    #                                                 A COMPLETER ICI
    ####################################################################################################################
    meteo_path = r'C:\Users\simon\python-scripts\Notebooks\meteo_parallel'
    data_path = r'C:\Users\simon\python-scripts\Notebooks\DATA\solweig'
    output_dir = r'C:\Users\simon\python-scripts\Notebooks\VARIABLE'

    ####################################################################################################################
    #
    ####################################################################################################################
    liste_path = glob.glob(os.path.join(meteo_path, '*.txt'))
    print(liste_path)
    # processJobs(liste_path)
    from typing import Any, Sequence, Iterator

    def get_chunks(sequence: Sequence[Any], chunk_size: int) -> Iterator[Any]:
        for i in range(0, len(sequence), chunk_size):
            yield sequence[i : i + chunk_size]

    # split_meteo = list(get_chunks(sequence=liste_path, chunk_size=7))
    # print(split_meteo)
    for meteo_paths in tqdm(liste_path):
        processJobs(meteo_paths, data_path, output_dir, n_cpu=6)
