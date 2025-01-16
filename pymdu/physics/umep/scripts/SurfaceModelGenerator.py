import json
import os
import sys
import pandas as pd
import argparse
from qgis.core import QgsApplication


def create_qgis_path():
    paths = sys.path
    paths.append(
        'C:/Users/simon/AppData/Roaming/QGIS/QGIS3\profiles\default/python/plugins'
    )

    df = pd.DataFrame({'paths': paths})
    df.to_csv('./qgis_sys_paths.csv', index=False)

    env = dict(os.environ)
    rem = ['SECURITYSESSIONID', 'LaunchInstanceID', 'TMPDIR']
    _ = [env.pop(r, None) for r in rem]
    with open('./qgis_env.json', 'w') as f:
        json.dump(env, f, ensure_ascii=False, indent=4)

    # set up system paths
    qspath = './qgis_sys_paths.csv'
    # provide the path where you saved this file.
    paths = pd.read_csv(qspath).paths.tolist()
    sys.path += paths
    # set up environment variables
    qepath = './qgis_env.json'
    qgis_json_path = json.loads(open(qepath, 'r').read())
    for k, v in qgis_json_path.items():
        os.environ[k] = v

    # os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    # os.environ['QGIS_DISABLE_MESSAGE_HOOKS'] = "1"
    os.environ['QGIS_NO_OVERRIDE_IMPORT'] = '1'
    return qgis_json_path


def run(input_solweig_path: os.path, qgis_json_path: dict):
    # initializing processing module
    QgsApplication.setPrefixPath(qgis_json_path['HOMEPATH'], True)
    qgs = QgsApplication([b''], False)
    qgs.initQgis()  # use qgs.exitQgis() to exit the processing module at the end of the script.
    from processing.core.Processing import Processing

    Processing.initialize()
    from processing_umep.processing_umep_provider import ProcessingUMEPProvider
    import processing

    umep_provider = ProcessingUMEPProvider()
    QgsApplication.processingRegistry().addProvider(umep_provider)
    processing.run(
        'umep:Spatial Data: Tree Generator',
        {
            'INPUT_POINTLAYER': os.path.join(input_solweig_path, 'trees.shp'),
            'TREE_TYPE': 'id',
            'TOT_HEIGHT': 'height',
            'TRUNK_HEIGHT': 'trunk zone',
            'DIA': 'diameter',
            'INPUT_BUILD': None,
            'INPUT_DSM': os.path.join(input_solweig_path, 'DSM.tif'),
            'INPUT_DEM': os.path.join(input_solweig_path, 'DEM.tif'),
            'INPUT_CDSM': None,
            'INPUT_TDSM': None,
            'CDSM_GRID_OUT': os.path.join(input_solweig_path, 'CDSM.tif'),
            'TDSM_GRID_OUT': os.path.join(input_solweig_path, 'TDSM.tif'),
        },
    )
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_solweig_path',
        help="donner le chemin d'entrée des fichiers inputs de solweig",
        default=r'C:/Users/simon/python-scripts/Notebooks/DATA/solweig/',
        required=False,
    )
    args = parser.parse_args()
    input_solweig_path = args.input_solweig_path
    qgis_json_path = create_qgis_path()
    qgis_json_path
    run(input_solweig_path=input_solweig_path, qgis_json_path=qgis_json_path)
