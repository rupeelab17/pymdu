import glob
import os

from joblib import delayed, Parallel
from pymdu.physics.umep.UmepCore import UmepCore


class Solweig(UmepCore):
    def __init__(
        self,
        meteo_path: os.path,
        output_dir: os.path,
        working_directory: os.path,
        input_filepath_landcover: os.path = 'landcover.tif',
        input_filepath_dsm: os.path = 'DSM.tif',
        input_filepath_dem: os.path = 'DEM.tif',
        input_filepath_cdsm: os.path = 'CDSM.tif',
        input_filepath_tdsm: os.path = 'TDSM.tif',
        input_filepath_height: os.path = 'HEIGHT.tif',
        input_filepath_aspect: os.path = 'ASPECT.tif',
        input_filepath_shadowmats_npz: os.path = 'shadowmats.npz',
        input_filepath_svf_zip: os.path = 'svfs.zip',
    ):
        """
        Initializes a new instance of the class.

        Args:
            input_solweig_path (os.path): The path to the input solweig file.
            meteo_path (os.path): The path to the meteo file.
            output_dir (os.path): The path to the output directory.
        """
        super().__init__(output_dir=working_directory)
        self.meteo_path = meteo_path
        self.output_dir = output_dir
        self.working_directory = working_directory
        self.input_filepath_dsm = input_filepath_dsm
        self.input_filepath_dem = input_filepath_dem
        self.input_filepath_height = input_filepath_height
        self.input_filepath_aspect = input_filepath_aspect
        self.input_filepath_cdsm = input_filepath_cdsm
        self.input_filepath_tdsm = input_filepath_tdsm
        self.input_filepath_svf_zip = input_filepath_svf_zip
        self.input_filepath_landcover = input_filepath_landcover
        self.input_filepath_shadowmats_npz = input_filepath_shadowmats_npz

    def run(
        self,
        meteo_path=None,
        output_dir=None,
        trans_veg=3,
        input_theight=25,
        use_lc_build=False,
        save_build=False,
        albedo_wall=0.2,
        albedo_ground=0.15,
        emis_wall=0.9,
        emis_ground=0.95,
        abs_s=0.7,
        abs_l=0.95,
        posture=0.5,
        cyl=True,
        only_global=False,
    ):
        if meteo_path is not None:
            self.meteo_path = meteo_path
        if output_dir is not None:
            self.output_dir = output_dir

        self.run_processing(
            name='umep:Outdoor Thermal Comfort: SOLWEIG',
            options={
                'INPUT_DSM': self.input_filepath_dsm,
                'INPUT_SVF': self.input_filepath_svf_zip,
                'INPUT_HEIGHT': self.input_filepath_height,
                'INPUT_ASPECT': self.input_filepath_aspect,
                'INPUT_CDSM': self.input_filepath_cdsm,
                'TRANS_VEG': trans_veg,
                'INPUT_TDSM': self.input_filepath_tdsm,
                'INPUT_THEIGHT': input_theight,
                'INPUT_LC': self.input_filepath_landcover,
                'USE_LC_BUILD': use_lc_build,
                'INPUT_DEM': self.input_filepath_dsm,
                'SAVE_BUILD': save_build,
                'INPUT_ANISO': self.input_filepath_shadowmats_npz,
                'ALBEDO_WALLS': albedo_wall,
                'ALBEDO_GROUND': albedo_ground,
                'EMIS_WALLS': emis_wall,
                'EMIS_GROUND': emis_ground,
                'ABS_S': abs_s,
                'ABS_L': abs_l,
                'POSTURE': posture,
                'CYL': cyl,
                'INPUTMET': self.meteo_path,
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
                'OUTPUT_DIR': self.output_dir,
                'ONLY_GLOBAL': only_global,
            },
        )

        # app.exitQgis()
        return None

    def processJobs(self, output_dir, n_cpu, liste_meteo_path):
        print(n_cpu)
        # with parallel_backend("loky", inner_max_num_threads=2):
        #     results = Parallel(n_jobs=number_of_cpu)(delayed(urock)(combinaison) for combinaison in combinaisons)
        with Parallel(verbose=100, n_jobs=n_cpu) as parallel:
            delayed_funcs = [
                delayed(self.run)(meteo_path, output_dir)
                for meteo_path in liste_meteo_path
            ]
            parallel(delayed_funcs)

    def run_parallel(self):
        liste_meteo_path = glob.glob(os.path.join(self.meteo_path, '*.txt'))
        self.processJobs(
            output_dir=self.output_dir, n_cpu=6, liste_meteo_path=liste_meteo_path
        )


if __name__ == '__main__':
    d = Solweig(
        meteo_path='./test/meteo/FRA_CE_Cap.Pertusato.077700_TMYx.2004-2018.txt',
        output_dir='./test',
        working_directory='./test',
        input_filepath_landcover='./landcover.tif',
        input_filepath_dsm='./DSM.tif',
        input_filepath_dem='./DEM.tif',
        input_filepath_cdsm='./CDSM.tif',
        input_filepath_tdsm='./TDSM.tif',
        input_filepath_height='./HEIGHT.tif',
        input_filepath_aspect='./ASPECT.tif',
        input_filepath_shadowmats_npz='./shadowmats.npz',
        input_filepath_svf_zip='./svfs.zip',
    )
    d.run()
