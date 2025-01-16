import os

from pymdu.physics.umep.UmepCore import UmepCore


class SVFModelGenerator(UmepCore):
    def __init__(
        self,
        working_directory: os.path,
        input_filepath_dsm: os.path = 'DSM.tif',
        input_filepath_dem: os.path = 'DEM.tif',
        input_filepath_cdsm: os.path = 'CDSM.tif',
        input_filepath_tdsm: os.path = 'TDSM.tif',
        ouptut_filepath_svf: os.path = 'SVF.tif',
    ):
        """
        Initializes an instance of the class.

        Args:

        Returns:
            None
        """
        super().__init__(output_dir=working_directory)
        self.working_directory = working_directory
        self.input_filepath_dsm = input_filepath_dsm
        self.input_filepath_dem = input_filepath_dem
        self.input_filepath_cdsm = input_filepath_cdsm
        self.input_filepath_tdsm = input_filepath_tdsm
        self.ouptut_filepath_svf = ouptut_filepath_svf

    def run(self, trans_veg: int = 3, aniso: bool = True, input_theight: float = 25.0):
        self.run_processing(
            name='umep:Urban Geometry: Sky View Factor',
            options={
                'INPUT_DSM': self.input_filepath_dsm,
                'INPUT_CDSM': self.input_filepath_cdsm,
                'TRANS_VEG': trans_veg,
                'INPUT_TDSM': self.input_filepath_tdsm,
                'INPUT_THEIGHT': input_theight,
                'ANISO': aniso,
                'OUTPUT_DIR': self.working_directory,
                'OUTPUT_FILE': self.ouptut_filepath_svf,
            },
        )
        return None


if __name__ == '__main__':
    pass
