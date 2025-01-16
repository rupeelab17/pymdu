import os

from pymdu.physics.umep.UmepCore import UmepCore


class SurfaceModelGenerator(UmepCore):
    def __init__(
        self,
        working_directory: os.path,
        input_filepath_dsm: os.path = 'DSM.tif',
        input_filepath_dem: os.path = 'DEM.tif',
        input_filepath_tree_shp: os.path = 'trees.shp',
        output_filepath_cdsm: os.path = 'CDSM.tif',
        output_filepath_tdsm: os.path = 'TDSM.tif',
    ):
        """
        Initializes an instance of the class.

        Args:
            input_solweig_path (os.path): The path to the input Solweig file.

        Returns:
            None
        """
        super().__init__(output_dir=working_directory)
        self.input_filepath_dsm = input_filepath_dsm
        self.input_filepath_dem = input_filepath_dem
        self.input_filepath_tree_shp = input_filepath_tree_shp
        self.output_filepath_cdsm = output_filepath_cdsm
        self.output_filepath_tdsm = output_filepath_tdsm

    def run(
        self,
        tree_type: str = 'type',
        diameter: str = 'diameter',
        total_height: str = 'height',
        trunk_height: str = 'trunk zone',
    ):
        self.run_processing(
            name='umep:Spatial Data: Tree Generator',
            options={
                'INPUT_POINTLAYER': self.input_filepath_tree_shp,
                'TREE_TYPE': tree_type,
                'TOT_HEIGHT': total_height,
                'TRUNK_HEIGHT': trunk_height,
                'DIA': diameter,
                'INPUT_BUILD': None,
                'INPUT_DSM': self.input_filepath_dsm,
                'INPUT_DEM': self.input_filepath_dem,
                'INPUT_CDSM': None,
                'INPUT_TDSM': None,
                'CDSM_GRID_OUT': self.output_filepath_cdsm,
                'TDSM_GRID_OUT': self.output_filepath_tdsm,
            },
        )
        return None


if __name__ == '__main__':
    surface_model = SurfaceModelGenerator(
        working_directory='./',
        input_filepath_dsm='./Ressources/Inputs/DSM.tif',
        input_filepath_dem='./Ressources/Inputs/DEM.tif',
        input_filepath_tree_shp='./Ressources/Inputs/trees.shp',
        output_filepath_cdsm='./Ressources/Inputs/CDSM_clip.tif',
        output_filepath_tdsm='./Ressources/Inputs/TDSM_clip.tif',
    )
    surface_model.run()
