import os

from pymdu.physics.umep.UmepCore import UmepCore


class HeightAspectModelGenerator(UmepCore):
    def __init__(
        self,
        working_directory: os.path,
        input_filepath_dsm: os.path = 'DSM.tif',
        output_filepath_aspect: os.path = 'ASPECT.tif',
        output_filepath_height: os.path = 'HEIGHT.tif',
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
        self.output_filepath_aspect = output_filepath_aspect
        self.output_filepath_height = output_filepath_height

    def run(self, input_limit: int = 3):
        self.run_processing(
            name='umep:Urban Geometry: Wall Height and Aspect',
            options={
                'INPUT': self.input_filepath_dsm,
                'INPUT_LIMIT': input_limit,
                'OUTPUT_HEIGHT': self.output_filepath_height,
                'OUTPUT_ASPECT': self.output_filepath_aspect,
            },
        )
        return None


if __name__ == '__main__':
    pass
