from pymdu.pyqgis.QGisCore import QGisCore


class UmepCore(QGisCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('__init__ UmepCore')

    def run_processing(
        self, name='umep:Spatial Data: Tree Generator', options: dict = None
    ):
        from processing.core.Processing import Processing

        Processing.initialize()
        from processing_umep.processing_umep_provider import ProcessingUMEPProvider
        import processing

        umep_provider = ProcessingUMEPProvider()
        self.qgsApp.processingRegistry().addProvider(umep_provider)
        print('Processing UMEP', name)
        print(options)
        processing.run(name, options)
        self.qgsApp = None
        print('Processing UMEP EXIT', name)

        return None
