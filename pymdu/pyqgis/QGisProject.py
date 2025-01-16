import os

from pymdu.pyqgis.QGisCore import QGisCore
from qgis.core import QgsProject


class QGisProject(QGisCore):
    def __init__(self, output_dir='./'):
        super().__init__(output_dir)
        # Create a QgsProject
        self.qgsVectorLayer = None
        self.project = QgsProject.instance()
        self.project.clear()

    def addLayer(self, layer):
        self.project.addMapLayer(layer)
        return self

    def save(self, project_name='project.qgs'):
        self.project.write(os.path.join(self.output_dir, project_name))

    @staticmethod
    def setExtent(roi):
        # canvas = QgsMapCanvas()
        # box = roi.boundingBoxOfSelected()
        # roi.setSelectedFeatures([])
        # canvas.setExtent(box)
        # canvas.refresh()
        # canvas = iface.mapCanvas()
        # extent = roi.extent()
        # canvas.setExtent(extent)
        # canvas.refresh()
        # render.setLayerSet(roi.getLayerID())
        # rect = QgsRectangle()
        # rect.scale(1.1)
        # roi.setExtent(rect.fullExtent())
        # if not isinstance(roi, QgsRectangle):
        #     if isinstance(roi, QgsVectorLayer):
        #         roi = roi.extent()
        #     elif isinstance(roi, GeoDataFrame):
        #         roi = QgsRectangle(*roi.total_bounds)
        #     elif GeomLib.isAShapelyGeometry(roi):
        #         roi = QgsRectangle(*roi.bounds)
        #     elif isinstance(roi, ndarray) and (4 == len(roi)):
        #         roi = QgsRectangle(*roi)
        #     else:
        #         raise Exception(roi, 'QgsRectangle, QgsVectorLayer, GeoDataFrame or ndarray')
        #
        # iface.mapCanvas().setExtent(roi)
        # iface.mapCanvas().refresh()
        return roi
