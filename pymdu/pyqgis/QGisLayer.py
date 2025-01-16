import glob
import os

from geopandas.geodataframe import GeoDataFrame
from qgis.PyQt.Qt import QVariant
from qgis.core import QgsCoordinateReferenceSystem, QgsFeature, QgsField, QgsGeometry
from qgis.core import QgsRasterLayer
from qgis.core import QgsVectorLayer


class QGisLayer:
    """
    classdocs
    """

    def __init__(self, layer_input, layername):
        """
        Constructor
        """
        self.layername = layername

        if isinstance(layer_input, GeoDataFrame):
            self.gdf = layer_input
            from pandas.api.types import is_datetime64_any_dtype as is_datetime

            self.gdf = self.gdf[
                [
                    column
                    for column in self.gdf.columns
                    if not is_datetime(self.gdf[column])
                ]
            ]
            self.qgsVectorLayer = self.__fromGeoDataFrameToQgsVectorLayer(
                self.gdf, self.layername
            )

        elif isinstance(layer_input, str):
            if layer_input.endswith('.tif'):
                self.qgsVectorLayer = self.__fromRasterToQgsRasterLayer(
                    layer_input, self.layername
                )
            else:
                self.qgsVectorLayer = QgsVectorLayer(layer_input, self.layername, 'ogr')

        else:
            raise Exception('NOT GeoDataFrame OR DatasetReader')

        if not isinstance(layername, str):
            raise Exception(layername, 'str')

    def create(self):
        return self.qgsVectorLayer

    @staticmethod
    def __fromGeoDataFrameToQgsVectorLayer(gdf, layername):
        geomTypes = gdf.geom_type.unique()
        if 1 == len(geomTypes):
            geomType = geomTypes[0]
        elif 2 == len(geomTypes):
            geomType = (
                geomTypes[0]
                if (str(geomTypes[0]).startswith('Multi'))
                else geomTypes[1]
            )

        qgsVectorLayer = QgsVectorLayer(geomType, layername, 'memory')
        provider = qgsVectorLayer.dataProvider()
        qgsVectorLayer.startEditing()
        fieldnames, qgsFields = [], []
        for _fieldname, _fieldtype in gdf.dtypes.items():
            if not ('geometry' == str(_fieldtype)):
                _fieldtype = QVariant.String
                if str(_fieldtype).startswith('int'):
                    _fieldtype = QVariant.Int
                elif str(_fieldtype).startswith('float'):
                    _fieldtype = QVariant.Double
                elif str(_fieldtype).startswith('datetime'):
                    _fieldtype = QVariant.DateTime
                fieldnames.append(_fieldname)
                qgsFields.append(QgsField(_fieldname, QVariant.String))
        provider.addAttributes(qgsFields)

        qgsFeatures = []
        for _, row in gdf.iterrows():
            qgsFeature = QgsFeature()
            qgsFeature.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
            qgsFeature.setAttributes([row[_f] for _f in fieldnames])
            qgsFeatures.append(qgsFeature)
        provider.addFeatures(qgsFeatures)

        # qgsVectorLayer.setCrs(QgsCoordinateReferenceSystem(gdf.crs.to_epsg()))
        qgsVectorLayer.setCrs(
            QgsCoordinateReferenceSystem.fromEpsgId(gdf.crs.to_epsg())
        )
        qgsVectorLayer.commitChanges()
        qgsVectorLayer.updateExtents()
        return qgsVectorLayer

    @staticmethod
    def __fromRasterToQgsRasterLayer(raster_path, layername):
        # Load the raster layer
        qgsRasterLayer = QgsRasterLayer(raster_path, layername, 'gdal')

        # Check if the raster layer was loaded successfully
        if not qgsRasterLayer.isValid():
            print('Error: Unable to load raster layer')

        return qgsRasterLayer


if __name__ == '__main__':
    from pymdu.geometric import Building
    from pymdu.pyqgis.QGisLayer import QGisLayer
    from pymdu.pyqgis.QGisProject import QGisProject
    from pymdu.pyqgis.QGisStyle import QGisStyle

    buildings = Building()
    buildings.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    buildings_gdf = buildings.run().to_gdf()
    buildings_gdf = buildings_gdf.to_crs(crs=2154)

    qgis_project = QGisProject()
    # qgis_layer = QGisLayer(layer_input=buildings_gdf, layername='buildings')
    # buildings_layer = qgis_layer.create()
    # qgis_style = QGisStyle(layer=buildings_layer)
    # qgis_style.setFillSymbol(color="lightgrey", outline_color="darkgrey", style_border="solid", width_border="0.75")
    # qgis_project.setExtent(buildings_layer)
    # qgis_project.addLayer(layer=buildings_layer)

    qgis_layer = QGisLayer(
        '/Users/Boris/Documents/TIPEE/pymdu/pymdu/physics/umep/Ressources/Inputs/buildings.shp',
        layername='buildings',
    )
    buildings_layer = qgis_layer.create()
    qgis_project.addLayer(layer=buildings_layer)
    qgis_style = QGisStyle(layer=buildings_layer)
    qgis_style.setFillSymbol(
        color='lightgrey',
        outline_color='darkgrey',
        style_border='solid',
        width_border='0.75',
    )

    src_tifs = glob.glob(
        os.path.join(
            '/Users/Boris/Documents/TIPEE/pymdu/pymdu/physics/umep/Ressources/Inputs/',
            '*.tif',
        )
    )

    for tif in src_tifs:
        qgis_layer = QGisLayer(layer_input=tif, layername=f'{tif}')
        tif_layer = qgis_layer.create()
        qgis_project.addLayer(layer=tif_layer)
        qgis_style = QGisStyle(layer=tif_layer)
        qgis_style.setQMLFile(
            path_qml_file='/Users/Boris/Documents/TIPEE/pymdu/pymdu/pyqgis/raster.qml'
        )
        # qgis_project.addLayer(layer=buildings_layer)
        qgis_project.setExtent(tif_layer)
    qgis_project.save()
