from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsArrowSymbolLayer,
    QgsCategorizedSymbolRenderer,
    QgsFillSymbol,
    QgsHeatmapRenderer,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsLinePatternFillSymbolLayer,
    QgsPointPatternFillSymbolLayer,
    QgsProperty,
    QgsRendererCategory,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsStyle,
    QgsSvgMarkerSymbolLayer,
    QgsSymbol,
    QgsSymbolLayer,
    QgsTextFormat,
    QgsUnitTypes,
    QgsVectorLayerSimpleLabeling,
)
from qgis.utils import iface


class QGisStyle:
    def __init__(self, layer):
        self.layer = layer

    def __setSymbol(self, symbol):
        if self.layer is not None:
            self.layer.renderer().setSymbol(symbol)
            # iface.layerTreeView().refreshLayerSymbology(self.layer.id())
            self.layer.triggerRepaint()

    def __altSetSymbol(self, symbol):
        if self.layer is not None:
            self.layer.renderer().symbol().changeSymbolLayer(0, symbol)
            iface.layerTreeView().refreshLayerSymbology(self.layer.id())
            self.layer.triggerRepaint()

    def __otherAltSetSymbol(self, renderer):
        if self.layer is not None:
            self.layer.setRenderer(renderer)
            iface.layerTreeView().refreshLayerSymbology(self.layer.id())
            self.layer.triggerRepaint()

    def setAlternateFillSymbol(
        self,
        color='black',
        patternDistance=1.0,
        lineAngle=45,
        penstyle='solid',
        width='0.35',
    ):
        lineSymbol = QgsLineSymbol.createSimple(
            {'penstyle': penstyle, 'width': width, 'color': color}
        )
        filled_pattern = QgsLinePatternFillSymbolLayer()
        filled_pattern.setLineAngle(lineAngle)
        filled_pattern.setDistance(patternDistance)
        filled_pattern.setSubSymbol(lineSymbol)
        self.__altSetSymbol(filled_pattern)

    def setAlternateFillSymbol2(
        self, color='red', color_border='black', name='diamond', size='3.0'
    ):
        markerSymbol = QgsMarkerSymbol.createSimple(
            {'color': color, 'color_border': color_border, 'name': name, 'size': size}
        )
        filled_pattern = QgsPointPatternFillSymbolLayer()
        filled_pattern.setDistanceX(4.0)
        filled_pattern.setDistanceY(4.0)
        filled_pattern.setSubSymbol(markerSymbol)
        self.__altSetSymbol(filled_pattern)

    def setAlternateFillSymbol3(self, color='black', style='b_diagonal'):
        # style = ["cross", "b_diagonal", "diagonal_x", "f_diagonal",
        # "horizontal", "solid", "vertical"]
        symbol_layer = QgsSimpleFillSymbolLayer.create({'color': color, 'style': style})
        self.__altSetSymbol(symbol_layer)

    def setArrowSymbol(
        self,
        color='black',
        headType=QgsArrowSymbolLayer.HeadSingle,
        width=0.60,
        headLength=2.05,
        headThickness=1.55,
        arrowType=QgsArrowSymbolLayer.ArrowPlain,
    ):
        # headType = {QgsArrowSymbolLayer.HeadDouble, QgsArrowSymbolLayer.HeadReversed, QgsArrowSymbolLayer.HeadSingle}
        # arrowType = {QgsArrowSymbolLayer.ArrowPlain, QgsArrowSymbolLayer.ArrowLeftHalf, QgsArrowSymbolLayer.ArrowRightHalf}

        arrowSymbol = QgsArrowSymbolLayer()
        arrowSymbol.setColor(QColor(color))

        arrowSymbol.setArrowStartWidth(width)
        arrowSymbol.setArrowWidth(width)

        arrowSymbol.setArrowType(arrowType)

        arrowSymbol.setHeadType(headType)
        arrowSymbol.setHeadLength(headLength)
        arrowSymbol.setHeadThickness(headThickness)

        arrowSymbol.setIsCurved(False)
        arrowSymbol.setIsRepeated(True)

        self.__altSetSymbol(arrowSymbol)

    def setCategorizedSymbol(self, fieldName, listOfValueColorLabels):
        """
        listOfValueColorLabels = (
            ("cat", "red"),
            ("dog", "blue", "Big dog"),
            ("sheep", "green"),
            ("", "yellow", "Unknown")
        )
        """
        # CREATE A CATEGORY FOR EACH ITEM IN 'fieldLookupTable'
        categories = []
        for item in listOfValueColorLabels:
            value, color, label = (
                item if 3 == len(item) else [item[0], item[1], item[0]]
            )
            symbol = QgsSymbol.defaultSymbol(self.layer.geometryType())
            symbol.setColor(QColor(color))
            categories.append(QgsRendererCategory(value, symbol, label))

        # CREATE THE RENDERER AND ASSIGN IT TO THE GIVEN LAYER
        renderer = QgsCategorizedSymbolRenderer(fieldName, categories)
        self.__otherAltSetSymbol(renderer)

    def setFillSymbol(
        self,
        color='lightgrey',
        outline_color='darkgrey',
        style_border='solid',
        width_border='0.75',
    ):
        fillSymbol = QgsFillSymbol.createSimple(
            {
                'color': color,
                'outline_color': outline_color,
                'width_border': width_border,
                'style_border': style_border,
            }
        )
        self.__setSymbol(fillSymbol)

    def setHeatMap(
        self,
        expr,
        radius=20,
        rampName='Blues',
        maxValue=0,
        color1=QColor(255, 255, 255, 0),
        unit=QgsUnitTypes.RenderUnit.MetersInMapUnits,
    ):
        # To get a list of available color ramp names, use:
        # QgsStyle().defaultStyle().colorRampNames()
        # ["Blues", "BrBG", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Inferno", "Magma",
        # "OrRd", "Oranges", "PRGn", "PiYG", "Plasma", "PuBu", "PuBuGn", "PuOr", "PuRd",
        # "Purples", "RdBu", "RdGy", "RdPu", "RdYlBu", "RdYlGn", "Reds", "Spectral",
        # "Viridis", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd"]
        ramp = QgsStyle().defaultStyle().colorRamp(rampName)
        ramp.setColor1(color1)

        heatmap = QgsHeatmapRenderer()
        heatmap.setColorRamp(ramp)
        heatmap.setMaximumValue(
            maxValue
        )  # Set to 0 for automatic calculation of max value
        heatmap.setRadius(radius)
        heatmap.setRadiusUnit(unit)
        heatmap.setRenderQuality(1)  # A value of 1 indicates maximum quality
        heatmap.setWeightExpression(expr)  # expr: fieldname

        self.__otherAltSetSymbol(heatmap)

    def setQMLFile(self, path_qml_file='./qml_file.qml'):
        print('Loading layer...', path_qml_file)
        print(dir(self.layer))
        self.layer.loadNamedStyle(path_qml_file)

        if self.layer.isValid():
            print('Layer loaded!')
        self.layer.triggerRepaint()

    def setLabelSymbol(
        self,
        fieldName,
        overPoint=True,
        size='14',
        color='black',
        positionX=None,
        positionY=None,
        offsetX=None,
        offsetY=None,
    ):
        label = QgsPalLayerSettings()
        label.fieldName = fieldName

        textFormat = QgsTextFormat()
        # bgColor = QgsTextBackgroundSettings()
        # bgColor.setFillColor(QColor("white"))
        # bgColor.setEnabled(True)
        # textFormat.setBackground(bgColor )
        textFormat.setColor(QColor(color))
        textFormat.setSize(int(size))
        label.setFormat(textFormat)
        label.isExpression = True

        self.layer.setLabeling(QgsVectorLayerSimpleLabeling(label))
        self.layer.setLabelsEnabled(True)
        iface.self.layerTreeView().refreshLayerSymbology(self.layer.id())
        self.layer.triggerRepaint()

    def setLineSymbol(self, color='black', penstyle='solid', width='0.55'):
        # dash, dash dot, dash dot dot, dot, solid
        lineSymbol = QgsLineSymbol.createSimple(
            {'color': color, 'penstyle': penstyle, 'width': width}
        )
        self.__setSymbol(lineSymbol)

    def setMarkerSymbol(self, color='black', size='3.6', name='circle'):
        # circle, square, rectangle, diamond, pentagon, triangle,
        # equilateral_triangle, star, regular_star, arrow
        nodesSymbol = QgsMarkerSymbol.createSimple(
            {'color': color, 'name': name, 'size': size, 'width_border': '0'}
        )
        self.__setSymbol(nodesSymbol)

    def setSimpleOutlineFillSymbol(self, color='red', width='1.1', penstyle='dot'):
        symbol_layer = QgsSimpleLineSymbolLayer.create(
            {'color': color, 'width': width, 'penstyle': penstyle}
        )
        self.__altSetSymbol(symbol_layer)

    @staticmethod
    def setSvgSymbol(
        markerLayer, fieldName, svgSymbolDirname, size='6', rotationFieldName=None
    ):
        if markerLayer is not None:
            fni = markerLayer.dataProvider().fieldNameIndex(fieldName)
            uniqValues = markerLayer.dataProvider().uniqueValues(fni)

            categories = []
            for value in uniqValues:
                mySymbol = QgsSymbol.defaultSymbol(markerLayer.geometryType())

                svgStyle = {'name': f'{svgSymbolDirname}/{value}.svg', 'size': size}
                svgSymbol = QgsSvgMarkerSymbolLayer.create(svgStyle)
                if rotationFieldName is not None:
                    svgSymbol.dataDefinedProperties().setProperty(
                        QgsSymbolLayer.PropertyAngle,
                        QgsProperty.fromField(rotationFieldName),
                    )

                mySymbol.changeSymbolLayer(0, svgSymbol)

                category = QgsRendererCategory(value, mySymbol, str(value))
                categories.append(category)

            renderer = QgsCategorizedSymbolRenderer(fieldName, categories)
            markerLayer.setRenderer(renderer)
            iface.self.layerTreeView().refreshLayerSymbology(markerLayer.id())
            markerLayer.triggerRepaint()
