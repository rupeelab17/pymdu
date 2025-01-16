<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology" version="3.34.0-Prizren">
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option value="" name="name" type="QString"/>
      <Option name="properties"/>
      <Option value="collection" name="type" type="QString"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" enabled="false"/>
    </provider>
    <rasterrenderer classificationMin="8.6091537" classificationMax="15.5647125" alphaBand="-1" opacity="1" nodataColor="" type="singlebandpseudocolor" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="15.564712500000001" classificationMode="1" clip="0" colorRampType="INTERPOLATED" minimumValue="8.6091537000000002" labelPrecision="4">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option value="202,0,32,255" name="color1" type="QString"/>
              <Option value="5,113,176,255" name="color2" type="QString"/>
              <Option value="ccw" name="direction" type="QString"/>
              <Option value="0" name="discrete" type="QString"/>
              <Option value="gradient" name="rampType" type="QString"/>
              <Option value="rgb" name="spec" type="QString"/>
              <Option value="0.25;244,165,130,255;rgb;ccw:0.5;247,247,247,255;rgb;ccw:0.75;146,197,222,255;rgb;ccw" name="stops" type="QString"/>
            </Option>
          </colorramp>
          <item value="8.6091537475586" color="#ca0020" label="8,6092" alpha="255"/>
          <item value="10.34804344177245" color="#f4a582" label="10,3480" alpha="255"/>
          <item value="12.0869331359863" color="#f7f7f7" label="12,0869" alpha="255"/>
          <item value="13.825822830200151" color="#92c5de" label="13,8258" alpha="255"/>
          <item value="15.564712524414" color="#0571b0" label="15,5647" alpha="255"/>
          <rampLegendSettings minimumLabel="" orientation="2" direction="0" maximumLabel="" suffix="" useContinuousLegend="1" prefix="">
            <numericFormat id="basic">
              <Option type="Map">
                <Option name="decimal_separator" type="invalid"/>
                <Option value="6" name="decimals" type="int"/>
                <Option value="0" name="rounding_type" type="int"/>
                <Option value="false" name="show_plus" type="bool"/>
                <Option value="true" name="show_thousand_separator" type="bool"/>
                <Option value="false" name="show_trailing_zeros" type="bool"/>
                <Option name="thousand_separator" type="invalid"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" gamma="1" contrast="0"/>
    <huesaturation colorizeOn="0" colorizeGreen="128" saturation="0" invertColors="0" colorizeRed="255" grayscaleMode="0" colorizeStrength="100" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
