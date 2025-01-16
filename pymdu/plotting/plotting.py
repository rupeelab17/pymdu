# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  Copyright                                                                   *
#                                                                              *
#  pymdu is free software: you can redistribute it and/or modify               *
#  it under the terms of the GNU General Public License as published by        *
#  the Free Software Foundation, either version 3 of the License, or           *
#  (at your option) any later version.                                         *
#                                                                              *
#  pymdu is distributed in the hope that it will be useful,                    *
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              *
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
#  GNU General Public License for more details.                                *
#                                                                              *
#  You should have received a copy of the GNU General Public License           *
#  along with pymdu.  If not, see <https://www.gnu.org/licenses/>.             *
# ******************************************************************************

import pandas as pd
import plotly.express as px
import pydeck as pdk

__all__ = [
    'pydeck_layer',
    'choropleth_map',
]


def pydeck_layer(layer_type, gdf, features, **kwargs):
    #     '''
    #     Prepare a DataFrame for Polygon plotting in PyDeck
    #
    #     Parameters
    #     ----------
    #
    #     gdf : GeoDataFrame with the geometries to plot
    #
    #     features : list
    #                   List of features to add to the polygon df
    #     Returns
    #     -------
    #
    #     polygon_df : DataFrame
    #                     df with the coordinates in a list as a column and the selected features
    #
    #     '''

    # geojson = gdf.__geo_interface__

    df = pd.DataFrame(gdf)

    if layer_type == 'H3HexagonLayer':
        layer = pdk.Layer(
            'H3HexagonLayer',
            df=df,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            get_hexagon='hex',
            get_fill_color='[255 - count, 255, count]',
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            **kwargs,
        )
    else:
        layer = pdk.Layer(
            'PolygonLayer',
            df=df,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            get_hexagon='hex',
            get_fill_color='[255 - count, 255, count]',
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            **kwargs,
        )

    lon, lat = gdf.geometry.unary_union.centroid.xy

    # Set the viewport location
    view_state = pdk.ViewState(
        latitude=lat[0], longitude=lon[0], zoom=14, bearing=0, pitch=30
    )

    # Render
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={'text': 'Count: {count}'},
    )
    r.to_html('h3_hexagon_layer.html')


def choropleth_map(gdf, color_column, df_filter=None, **kwargs):
    """
    Produce a Choroplethmap using plotly by passing a GeoDataFrame.
    Parameters
    https://plotly.com/python/mapbox-county-choropleth/
    Choropleth map using plotly.express and carto base map (no token needed)
    ----------
    gdf: GeoDataFrame
             Input data containing a geometry column and features
    color_column: str
                      Column from gdf to use as color
    df_filter: pd.Series, default to None
                   Pandas Series containing true/false values that satisfy a
                   condition (e.g. (df['population'] > 100))
    **kargs: Any parameter of plotly.px.choroplethmapbox.
    Examples
    --------
    """

    if df_filter is not None:
        gdff = gdf[df_filter].copy()
    else:
        gdff = gdf.copy()

    gdff = gdff.reset_index()[['index', color_column, 'geometry']].dropna()
    lon, lat = gdff.geometry.unary_union.centroid.xy

    fig = px.choropleth_mapbox(
        gdff,
        geojson=gdff[['geometry']].__geo_interface__,
        color=color_column,
        locations='index',
        center={'lat': lat[0], 'lon': lon[0]},
        zoom=10,
        mapbox_style='carto-positron',
        **kwargs,
    )
    return fig
