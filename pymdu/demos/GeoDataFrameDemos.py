# ******************************************************************************
#  This file is part of pymdu.                                                 *
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

from geopandas import GeoDataFrame
from pandas import read_csv
from shapely.wkt import loads


class GeoDataFrameDemos(object):
    """
    classdocs
    """

    @staticmethod
    def postprocess(sio, crs='epsg:2154'):
        _df = read_csv(sio, sep=';')
        _df.rename(columns={'Unnamed: 0': ''}, inplace=True)
        _df.set_index('', inplace=True)
        _df.reset_index(drop=True, inplace=True)
        _df.geometry = _df.geometry.apply(lambda g: loads(g))
        return GeoDataFrame(_df, crs=crs)

    @staticmethod
    def postprocess2(sio, crs='epsg:2154'):
        _df = read_csv(sio, sep=';')
        _df.geometry = _df.geometry.apply(lambda g: loads(g))
        return GeoDataFrame(_df, crs=crs)
