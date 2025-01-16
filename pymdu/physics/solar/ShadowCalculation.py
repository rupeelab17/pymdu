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
from datetime import datetime

import geopandas as gpd
from t4gpd.commons.DatetimeLib import DatetimeLib
from t4gpd.sun.STHardShadow import STHardShadow

from pymdu.GeoCore import GeoCore
from pymdu.commons.BasicFunctions import BasicFunctions
from pymdu.demos.Technoforum import Technoforum


class ShadowCalculation(GeoCore):
    """
    ===
    Classe qui permet
    - de calculer les ombrages des bâtiments
    ===
    """

    def __init__(
        self,
        buildings_gdf: gpd.GeoDataFrame = Technoforum().buildings(),
        init: str = '2022-06-21 06:00:00',
        end: str = '2022-06-21 19:00:00',
        time_delta_hours: int = 3,
        year: str = '2022',
        annual_calculation: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shadows = None
        self.buildings_gdf = buildings_gdf
        if annual_calculation:
            self.start = f'{year}-01-01 00:00:00'
            self.liste_date = [
                datetime(int(year), i + +1, 1, ii, 0, 0, 0)
                for i in range(12)
                for ii in range(24)
            ]

        else:
            self.liste_date = BasicFunctions.generate_datetime_list(
                init, end, time_delta_hours
            )

    def run(self):
        # self.buildings_gdf = convert_crs(self.buildings_gdf, crs=self._epsg)
        self.buildings_gdf = self.buildings_gdf.to_crs(self._epsg)
        datetimes = DatetimeLib.generate(self.liste_date)
        self.shadows = STHardShadow(
            occludersGdf=self.buildings_gdf,
            datetimes=datetimes,
            occludersElevationFieldname='hauteur',
            altitudeOfShadowPlane=0,
            aggregate=True,
            tz='Europe/Paris',
            model='pysolar',
        ).run()
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.shadows
