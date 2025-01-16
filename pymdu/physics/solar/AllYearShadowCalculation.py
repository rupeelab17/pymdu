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
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.commons.BasicFunctions import BasicFunctions
from pymdu.demos.Technoforum import Technoforum
from pymdu.physics.solar.ShadowCalculation import ShadowCalculation

warnings.filterwarnings('ignore')


class AllYearShadowCalculation(BasicFunctions):
    """
    ===
    Classe qui permet
    - d'interpoler le % d'ombrage par bloc de quartier sur toute l'année
    ===
    """

    def __init__(
        self,
        blocs: gpd.GeoDataFrame = Technoforum().blocs(),
        buildings: gpd.GeoDataFrame = Technoforum().buildings(),
    ):
        # super().__init__(*args, **kwargs)
        self.buildings = buildings
        self.buildings.to_file(os.path.join(TEMP_PATH, 'buildings.shp'))
        self.blocs = blocs

    def run(self):
        obj_shadow = ShadowCalculation(annual_calculation=True)
        df_shadows = obj_shadow.run()

        # création de la listedes jours
        liste_jours = []

        df_shadows['nbr-jours'] = pd.to_datetime(
            df_shadows['datetime'].values
        ).dayofyear

        ombrage = {}
        ground = self.blocs.overlay(
            self.buildings, how='difference', keep_geom_type=True
        )
        ground = ground.to_crs(2154)
        for date in tqdm(list(dict.fromkeys([x for x in df_shadows['datetime']]))):
            shd = df_shadows[df_shadows['datetime'] == date]
            liste_percentage = []

            for blc, nbr in zip(ground['geometry'], ground['geometry'].index):
                gdf = gpd.GeoDataFrame(index=[0], crs='epsg:2154', geometry=[blc])

                intersection = gpd.clip(shd, gdf)
                percent_shade = intersection.area
                liste_percentage.append(np.sum(percent_shade) / blc.area)

            to_save = gpd.GeoDataFrame(
                index=ground['geometry'].index,
                crs='epsg:2154',
                geometry=ground['geometry'],
            )
            to_save['%_shadow'] = liste_percentage
            ombrage[date] = to_save

        liste_keys = ombrage.keys()

        shadowDict = {}
        for key in liste_keys:
            shadowDict[key.replace('T', ' ').replace('+00:00', '')] = ombrage[key]

        liste_jours = []
        for x in shadowDict.keys():
            shadowDict[x]['nbr-jours'] = pd.to_datetime(x).dayofyear
            liste_jours.append(pd.to_datetime(x).dayofyear)
        mylist = list(dict.fromkeys(liste_jours))
        mylist.append(365)

        premiereListe = []
        for x in shadowDict.keys():
            if '-01-' in x:
                premiereListe.append(x)
            else:
                pass

        mes_dates = []
        for i in range(8760):
            mes_dates.append(pd.to_datetime(self.start) + pd.Timedelta(i, unit='h'))

        df = pd.DataFrame()
        df.index = pd.to_datetime(mes_dates)

        add_bloc = {}
        for x in self.blocs.index:
            add_bloc[x] = []
            for ii in df.index:
                if str(ii) in shadowDict.keys():
                    add_bloc[x].append(shadowDict[str(ii)]['%_shadow'][x])
                else:
                    add_bloc[x].append(1)
            df[x] = add_bloc[x]
        # pour boucler

        df[f'{self.year}-12-31'] = df[f'{self.year}-01-01']
        df['dayOfyear'] = [x.dayofyear for x in df.index]
        df['hour'] = [x.hour for x in df.index]
        #     to_save = gpd.GeoDataFrame(index=ground['geometry'].index, crs='epsg:2154', geometry=ground['geometry'])
        #     to_save['%_shadow'] = liste_percentage
        #     ombrage[date] = to_save
        # liste_jours.append(pd.to_datetime(df_shadows['datetime']).dayofyear)
        # my_list = list(dict.fromkeys(liste_jours))
        # my_list.append(365)

        biglist = []
        for day, hour, index in zip(df.dayOfyear, df.hour, df.index):
            if day in mylist:
                old_df = df.loc[(df.dayOfyear == day) & (df.hour == hour)]
                biglist.append(old_df.values[0])
            else:
                init, end = self.get_interval(day, mylist)
                old_df = df.loc[(df.dayOfyear == day) & (df.hour == hour)]
                df1 = df.loc[(df.dayOfyear == init) & (df.hour == hour)]
                df2 = df.loc[(df.dayOfyear == end) & (df.hour == hour)]
                old_df = (
                    df1
                    + ((df2 - df1.values[0]) * ((day - init) / (end - init))).values[0]
                )
                biglist.append(old_df.values[0])

        data = pd.DataFrame.from_records(biglist)
        data.index = [pd.to_datetime(str(x)) for x in df.index]
        data.columns = df.columns
        data.to_csv(os.path.join(TEMP_PATH, 'shadows.csv'))

        return data


if __name__ == '__main__':
    test = AllYearShadowCalculation()
    data = test.run()
    print(data)
    # print(dict_shadows)
