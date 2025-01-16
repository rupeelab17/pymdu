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

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.physics.thermal.uwg import UWG
from pymdu.physics.thermal.uwg.readDOE import readDOE


class TemperatureCalculation(GeoCore):
    """
    ===
    Classe qui permet
    - de calculer la température d'air dans les RSU du quartier
    ===
    """

    def __init__(self):
        self.toto = 'toto'
        # self.data = pd.read_excel(MAIN_PATH+'/physics/thermal/param_uwg.xlsx', index_col=0, header=None).rename(columns={1: "test"})
        # self.data['test']['schtraffic'] = eval(self.data['test']['schtraffic'])
        # self.data['test']['bld'] = eval(self.data['test']['bld'])
        # self.blocs = pd.read_csv(TEMP_PATH+'/uwg_blocs.csv')
        #
        # liste_index = ['blddensity', 'bldheight', 'grasscover', 'vertohor', 'gid', 'charlength']
        # for k in self.blocs['gid']:
        #     to_fill = []
        #     for index in self.data.index:
        #         if index in liste_index:
        #             to_fill.append(self.blocs[index][k])
        #         else:
        #             try:
        #                 to_fill.append(eval(self.data['test'][index][-1]))
        #             except:
        #                 to_fill.append(self.data['test'][index])
        #     self.data[k] = to_fill
        #
        # self.data['num'] = [x for x in range(len(self.data['test']))]
        # columns = list(set(self.data.keys()) - set(['test', 'num']))
        # self.dataframe = self.data[columns]
        # self.dataframe = self.dataframe.dropna(axis=1)

    def run(self, epw_name: str = 'technoforum.epw'):
        """

        Returns:
            object:
        """
        columns = list(set(self.dataframe.keys()) - set(['test']))
        for col in columns:
            model = UWG.from_param_args(
                float(self.dataframe[col].bldheight),
                float(self.dataframe[col].blddensity),
                float(self.dataframe[col].vertohor),
                float(self.dataframe[col].grasscover),
                float(self.dataframe[col].treecover),
                self.dataframe[col].zone,
                month=self.dataframe[col].month,
                day=self.dataframe[col].day,
                nday=self.dataframe[col].nday,
                dtsim=self.dataframe[col].dtsim,
                dtweather=self.dataframe[col].dtweather,
                bld=self.dataframe[col].bld,
                autosize=self.dataframe[col].autosize,
                h_mix=self.dataframe[col].h_mix,
                sensocc=self.dataframe[col].sensocc,
                latfocc=self.dataframe[col].latfocc,
                radfocc=self.dataframe[col].radfocc,
                radfequip=self.dataframe[col].radfequip,
                radflight=self.dataframe[col].radflight,
                charlength=float(self.dataframe[col].charlength),
                # albroad=dataframe[col].albroad,
                # droad=dataframe[col].droad,
                # kroad=dataframe[col].kroad,
                # croad=dataframe[col].croad,
                # rurvegcover=dataframe[col].rurvegcover,
                # vegstart=dataframe[col].vegstart,
                # vegend=dataframe[col].vegend,
                # albveg=dataframe[col].albveg,
                # latgrss=dataframe[col].latgrss,
                # lattree=dataframe[col].lattree,
                # sensanth=dataframe[col].sensanth,
                # schtraffic=dataframe[col].schtraffic,
                # h_ubl1=dataframe[col].h_ubl1,
                # h_ubl2=dataframe[col].h_ubl2,
                # h_ref=dataframe[col].h_ref,
                # h_temp=dataframe[col].h_temp,
                # h_wind=dataframe[col].h_wind,
                # c_circ=dataframe[col].c_circ,
                # c_exch=dataframe[col].c_exch,
                # maxday=dataframe[col].maxday,
                # maxnight=dataframe[col].maxnight,
                # windmin=dataframe[col].windmin,
                # h_obs=dataframe[col].h_obs,
                epw_path=os.path.join(TEMP_PATH, epw_name),
                new_epw_dir=None,
                new_epw_name=f'bloc_{col}.epw',
                ref_bem_vector=None,
                ref_sch_vector=None,
            )
            # model = UWG.from_param_args(epw_path=epw_path, bldheight=float(data[0].bldheight), blddensity=0.5,
            #                             vertohor=0.8, grasscover=0.1, treecover=0.1, zone=data[0].zone)

            # Uncomment these lines to initialize the UWG model using a .uwg parameter file
            # param_path = "initialize_singapore.uwg"  # available in resources directory.
            # model = UWG.from_param_file(param_path, epw_path=epw_path)
            readDOE(serialize_output=True)
            model.generate()
            model.simulate()

            # Write the simulation result to a file.
            model.write_epw()
            # model.write_additionnal_epw()

    def single_run(
        self,
        geoclimate_pah=r'./qapeosud/osm_atlantech/',
        epw_path=r'C:\Users\simon\python-scripts\pymdu/FRA_AC_La.Rochelle.073150_TMYx.epw',
        log=False,
    ):
        bbox = GeoCore().bbox

        building = gpd.read_file(
            os.path.join(geoclimate_pah, 'building.geojson')
        ).to_crs(2154)
        building_blocks = gpd.read_file(
            os.path.join(geoclimate_pah, 'block_indicators.geojson')
        ).to_crs(2154)
        bloc = gpd.read_file(
            os.path.join(geoclimate_pah, 'rsu_indicators.geojson')
        ).to_crs(2154)
        bloc['area'] = [g.area for g in bloc['geometry']]
        uwg_data = bloc[
            [
                'BUILDING_TOTAL_FRACTION',
                'AVG_HEIGHT_ROOF_AREA_WEIGHTED',
                'FREE_EXTERNAL_FACADE_DENSITY',
                'area',
            ]
        ]

        geom = box(
            GeoCore().bbox[0], GeoCore().bbox[1], GeoCore().bbox[2], GeoCore().bbox[3]
        )

        district = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom]).to_crs(
            2154
        )

        for k in [
            'BUILDING_TOTAL_FRACTION',
            'AVG_HEIGHT_ROOF_AREA_WEIGHTED',
            'FREE_EXTERNAL_FACADE_DENSITY',
        ]:
            uwg_data[k + '_AREA_WEIGHTED'] = uwg_data[k] * uwg_data['area']

        blddensity = (
            uwg_data['BUILDING_TOTAL_FRACTION_AREA_WEIGHTED'].sum()
            / uwg_data['area'].sum()
        )
        bldheight = (
            uwg_data['AVG_HEIGHT_ROOF_AREA_WEIGHTED_AREA_WEIGHTED'].sum()
            / uwg_data['area'].sum()
        )
        vertohor = (
            uwg_data['FREE_EXTERNAL_FACADE_DENSITY_AREA_WEIGHTED'].sum()
            / uwg_data['area'].sum()
        )

        try:
            veg = gpd.read_file(os.path.join(geoclimate_pah, 'vegetation.shp'))
        except:
            veg = gpd.read_file(
                os.path.join(geoclimate_pah, 'vegetation.geojson')
            ).to_crs(2154)
        grasscover = veg.area.sum() / uwg_data['area'].sum()
        charlength = np.sqrt(district.area.values[0])

        liste_index = [
            'blddensity',
            'bldheight',
            'grasscover',
            'vertohor',
            'gid',
            'charlength',
        ]
        liste_values = [blddensity, bldheight, grasscover, vertohor, 0, charlength]
        to_fill = pd.DataFrame()

        to_fill.index = liste_index
        to_fill['values'] = liste_values

        data = pd.read_excel(
            str(self.physics_path.joinpath('thermal/param_uwg.xlsx')),
            index_col=0,
            header=None,
        ).rename(columns={1: 'test'})
        vector_uwg = []
        for index in data.index:
            if index in to_fill.index:
                vector_uwg.append(to_fill[to_fill.index == index].values[0][0])
            else:
                try:
                    vector_uwg.append(eval(data['test'][index][-1]))
                except:
                    vector_uwg.append(data['test'][index])
        data['uwg'] = vector_uwg
        if log:
            args = [
                ('bldheight', float(data['uwg'].bldheight)),
                ('blddensity', float(data['uwg'].blddensity)),
                ('vertohor', float(data['uwg'].vertohor)),
                ('grasscover', float(data['uwg'].grasscover)),
                ('treecover', float(data['uwg'].treecover)),
                ('zone', data['uwg'].zone),
                ('month', data['uwg'].month),
                ('day', data['uwg'].day),
                ('nday', data['uwg'].nday),
                ('dtsim', data['uwg'].dtsim),
                ('dtweather', data['uwg'].dtweather),
                ('bld', eval(data['uwg'].bld)),
                ('autosize', data['uwg'].autosize),
                ('h_mix', data['uwg'].h_mix),
                ('sensocc', data['uwg'].sensocc),
                ('latfocc', data['uwg'].latfocc),
                ('radfocc', data['uwg'].radfocc),
                ('radfequip', data['uwg'].radfequip),
                ('radflight', data['uwg'].radflight),
                ('charlength', float(data['uwg'].charlength)),
                ('albroad', data['uwg'].albroad),
                ('droad', data['uwg'].droad),
                ('kroad', data['uwg'].kroad),
                ('croad', data['uwg'].croad),
                ('rurvegcover', data['uwg'].rurvegcover),
                ('vegstart', data['uwg'].vegstart),
                ('vegend', data['uwg'].vegend),
                ('albveg', data['uwg'].albveg),
                ('latgrss', data['uwg'].latgrss),
                ('lattree', data['uwg'].lattree),
                ('sensanth', data['uwg'].sensanth),
                ('schtraffic', eval(data['uwg'].schtraffic)),
                ('h_ubl1', data['uwg'].h_ubl1),
                ('h_ubl2', data['uwg'].h_ubl2),
                ('h_ref', data['uwg'].h_ref),
                ('h_temp', data['uwg'].h_temp),
                ('h_wind', data['uwg'].h_wind),
                ('c_circ', data['uwg'].c_circ),
                ('c_exch', data['uwg'].c_exch),
                ('maxday', data['uwg'].maxday),
                ('maxnight', data['uwg'].maxnight),
                ('windmin', data['uwg'].windmin),
                ('h_obs', data['uwg'].h_obs),
                ('epw_path', epw_path),
                ('new_epw_dir', None),  # new_epw_dir
                ('new_epw_name', None),  # new_epw_name
                ('ref_bem_vector', None),  # ref_bem_vector
                ('ref_sch_vector', None),  # ref_sch_vector
            ]
            for x in args:
                print(x)
        model = UWG.from_param_args(
            float(data['uwg'].bldheight),
            float(data['uwg'].blddensity),
            float(data['uwg'].vertohor),
            float(data['uwg'].grasscover),
            float(data['uwg'].treecover),
            data['uwg'].zone,
            month=data['uwg'].month,
            day=data['uwg'].day,
            nday=data['uwg'].nday,
            dtsim=data['uwg'].dtsim,
            dtweather=data['uwg'].dtweather,
            bld=eval(data['uwg'].bld),
            autosize=data['uwg'].autosize,
            h_mix=data['uwg'].h_mix,
            sensocc=data['uwg'].sensocc,
            latfocc=data['uwg'].latfocc,
            radfocc=data['uwg'].radfocc,
            radfequip=data['uwg'].radfequip,
            radflight=data['uwg'].radflight,
            charlength=float(data['uwg'].charlength),
            albroad=data['uwg'].albroad,
            droad=data['uwg'].droad,
            kroad=data['uwg'].kroad,
            croad=data['uwg'].croad,
            rurvegcover=data['uwg'].rurvegcover,
            vegstart=data['uwg'].vegstart,
            vegend=data['uwg'].vegend,
            albveg=data['uwg'].albveg,
            latgrss=data['uwg'].latgrss,
            lattree=data['uwg'].lattree,
            sensanth=data['uwg'].sensanth,
            schtraffic=eval(data['uwg'].schtraffic),
            h_ubl1=data['uwg'].h_ubl1,
            h_ubl2=data['uwg'].h_ubl2,
            h_ref=data['uwg'].h_ref,
            h_temp=data['uwg'].h_temp,
            h_wind=data['uwg'].h_wind,
            c_circ=data['uwg'].c_circ,
            c_exch=data['uwg'].c_exch,
            maxday=data['uwg'].maxday,
            maxnight=data['uwg'].maxnight,
            windmin=data['uwg'].windmin,
            h_obs=data['uwg'].h_obs,
            epw_path=epw_path,
            new_epw_dir=None,
            new_epw_name=None,
            ref_bem_vector=None,
            ref_sch_vector=None,
        )
        # model = UWG.from_param_args(epw_path=epw_path, bldheight=float(data[0].bldheight), blddensity=0.5,
        #                             vertohor=0.8, grasscover=0.1, treecover=0.1, zone=data[0].zone)

        # Uncomment these lines to initialize the UWG model using a .uwg parameter file
        # param_path = "initialize_singapore.uwg"  # available in resources directory.
        # model = UWG.from_param_file(param_path, epw_path=epw_path)
        readDOE(serialize_output=True)
        model.generate()
        model.simulate()

        # Write the simulation result to a file.
        model.write_epw()


if __name__ == '__main__':
    uwg = TemperatureCalculation()
    uwg.single_run(
        geoclimate_pah=r'C:\Users\simon\TIPEE\TIPEE - PLABAT\2.8 GROUPE THERMIQUE\2.8.6 Simulations thermiques\Travail\notebooks\geoclimate-data/',
        epw_path=r'C:\Users\simon\TIPEE\TIPEE - PLABAT\2.8 GROUPE THERMIQUE\2.8.6 Simulations thermiques\Travail\notebooks/FRA_AC_La.Rochelle.Intl.AP.073160_TMYx.epw',
    )
