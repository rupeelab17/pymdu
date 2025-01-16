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

import datetime
import glob
import os
from datetime import datetime
from math import exp

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import seaborn as sns
from osgeo import gdal
from pythermalcomfort.models import utci
from tqdm import tqdm

from pymdu.image import geotiff
from pymdu.physics.comfort.Utci import UTCI


class HeatStressIndicators(object):
    def __init__(self):
        self.input_path = r'T:\FR_MDU\ATLANTECH_3'
        self.umep_path = r'E:\RESULTATS_ATLANTECH\ATLANTECH_3'
        self.urock_path = r'T:\FR_MDU\ATLANTECH_2\UROCK-proj'
        self.weather_file = (
            r'T:\FR_MDU\LaRochelle_historical_IPSL_bc_type_list_UMEP.txt'
        )
        self.pedestrian_area = 'pedestrian.shp'
        self.name_result_file = r'result.csv'
        self.pedestrian_path = os.path.join(self.input_path, self.pedestrian_area)
        self.data = UTCI(input_path=self.umep_path, weather_file=self.weather_file)
        self.filepath = './data/solweig/results/'

    def recup_umep_data(self):
        self.data.recup_Tmr_files()
        self.data.recup_shadow_files()
        self.result = self.data.extract_umep_weather()
        return self

    def compute_utci(self, write=True, urock=False):
        UTCI = {}
        all_wind_files = glob.glob(self.urock_path + '/*')
        for k, row in tqdm(self.result.iterrows()):
            # traitement des ombres
            dataset = rasterio.open(fp=row['tmr-path'])
            dataset_rio = rioxarray.open_rasterio(filename=row['tmr-path'])
            new_data = dataset_rio.copy()
            image = dataset.read()

            step1 = gdal.Open(row['tmr-path'], gdal.GA_ReadOnly)
            GT_input = step1.GetGeoTransform()
            step2 = step1.GetRasterBand(1)
            tmr_as_array = step2.ReadAsArray()
            size1, size2 = tmr_as_array.shape
            output = np.zeros(shape=(size1, size2))

            if urock:
                dir = round(row['Wd'] / 10) * 10
                speed = row['Wind']
                wind_file = f'/DIR_{int(dir)}_VENT_1.tif'
                wind_file_path = self.urock_path + wind_file
                # select file in the U-rock folder

                step1 = gdal.Open(wind_file_path, gdal.GA_ReadOnly)
                GT_input = step1.GetGeoTransform()
                step2 = step1.GetRasterBand(1)
                wind_as_array = speed * step2.ReadAsArray()

                for i in range(0, size1):
                    #         #>1.1. Calcul indicateurs sans vent
                    #         output[i, :] = utci(tdb=row['Td'], tr=img_as_array[i, :], v=2., rh=row['RH'], limit_inputs=False)
                    # >1.2. Calcul indicateurs avec vent
                    output[i, :] = utci(
                        tdb=row['Td'],
                        tr=tmr_as_array[i, :],
                        v=wind_as_array[i, :],
                        rh=row['RH'],
                        limit_inputs=False,
                    )

            else:
                for i in range(0, size1):
                    output[i, :] = utci(
                        tdb=row['Td'],
                        tr=tmr_as_array[i, :],
                        v=row['Wind'],
                        rh=row['RH'],
                        limit_inputs=False,
                    )

            new_data.data[0] = output
            UTCI[row['tmr']] = new_data
            name = row['tmr']
            if write:
                UTCI[row['tmr']].rio.to_raster(
                    self.umep_path
                    + f'/utci_{name}_WindDir_{int(dir)}_WindSpeed_{int(speed)}.tif',
                    compress='lzw',
                    bigtiff='NO',
                    num_threads='all_cpus',
                    driver='GTiff',
                    predictor=2,
                    discard_lsb=2,
                )

        return self

    def compute_utci_in_shadows(self, write=True):
        # conversion des shadows.tif en shadows.shp
        for path, name in zip(self.result['shadows-path'], self.result['shadows']):
            gdf = geotiff.raster_to_gdf(src_tif=path, new_field_name='Shadow')
            gdf.to_file(self.umep_path + f'/{name}.shp', 'ESRI Shapefile')

        # clip des rasters utci dans les zones ombragées
        for k, row in self.result.iterrows():
            tmr = row['tmr']
            shades = row['shadows']
            src_tif = glob.glob(self.umep_path + f'/utci_{tmr}*.tif')[0]
            geotiff.clip_raster(
                dst_tif=self.umep_path + f'/clipped_utci_{tmr}.tif',
                src_tif=src_tif,
                format='GTiff',
                cut_shp=self.umep_path + f'/{shades}.shp',
                cut_name=f'{shades}',
            )

        # calcul de la moyenne des UTCI, au soleil et à l'ombre
        MASKED = {}
        all_files = glob.glob(self.umep_path + '/clipped*')
        for file in all_files:
            name = file.split('\\')[-1].split('.tif')[0]

            src = rasterio.open(file)

            # test ok
            # =====
            msk = src.read(1) == -9999
            myplot = np.ma.array(data=src.read(1), mask=msk, fill_value=np.nan)
            MASKED[name] = myplot

        #
        # UTCI moyen de la zone
        #

        GLOBAL_UTCI = {}
        all_files = glob.glob(self.umep_path + '/utci*')
        for file in all_files:
            name = file.split('\\')[-1].split('.tif')[0]

            src = rasterio.open(file)

            # test ok
            # =====
            msk = src.read(1) == -9999
            myplot = np.ma.array(data=src.read(1), mask=msk, fill_value=np.nan)
            GLOBAL_UTCI[name] = myplot

        # sauvegarde dans l'objet results

        liste_utci_shades = []
        for tmr in tqdm(self.result['tmr']):
            k = 0
            for key in MASKED.keys():
                if tmr in key:
                    liste_utci_shades.append(MASKED[key].mean())
                    k = k + 1
                else:
                    pass

            if k == 1:
                pass
            else:
                liste_utci_shades.append(np.nan)

        liste_utci_global = []
        for tmr in tqdm(self.result['tmr']):
            k = 0
            for key in GLOBAL_UTCI.keys():
                if tmr in key:
                    liste_utci_global.append(GLOBAL_UTCI[key].mean())
                    k = k + 1
                else:
                    pass

            if k == 1:
                pass
            else:
                liste_utci_global.append(np.nan)
        self.result['utci-a-lombre'] = liste_utci_shades
        self.result['utci-au-soleil'] = liste_utci_global

        # sauvegarde de l'aire totale de l'ombre sur le sol

        list_shadows = glob.glob(self.umep_path + '/Shadow*.shp')
        liste_shades_area = []
        for shade in self.result['shadows']:
            k = 0
            for key in list_shadows:
                if shade in key:
                    data = gpd.read_file(key)
                    data['area'] = [x.area for x in data['geometry']]
                    liste_shades_area.append(data['area'].sum())
                    k = k + 1
                else:
                    pass

            if k == 1:
                pass
            else:
                liste_shades_area.append(np.nan)
        self.result['area_shadows'] = liste_shades_area

        # calculs sur la zone piétonne
        pedestrian = gpd.read_file(
            os.path.join(self.input_path, 'pedestrian.shp')
        ).explode(ignor_index=True)
        pedestrian.to_file(os.path.join(self.input_path, 'pieton.shp'))
        Spd = gpd.read_file(os.path.join(self.input_path, 'pieton.shp'))
        Spd['area'] = [x.area for x in Spd['geometry']]
        s_pd = Spd['area'].sum()
        liste_sshad_spd = []
        aire_pietonne = []
        ombre_sur_aire_pietonne = []
        for k, row in self.result.iterrows():
            name = row['tmr'].split('Tmrt_')[-1]
            shadow = gpd.read_file(os.path.join(self.umep_path, f'Shadow_{name}.shp'))
            gdf = gpd.clip(shadow, Spd)
            gdf['area'] = [x.area for x in gdf['geometry']]
            gdf.to_file(os.path.join(self.umep_path, f'Shadow_Pedestrian_{name}.shp'))
            s_shad = gdf['area'].sum()
            liste_sshad_spd.append(s_shad / s_pd)
            aire_pietonne.append(s_pd)
            ombre_sur_aire_pietonne.append(s_shad)
        self.result['%ombre_sur_aire_pietonne'] = liste_sshad_spd
        self.result['ombre_sur_aire_pietonne'] = ombre_sur_aire_pietonne
        self.result['aire_pietonne'] = aire_pietonne

        if write:
            self.result.to_csv(os.path.join(self.input_path, self.name_result_file))

        return self

    # TODO : mieux l'écrire > voir avec BB
    # 1. fonction PPD = f(PMV)
    def PPD(self, PMV):
        PMV = max(0, PMV)
        return 100 - 95 * exp(-0.03353 * PMV**4 - 0.2179 * PMV**2)

    # 2. Une interpolation linéaire PMV = f(UTCI)
    def UTCI_to_PMV(self, UTCI, limPMV=[0.5, 3], limUTCI=[26, 42]):
        return (limPMV[1] - limPMV[0]) * (UTCI - limUTCI[0]) / (
            limUTCI[1] - limUTCI[0]
        ) + limPMV[0]

    # 3. Une interpolation linéraire PPD = f(%ombre)
    def PPDshad(self, shad, limShad=[100, 0], limPMV=[0.5, 3]):
        PMVeqShad = (limPMV[1] - limPMV[0]) * (shad - limShad[1]) / (
            limShad[0] - limShad[1]
        ) + limPMV[0]
        PMVeqShad = max(0, PMVeqShad)
        return self.PPD(PMVeqShad)

    # 4. Une interpolation linéaire %ombre_max = f(UTCI_soleil)
    def shadow_max(self, utci_soleil, limShad=[10, 75], limUTCI=[26, 38]):
        shadow_value = limShad[0] + (limShad[1] - limShad[0]) * (
            utci_soleil - limUTCI[0]
        ) / (limUTCI[1] - limUTCI[0])
        return shadow_value

    # 5. La fonction PPDshad = f(%ombre, UTCI_ombre)
    def PPDshad_shad_utci(self, shad, utci_ombre, limPMV=[0.5, 3]):
        moving_shadow = self.shadow_max(shad)
        moving_pmv = self.UTCI_to_PMV(utci_ombre)
        PMVeqShad = (limPMV[1] - moving_pmv) * (shad - 0) / (
            moving_shadow - 0
        ) + moving_pmv
        PMVeqShad = max(0, PMVeqShad)
        return self.PPD(PMVeqShad)

    def run(self):
        indicator = HeatStressIndicators()
        result = indicator.recup_umep_data()
        utci = result.compute_utci(urock=True)
        utci_in_shadows = utci.compute_utci_in_shadows()

    def collect_data(main_path='./brut/'):
        """
        récupération des données
        """
        list_files = glob.glob(main_path + '/*.csv')
        list_ppd = glob.glob('./*ppd.csv')
        data = {}
        for file in list_files:
            name = file.split('\\')[-1].split('.csv')[0]
            data[name] = pd.read_csv(file, sep=',')
        for file_ppd in list_ppd:
            os.remove(file_ppd)
        return data

    def PPD_pmv(PMV):
        """
        PPD = f(PMV)
        """
        PMV = max(0, PMV)
        return 100 - 95 * exp(-0.03353 * PMV**4 - 0.2179 * PMV**2)

    # 2. Une interpolation linéaire PMV = f(UTCI)
    def UTCI_to_PMV(UTCI, limPMV=[0.5, 3], limUTCI=[26, 42]):
        """
        UTCI = f(PMV)
        """
        return (limPMV[1] - limPMV[0]) * (UTCI - limUTCI[0]) / (
            limUTCI[1] - limUTCI[0]
        ) + limPMV[0]

    # 3. Une interpolation linéraire PPD = f(%ombre)
    def PPD_shad(self, shad, limShad=[100, 0], limPMV=[0.5, 3]):
        """
        PPD_shadows = f(shadow)
        """
        PMVeqShad = limPMV[1] - (limPMV[1] - limPMV[0]) * (100 * shad - limShad[1]) / (
            limShad[0] - limShad[1]
        )

        return self.PPD_pmv(PMVeqShad)

    def calculate_indicators(
        self, data, print_to_console=True, shad_policy=[10, 75], utci_policy=[28, 42]
    ):
        """
        la fonction qui réalise le calcul des PPDs
        """
        try:
            all_ppd_files = glob.glob('*_ppd.csv')
            for file in all_ppd_files:
                os.remove(file)
        except:
            pass

        for name in data.keys():
            ppd_utci = []
            ppd_shade = []
            ppd_indicateur = []
            for shad, utci_ombre, utci_soleil in zip(
                data[name]['%ombre_sur_aire_pietonne'],
                data[name]['utci-a-lombre'],
                data[name]['utci-au-soleil'],
            ):
                ombre_max = shad_policy[0] + (shad_policy[1] - shad_policy[0]) * (
                    utci_soleil - utci_policy[0]
                ) / (utci_policy[1] - utci_policy[0])
                # ombre_max = max(ombre_max,0)
                if print_to_console:
                    print('shad        ', shad)
                    print('ombre_max    ', ombre_max)
                    print('utci_ombre  ', utci_ombre)
                    print('PPD_shad    ', self.PPD_shad(shad, limShad=[ombre_max, 0]))
                    print('utci_soleil ', utci_soleil)
                    print('PPD_utci    ', self.PPD_pmv(self.UTCI_to_PMV(utci_ombre)))
                    print('#===================')
                ppd_utci.append(self.PPD_pmv(self.UTCI_to_PMV(utci_ombre)))

                ppd_shade.append(self.PPD_shad(shad, limShad=[ombre_max, 0]))
                ppd_indicateur.append(
                    max(
                        self.PPD_pmv(self.UTCI_to_PMV(utci_ombre)),
                        self.PPD_shad(shad, limShad=[ombre_max, 0]),
                    )
                )

            data[name]['ppd-utci'] = ppd_utci
            data[name]['ppd-shade'] = ppd_shade
            data[name]['ppd-indicator'] = ppd_indicateur
            data[name].to_csv(name + '_ppd.csv')

        return data

    def classe_confort(
        self,
        data,
        key='utci-a-lombre',
        interval=[26, 32, 38, 46],
        values=[0, 1, 2, 3, 4],
    ):
        """
        Determination de la classe confort, pour les plots
        """
        for n in data.keys():
            classe_confort = []
            for utci in data[n][key]:
                if utci < interval[0]:
                    classe_confort.append(values[0])
                # 93d150
                elif utci < interval[1]:
                    classe_confort.append(values[1])
                # ffc100
                elif utci < interval[2]:
                    classe_confort.append(values[2])
                # ff9932
                elif utci < interval[3]:
                    classe_confort.append(values[3])
                # ff3200
                else:
                    classe_confort.append(values[4])
            data[n]['classe-confort'] = classe_confort
        return data

    def prepare_to_plot(self, data, main_path='./', year=2022):
        """
        Mise en forme des données pour le plot, calcul de data_max
        """
        list_files = glob.glob(main_path + '/*_ppd.csv')
        data = {}
        data_max = {}
        YEAR = year
        for file in list_files:
            name = file.split('\\')[-1].split('.csv')[0]
            df = pd.read_csv(file, sep=',')

            # ajout des dates
            liste_date = []
            for day, hour in zip(df['id'], df['it']):
                d = (
                    datetime.datetime(YEAR, 1, 1)
                    + datetime.timedelta(days=day - 7)
                    + datetime.timedelta(hours=hour / 100)
                )
                liste_date.append(d.strftime('%Y-%m-%d %H:%M:%S'))
            df['date'] = liste_date
            df.index = pd.to_datetime(liste_date)
            df['hours'] = df.index.hour
            df['day'] = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in df.index]
            df['solar_hot'] = np.where(df['utci-au-soleil'] > 26, 1, np.nan)
            df['ppd-indicator'] = df['ppd-indicator'] * df['solar_hot']
            df['ppd-shade'] = df['ppd-shade'] * df['solar_hot']
            df['ppd-utci'] = df['ppd-utci'] * df['solar_hot']

            s = df.groupby(pd.Grouper(freq='D'))['ppd-indicator'].transform('max')
            data_max[name] = df[df['ppd-indicator'] == s]
            data[name] = df
        return data, data_max

    def heatmap_with_specified_thresold(
        self,
        data,
        file='result-atlantech-actuel-0_ppd',
        key='utci-a-lombre',
        to_plot='ppd-utci',
        thresold=[9, 26, 32, 38, 42],
        thresold_colors=['#93d150', '#ffc100', '#ff9932', '#ff3200', '#bf0000'],
    ):
        """
        le tracé adaptatif des heatmap
        """
        df = data[file]
        df['solar_hot'] = np.where(df['utci-au-soleil'] > 26, 1, np.nan)
        df['ppd-indicator'] = df['ppd-indicator'] * df['solar_hot']
        df['ppd-shade'] = df['ppd-shade'] * df['solar_hot']
        df['ppd-utci'] = df['ppd-utci'] * df['solar_hot']

        df.index = pd.to_datetime(df['date'])
        df['date-heatmap'] = pd.to_datetime(df['date'])

        # cmap_name = 'my_list'
        # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        # # create the new map
        # import matplotlib, numpy as np, matplotlib.pyplot as plt
        # from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        # ['#93d150', '#ffc100', '#ff9932', '#ff3200', '#bf0000']
        # shades_colors  = ['#757575','#bdbdbd', '#e0e0e0', '#eeeeee', '#fafafa']
        utci_colors = thresold_colors
        classe_utci = []
        for x in df[key]:
            if x < thresold[0]:
                classe_utci.append(0)
            elif x < thresold[1]:
                classe_utci.append(1)
            elif x < thresold[2]:
                classe_utci.append(2)
            elif x < thresold[3]:
                classe_utci.append(3)
            else:
                classe_utci.append(4)
        df['color'] = classe_utci
        dataname = 'color'
        datalabel = 'UTCI$_{shades}$'  # nom de la variable mis en forme au format LATEX pour les variables scientifiques pour des légendes propres
        cmap = utci_colors
        # add a column hours and days
        df['hourOfTheDay'] = df.index.hour
        # df['date'] = pd.to_datetime(df.index.date)

        piv = df.pivot(index='hours', columns='day', values=to_plot)
        # pmin_max = pd.pivot_table(df, values=dataname, index=["date-heatmap"], aggfunc= [min, max]) # get min and max values for each day
        # piv.columns=piv.columns.strftime('%d-%b') # on met en forme les en têtes de colonne au format jour du mois/mois (ex 31-jan)
        # pmin_max.index=pmin_max.index.strftime('%d-%b')

        cmap = mpl.colors.ListedColormap(utci_colors)
        cmap.set_over('0.99')
        cmap.set_under('0.99')
        bounds = [9, 26, 32, 38, 42, 58]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            boundaries=[0] + bounds + [13],
            extend='both',
            ticks=bounds,
            spacing='proportional',
            orientation='horizontal',
        )
        cb2.set_label('UTCI$_{shades}$')
        sns.set(context='talk')
        # 0) définition FIGURE sur deux lignes / une colonne (2,1) + ratio de 1 pour 3 en hauteur (height_ratios)

        plt.savefig('colorbar.svg')
        figure, ax_dubas = plt.subplots(1, 1, figsize=(16, 5))
        # 1) HEATMAP graphique du bas (ax_dubas) + couleur cmap + légende colorbar sur le côté

        cmap = mpl.colors.ListedColormap(utci_colors[0 : int(df['color'].max() + 1)])
        # plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),  orientation="vertical", spacing='proportional')
        sns.heatmap(
            piv,
            xticklabels=30,
            yticklabels=6,
            ax=ax_dubas,
            cmap=cmap,
            edgecolor='k',
            linewidth=0.1,
            cbar=True,
        )
        ax_dubas.invert_yaxis()

    def time_index_from_filenames(self, filepath=None, namevar='Tmrt', year='2022'):
        if filepath is None:
            filepath = self.filepath
        # on supprime la dernière ligne car il y a une average.tif
        filenames = glob.glob(filepath + f'/{namevar}*.tif')[:-1]
        liste_of_day = [
            x.split(f'{namevar}_1997_')[-1].split('_')[0] for x in filenames
        ]
        liste_of_hour = [
            x.split('.tif')[0].split('_')[-1].split('00N')[0].split('00D')[0]
            for x in filenames
        ]
        liste_of_date = [
            datetime.strptime(year + '-' + day, '%Y-%j') for day in liste_of_day
        ]
        date = [
            res.strftime(f'%Y-%m-%d {hour}:00:00')
            for (res, hour) in zip(liste_of_date, liste_of_hour)
        ]
        return [pd.to_datetime(x) for x in date]


if __name__ == '__main__':
    indicator = HeatStressIndicators()
    indicator.run()
