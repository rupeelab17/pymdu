import glob
import multiprocessing

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from joblib import Parallel, delayed
from osgeo import gdal
from pythermalcomfort.models import utci
from tqdm import tqdm

from pymdu.GeoCore import GeoCore
from pymdu.image import geotiff


class UTCI(GeoCore):
    def __init__(
        self,
        input_path=r'E:\RESULTATS_ATLANTECH\ATLANTECH_0\UMEP',
        weather_file=r'T:\FR_MDU/LaRochelle_historical_IPSL_bc_type_list_UMEP.txt',
    ):
        self.UTCI = {}
        self.weather_file = weather_file
        self.input_path = input_path
        self.results = pd.DataFrame()

    def recup_Tmr_files(self):
        self.list_of_files = glob.glob(self.input_path + '/' + 'Tmrt*.tif')

        return self

    def recup_shadow_files(self):
        self.list_of_shadows = glob.glob(self.input_path + '/' + 'Shadow*.tif')
        return self

    def extract_umep_weather(self):
        """

        Returns:

        """
        file_path = self.weather_file
        # TODO : cette fonction dans
        umep_header = [
            '%iy',
            'id',
            'it',
            'imin',
            'Q*',
            'QH',
            'QE',
            'Qs',
            'Qf',
            'Wind',
            'RH',
            'Td',
            'press',
            'rain',
            'Kdn',
            'snow',
            'ldown',
            'fcld',
            'wuh',
            'xsmd',
            'lai_hr',
            'Kdiff',
            'Kdir',
            'Wd',
        ]
        weather = pd.read_csv(file_path, sep=' ', index_col=None)
        weather.set_axis(umep_header, axis=1, inplace=True)

        self.results['Td'] = weather['Td'].values
        self.results['RH'] = weather['RH'].values
        self.results['Wind'] = weather['Wind'].values
        self.results['Wd'] = weather['Wd'].values
        self.results['%iy'] = weather['%iy'].values
        self.results['id'] = weather['id'].values
        self.results['it'] = [str(x).zfill(2) + '00' for x in weather['it'].values]
        self.results['tmr'] = [
            'Tmrt_' + str(iy) + '_' + str(id) + '_' + str(it)
            for (iy, id, it) in zip(
                self.results['%iy'], self.results['id'], self.results['it']
            )
        ]
        self.results['shadows'] = [
            'Shadow_' + str(iy) + '_' + str(id) + '_' + str(it)
            for (iy, id, it) in zip(
                self.results['%iy'], self.results['id'], self.results['it']
            )
        ]
        self.results['index'] = [
            str(iy) + '_' + str(id) + '_' + str(it)
            for (iy, id, it) in zip(
                self.results['%iy'], self.results['id'], self.results['it']
            )
        ]

        self.results.index = self.results['index']

        tmr = pd.DataFrame()
        list_tif = self.list_of_files
        temp_list = []
        temp_path = []
        for index in list_tif:
            temp_list.append(
                index.split('\\')[-1]
                .split('Tmrt_')[-1]
                .split('D.tif')[0]
                .split('N.tif')[0]
            )
            temp_path.append(index)

        tmr['data'] = ['Tmr' for x in temp_list]
        tmr['tmr-path'] = temp_path
        tmr.index = temp_list

        shades = pd.DataFrame()
        list_tif = self.list_of_shadows
        temp_list = []
        temp_path = []
        for index in list_tif:
            temp_list.append(
                index.split('\\')[-1]
                .split('Shadow_')[-1]
                .split('D.tif')[0]
                .split('N.tif')[0]
            )
            temp_path.append(index)
        shades['shadows-path'] = temp_path
        shades['data'] = ['Tmr' for x in temp_list]
        shades.index = temp_list

        self.results = pd.concat([self.results, tmr, shades], axis=1)
        self.results = self.results.dropna()

        return self.results

    def run_compute_utci(self):
        self.UTCI = {}
        Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend='threading')(
            delayed(self.compute_utci_dev)(row)
            for k, row in tqdm(self.results.iterrows())
        )

    def compute_utci_dev(self, row, save=False):
        """

        Returns:
            object:
        """
        # TODO : faire une boucle

        # traitement des ombres
        dataset = rasterio.open(fp=row['tmr-path'])
        dataset_rio = rioxarray.open_rasterio(filename=row['tmr-path'])
        new_data = dataset_rio.copy()

        step1 = gdal.Open(row['tmr-path'], gdal.GA_ReadOnly)
        GT_input = step1.GetGeoTransform()
        step2 = step1.GetRasterBand(1)
        self.img_as_array = step2.ReadAsArray()

        size1, size2 = self.img_as_array.shape
        self.output = np.zeros(shape=(size1, size2))

        for i in range(0, size1):
            self.output[i, :] = utci(
                tdb=row['Td'],
                tr=self.img_as_array[i, :],
                v=row['Wind'],
                rh=row['RH'],
                limit_inputs=False,
            )

        new_data.data[0] = self.output
        self.UTCI[row['tmr']] = new_data
        return self.UTCI

    def compute_utci(self, save=False):
        """

        Returns:
            object:
        """
        # TODO : faire une boucle
        UTCI = {}

        for k, row in tqdm(self.results.iterrows()):
            # traitement des ombres
            dataset = rasterio.open(fp=row['tmr-path'])
            dataset_rio = rioxarray.open_rasterio(filename=row['tmr-path'])
            new_data = dataset_rio.copy()
            image = dataset.read()

            step1 = gdal.Open(row['tmr-path'], gdal.GA_ReadOnly)
            GT_input = step1.GetGeoTransform()
            step2 = step1.GetRasterBand(1)
            img_as_array = step2.ReadAsArray()

            size1, size2 = img_as_array.shape
            output = np.zeros(shape=(size1, size2))

            for i in range(0, size1):
                output[i, :] = utci(
                    tdb=row['Td'],
                    tr=img_as_array[i, :],
                    v=row['Wind'],
                    rh=row['RH'],
                    limit_inputs=False,
                )

            new_data.data[0] = output
            UTCI[row['tmr']] = new_data
        return UTCI

    def create_tif_from_rio(self, utci_dict):
        for file in utci_dict.keys():
            utci_dict[file].rio.to_raster(
                self.input_path + f'/utci_{file}.tif',
                compress='lzw',
                bigtiff='NO',
                num_threads='all_cpus',
                driver='GTiff',
                predictor=2,
                discard_lsb=2,
            )

    def create_shadows_shp(self, table):
        for path, name in zip(table['shadows-path'], table['shadows']):
            gdf = geotiff.raster_to_gdf(src_tif=path, new_field_name='Shadow')
            gdf.to_file(self.input_path + f'/{name}.shp', 'ESRI Shapefile')

    def create_clipped_utci(self, table):
        for k, row in table.iterrows():
            tmr = row['tmr']
            shades = row['shadows']
            geotiff.clip_raster(
                dst_tif=self.input_path + f'/clipped_utci_{tmr}.tif',
                src_tif=self.input_path + f'/utci_{tmr}.tif',
                format='GTiff',
                cut_shp=self.input_path + f'/{shades}.shp',
                cut_name=f'{shades}',
            )

    def mask_clipped_utci(self):
        MASKED = {}
        all_files = glob.glob(self.input_path + '/clipped*')
        for file in all_files:
            name = file.split('\\')[-1].split('.tif')[0]

            src = rasterio.open(file)

            # test ok
            # =====
            msk = src.read(1) == -9999
            myplot = np.ma.array(data=src.read(1), mask=msk, fill_value=np.nan)
            MASKED[name] = myplot

        return MASKED

    def shab_quartier(
        self, bld_path=r'C:\Users\simon\python-scripts\exemple-umep/buildings.shp'
    ):
        data = gpd.read_file(bld_path)
        data['hauteur'] = data['hauteur'].fillna(3)
        data['etage'] = [max(int(x), 1) for x in data['hauteur'] // 3]
        data['shab'] = data['etage'] * data['geometry'].area
        return data['shab'].sum()

    def calculate_avg_table_utci(self, masked, table, shab=1000):
        # calcul de l'utci moyen dans les ombres
        liste_utci = []
        for tmr in table['tmr']:
            k = 0
            for key in masked.keys():
                if tmr in key:
                    liste_utci.append(masked[key].mean())
                    k = k + 1
                else:
                    pass

            if k == 1:
                pass
            else:
                liste_utci.append(np.nan)
        table['utci'] = liste_utci

        # calcul de la surrface totale d'ombre
        list_shadows = glob.glob(self.input_path + '/Shadow*.shp')
        liste_shades_area = []
        for shade in table['shadows']:
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
        table['area_shdows'] = liste_shades_area
        table['si/shab'] = [x / shab for x in table['area_shdows']]

        table['category'] = [2 if x > 32 else 0 if x < 28 else 1 for x in table['utci']]

        return table

    def run(self):
        self.recup_Tmr_files()
        self.recup_shadow_files()
        table = self.extract_umep_weather()
        utci = self.compute_utci()
        self.create_tif_from_rio(utci)
        self.create_clipped_utci(table)
        shab = self.shab_quartier()
        masked = self.mask_clipped_utci()
        table_utci = self.calculate_avg_table_utci(masked, table, shab=shab)
        table_utci.to_csv(r'C:\Users\simon\python-scripts\pymdu/my_example.csv')


if __name__ == '__main__':
    test = UTCI()
    test.run()
