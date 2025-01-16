import glob
import math
import os

import numpy
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from tqdm import tqdm

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.image import geotiff


class IndicatorsComfort:
    def __init__(
        self,
        output_path: os.path,
        datasets_path: os.path,
        model_tif_like: os.path = None,
        pedestrians_shp_path: os.path = None,
    ):
        self.output_path = output_path if output_path else TEMP_PATH
        self.datasets_path = datasets_path
        self.model_tif_like_path = model_tif_like
        self.pedestrians_shp_path = pedestrians_shp_path
        self.results: pd.DataFrame = None

    def run(self):
        self.__prepare_indicator()
        self.__compute_indicator()

    def __prepare_indicator(self):
        Shadow = xr.open_mfdataset(os.path.join(self.datasets_path, 'Shadow*.nc'))
        Tmrt = xr.open_mfdataset(os.path.join(self.datasets_path, 'Tmrt*.nc'))
        Utci_shadow = xr.open_mfdataset(os.path.join(self.datasets_path, 'ShdUtci*.nc'))
        # Utci_soleil = xr.open_mfdataset(os.path.join(self.datasets_path, 'SunUtci*.nc'))
        Utci = xr.open_mfdataset(os.path.join(self.datasets_path, 'Utci*.nc'))

        # Utci_shadow["Utci_shadow"][10].rio.to_raster(os.path.join(self.datasets_path, 'shade_utci.tif'))

        geotiff.shp_to_tif(
            InputShp=self.pedestrians_shp_path,
            OutputImage=os.path.join(self.output_path, 'pedestrians.tif'),
            RefImage=self.model_tif_like_path,
        )

        zone_pietonne = rioxarray.open_rasterio(
            os.path.join(self.output_path, 'pedestrians.tif')
        )
        ds_zone_pietonne = zone_pietonne.to_dataset(name='Pedestrians')
        # test % ombrages sur zone pietonne
        ombrage_liste = []
        for x in tqdm(range(len(Shadow['Shadow']))):
            ombrage = Shadow['Shadow'][x].copy()
            ombrage = ombrage.rename('OmbragePieton')

            # l'ombre (1 à l'ombre)
            arr = np.array(Shadow['Shadow'][x].data)
            where_0 = np.where(arr > 0)
            where_1 = np.where(arr == 1)
            arr[where_0] = 1
            arr[where_1] = 0

            # le masque (1 dans la zone piétonne)
            mask = zone_pietonne.data.copy()
            where_0 = np.where(mask == 0)
            where_1 = np.where(mask == 1)
            mask[where_0] = 1
            mask[where_1] = 0

            ombrage.data = np.ma.array(data=arr, mask=mask, fill_value=np.nan)
            ombrage_liste.append(ombrage)

        ombrage_dataset = xr.concat(ombrage_liste, dim=Shadow['Shadow'].time)
        Shadow = xr.merge([Shadow, ombrage_dataset])

        datakey_to_csv = ['%ombre_sur_aire_pietonne', 'utci-a-lombre', 'utci-au-soleil']

        data_to_csv = {}
        for k, time in tqdm(enumerate(Shadow.time.data)):
            data_to_csv[time] = []
            # =====================================================
            # TODO : améliorer cette partie
            # problème avec les np.float32
            # =====================================================
            # %ombre
            unique, counts = np.unique(
                np.array(Shadow['OmbragePieton'][k].data), return_counts=True
            )
            shadow_zone = dict(zip(unique, counts))
            percent_ombre = shadow_zone
            liste_keys = list(shadow_zone.keys())[:-1]
            print(shadow_zone)
            try:
                percent_ombre = shadow_zone[1.0] / (shadow_zone[1.0] + shadow_zone[0.0])
                data_to_csv[time].append(percent_ombre)
            except:
                percent_ombre = 1
                data_to_csv[time].append(percent_ombre)
            # utci à l'ombre
            data_to_csv[time].append(
                np.nanmean(np.array(Utci_shadow['Utci_shadow'][k].data))
            )

            # utci au soleil
            merge_xr = xr.merge(
                [Utci['Utci'][k], Shadow['Shadow'][k]], compat='override'
            )
            data_utci_soleil = merge_xr.where(merge_xr.Shadow > 0.5)
            ds_zone_pietonne['x'] = data_utci_soleil['x']
            ds_zone_pietonne['y'] = data_utci_soleil['y']
            dataset = xr.merge([data_utci_soleil, ds_zone_pietonne], compat='override')
            data_utci_soleil_zone_pietonne = dataset.where(dataset.Pedestrians > 0.1)
            data_to_csv[time].append(
                np.nanmean(np.array(data_utci_soleil_zone_pietonne['Utci'].data))
            )

        self.results = pd.DataFrame.from_dict(
            data_to_csv, orient='index', columns=datakey_to_csv
        )
        # Correction utci_ombre
        correction_utci_shadow = []
        for utci_shadow, utci_soleil in tqdm(
            zip(self.results['utci-a-lombre'], self.results['utci-au-soleil'])
        ):
            if math.isnan(utci_shadow):
                correction_utci_shadow.append(utci_soleil)
            else:
                correction_utci_shadow.append(utci_shadow)

        self.results['utci-a-lombre'] = correction_utci_shadow
        # self.results.to_csv(os.path.join(self.datasets_path, 'indicateurs.csv'))

    @staticmethod
    def PPD_pmv(PMV):
        """
        PPD = f(PMV)
        """

        PMV = max(0, PMV)
        return 100 - 95 * numpy.exp(-0.03353 * PMV**4 - 0.2179 * PMV**2)

    @staticmethod
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

    def __compute_indicator(self):
        # les valeurs clés sont ici
        # ============================
        shad_policy = [2, 25]
        utci_policy = [26, 38]
        print_to_console = False
        # ============================

        ppd_utci = []
        ppd_shade = []
        ppd_indicateur = []
        for shad, utci_ombre, utci_soleil, time in zip(
            self.results['%ombre_sur_aire_pietonne'],
            self.results['utci-a-lombre'],
            self.results['utci-au-soleil'],
            self.results.index,
        ):
            ombre_max = shad_policy[0] + (shad_policy[1] - shad_policy[0]) * (
                utci_soleil - utci_policy[0]
            ) / (utci_policy[1] - utci_policy[0])
            # ombre_max = max(ombre_max,0)
            shad = 100 * shad

            ppd_utci.append(self.PPD_pmv(self.UTCI_to_PMV(utci_ombre)))
            PPD_utci_value = self.PPD_pmv(self.UTCI_to_PMV(utci_ombre))
            if shad == 100:
                ppd_shade.append(5.0)
                PPD_shad_value = 5.0
            elif utci_soleil < 26:
                ppd_shade.append(5.0)
                PPD_shad_value = 5.0
            else:
                ppd_shade.append(self.PPD_shad(shad, limShad=[ombre_max, 0]))
                PPD_shad_value = self.PPD_shad(shad, limShad=[ombre_max, 0])
            ppd_indicateur.append(max(PPD_utci_value, PPD_shad_value))

            if print_to_console:
                print(time)
                print('shad        ', shad)
                print('ombre_max    ', ombre_max)
                print('utci_ombre  ', utci_ombre)
                print('PPD_shad    ', PPD_shad_value)
                print('utci_soleil ', utci_soleil)
                print('PPD_utci    ', PPD_utci_value)
                print('#===================')

        self.results['ppd'] = ppd_indicateur
        self.results['ppd_shade'] = ppd_shade
        self.results['ppd_utci'] = ppd_utci
        self.results.to_csv(os.path.join(self.datasets_path, 'indicators.csv'))


########################################################################################################################
#
########################################################################################################################
if __name__ == '__main__':
    model_tif = glob.glob(os.path.join('Ressources', 'Tmrt*.tif'))[0]
    indicator = IndicatorsComfort(
        output_path='Ressources',
        datasets_path='Ressources/datasets',
        model_tif_like=model_tif,
        pedestrians_shp_path='Ressources/pedestrians.shp',
    )
    indicator.run()
