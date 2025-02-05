import io
import os
import pickle
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from windrose import WindroseAxes, plot_windrose

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH


class Meteo(GeoCore):
    def __init__(
        self,
        output_path: str = None,
    ):
        self.output_path = output_path if output_path else TEMP_PATH
        self.init_weather_file = os.path.join(self.meteo_path, 'base_weather.txt')

        self.epw_columns = [
            'Year',
            'Month',
            'Day',
            'Hour',
            'Minute',
            'Data Source and Uncertainty Flags',
            'Dry Bulb Temperature',
            'Dew Point Temperature',
            'Relative Humidity',
            'Atmospheric Station Pressure',
            'Extraterrestrial Horizontal Radiation',
            'Extraterrestrial Direct Normal Radiation',
            'Horizontal Infrared Radiation Intensity',
            'Global Horizontal Radiation',
            'Direct Normal Radiation',
            'Diffuse Horizontal Radiation',
            'Global Horizontal Illuminance',
            'Direct Normal Illuminance',
            'Diffuse Horizontal Illuminance',
            'Zenith Luminance',
            'Wind Direction',
            'Wind Speed',
            'Total Sky Cover',
            'Opaque Sky Cover (used if Horizontal IR Intensity missing)',
            'Visibility',
            'Ceiling Height',
            'Present Weather Observation',
            'Present Weather Codes',
            'Precipitable Water',
            'Aerosol Optical Depth',
            'Snow Depth',
            'Days Since Last Snowfall',
            'Albedo',
            'Liquid Precipitation Depth',
            'Liquid Precipitation Quantity',
            '',
        ]

    def run(
        self,
        begin: str = '2022-06-30 00:00:00',
        end: str = '2022-06-30 23:00:00',
        weather_filename: str = None,
    ):
        self.__gen_umep_meteo_from_epw(
            begin=begin, end=end, weather_filename=weather_filename
        )

    def _get_weather_from_position(self, token_gitlab: str = None) -> pd.DataFrame:
        PointOfInterest = [self.bbox[1], self.bbox[0]]
        with open(
            os.path.join(self.meteo_path, 'PositionStations.pickle'), 'rb'
        ) as handle:
            WeatherPosition = pickle.load(handle)
        listOfStations = []
        listOfName = []
        for key in WeatherPosition.keys():
            listOfStations.append(
                [
                    float(WeatherPosition[key]['loc'][0]),
                    float(WeatherPosition[key]['loc'][1]),
                ]
            )
            listOfName.append(key)
        A = np.array(listOfStations)
        distances = np.linalg.norm(A - PointOfInterest, axis=1)
        min_index = np.argmin(distances)

        weather_filename_found = listOfName[min_index]

        WEATHERFILE_PATH = weather_filename_found + '.epw'
        print('WEATHERFILE_PATH FOUND', WEATHERFILE_PATH)

        if token_gitlab:
            raw_content = self.download_file_gitlab(
                host='https://gitlab.plateforme-tipee.com',
                token=token_gitlab,
                project_name='epw-data',
                branch_name='main',
                file_path=f'france/{WEATHERFILE_PATH}',
                output=None,
            )
        else:
            file_path = f'france/{WEATHERFILE_PATH}'
            raw_content = f'https://raw.githubusercontent.com/rupeelab17/epw-data/refs/heads/main/{file_path}'
        weather_dataframe = pd.read_csv(
            raw_content,
            low_memory=False,
            skiprows=8,
            names=self.epw_columns)

        return weather_filename_found, weather_dataframe

    def __gen_umep_meteo_from_epw(
        self,
        begin: str = '2022-06-30 00:00:00',
        end: str = '2022-06-30 23:00:00',
        weather_filename: str = None,
    ):
        if weather_filename:
            weather_dataframe = pd.read_csv(
                weather_filename, low_memory=False, skiprows=8, names=self.epw_columns
            )
        else:
            weather_filename, weather_dataframe = self._get_weather_from_position()

        begin_str = begin
        date_obj = datetime.strptime(begin_str, '%Y-%m-%d %H:%M:%S')
        annee = date_obj.year
        index = pd.date_range(start=f"{annee}-01-01 00:00:00", freq='1h', periods=8760)
        weather_dataframe.index = index

        umep_data = pd.DataFrame(
            index=range(len(weather_dataframe[begin:end].index)),
            columns=[
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
            ],
        )
        umep_data = umep_data.fillna(-999.00)
        weather_dataframe['year'] = [index.year for index in weather_dataframe.index]
        weather_dataframe['dayofyear'] = [
            index.dayofyear for index in weather_dataframe.index
        ]
        weather_dataframe['hour'] = weather_dataframe['Hour'].replace(
            to_replace=24, value=0
        )
        table_corresp = {
            'Wd': 'Wind Direction',
            'Wind': 'Wind Speed',
            'RH': 'Relative Humidity',
            'Td': 'Dry Bulb Temperature',
            'Kdn': 'Global Horizontal Radiation',
            'Kdiff': 'Diffuse Horizontal Radiation',
            'Kdir': 'Direct Normal Radiation',
            '%iy': 'year',
            'id': 'dayofyear',
            'it': 'hour',
            'imin': 'Minute',
        }

        # day + ' 00:00:00': day + ' 23:00:00'
        data_select = weather_dataframe[begin:end]
        print(data_select)
        for key in table_corresp.keys():
            umep_data[key] = data_select[table_corresp[key]].values

        weather_filename = weather_filename + '.txt'
        umep_data.to_csv(
            os.path.join(self.output_path, weather_filename), sep=' ', index=False
        )
        print(f'Data saved in {weather_filename}')
        return umep_data

    def gen_umep_weather_from_list(
        self,
        listofdays,
        epw_file='LaRochelle_historical_IPSL_bc_type.epw',
        init_weather_file='base_weather.txt',
        compute_night=True,
    ):
        """
        TODO : mieux définir les jours/nuits

        Args:
            listofdays:
            epw_file:
            init_weather_file:
            compute_night:
        """
        concat = pd.DataFrame()

        for k, day in enumerate(listofdays):
            data = self.__gen_umep_meteo_from_epw(
                begin='2022-06-30 00:00:00', end='2022-06-30 23:00:00'
            )
            data['id'] = [x + k for x in data['id']]
            concat = pd.concat([concat, data])

        if compute_night == False:
            evenings = concat[concat['it'] > 8]
            concat = evenings[evenings['it'] < 22]
        else:
            pass
        print(epw_file.replace('.epw', '_list_UMEP.txt'))
        concat.to_csv(epw_file.replace('.epw', '_list_UMEP.txt'), sep=' ', index=False)

        data = pd.read_csv(epw_file, names=self.epw_columns, skiprows=8)
        data.index = self.index

    @staticmethod
    def find_url_meteo_france(year: int = 1835, dep: int = 17) -> str | None:
        resp = requests.get(
            f'https://www.data.gouv.fr/api/2/datasets/6569b4473bedf2e7abad3b72/resources//?page=1&page_size=30&type=main&q=HOR_departement_{dep}_periode_'
        )
        data = resp.json()
        df_json = pd.DataFrame.from_dict(data['data'])

        # ======
        init = []
        end = []
        for x in list(df_json.title):
            period = x.split('_')[-1]
            init.append(int(period.split('-')[0]))
            end.append(int(period.split('-')[1]))

        df_json['init'] = init
        df_json['end'] = end
        # ======

        row = df_json.loc[(df_json['init'] <= year) & (df_json['end'] > year), ['url']]
        # idx = np.where((df_json['init'] <= year) & (df_json['end'] > year))
        if len(row['url'].values) == 0:
            return None

        return row['url'].values[0]

    @staticmethod
    def gen_windrose_from_dataframe(
        df: pd.DataFrame,
        column_wind_speed=' FXI',
        column_wind_direction='DXI',
        show: bool = False,
        write: bool = False,
        station: str = 'LA ROCHELLE-ILE DE RE',
        url: str = 'https://www.data.gouv.fr/api/2/datasets/6569b4473bedf2e7abad3b72/resources//?page=1',
    ):
        df['speed'] = df[column_wind_speed]
        df['direction'] = df[column_wind_direction]

        sns.axes_style('white')
        sns.set_style('ticks')
        sns.set_context('talk')

        fig = plt.figure(figsize=(100, 60))
        fig.set_size_inches(18, 10, forward=True)
        ax = WindroseAxes.from_ax()
        # ax.bar(df.direction.values, df.speed.values, bins=np.arange(0.01, 30, 4), cmap=cm.coolwarm_r, normed=True, lw=3)
        # plot_windrose(direction_or_df=df.direction, var=df.speed, kind="contour", bins=np.arange(0.01, 30, 4),
        #               cmap=cm.hot, lw=3)
        plot_windrose(
            direction_or_df=df,
            kind='contourf',
            bins=np.arange(0.01, 30, 4),
            cmap=cm.coolwarm_r,
            lw=3,
        )

        ax.set_legend(
            title='Vitesse du vent [km/h]',
            fontsize='11',
            bbox_to_anchor=(1.1, -0.3),
            fancybox=True,
            loc='lower right',
        )
        plt.title(
            station
            + '\n'
            + url
            + '\n'
            + '---'
            + '\n> force maximale du vent instantané dans l’heure, mesurée à 10 m (en m/s et 1/10)\n'
            + '> direction de FXI (en rose de 360)\n'
            + ''
        )

        file_byte = io.BytesIO()
        plt.savefig(file_byte, format='png', dpi=150)
        # Cleanup plot
        plt.close(plt.gcf())
        plt.clf()

        file_byte.seek(0)

        if write:
            with open('image.png', 'wb') as f:
                f.write(file_byte.getvalue())
            file_byte.seek(0)

        if show:
            plt.show()

        content = file_byte.read()
        return content


if __name__ == '__main__':
    meteo_test = Meteo(output_path=r'./')
    meteo_test.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    meteo_test.run(
        begin='2017-06-30 00:00:00',
        end='2018-06-30 23:00:00',
        weather_filename='FRA_AC_La.Rochelle.Intl.AP.073160_TMYx_UWG.epw',
    )

    url = meteo_test.find_url_meteo_france(year=2000, dep=17)
    print('meteo_test', url)
    data = pd.read_csv(url, compression='gzip', header=0, sep=';', quotechar='"')
    data['date'] = [
        datetime.strptime(str(x), '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00')
        for x in data['AAAAMMJJHH']
    ]
    for x in list(dict.fromkeys(data['NOM_USUEL'].values)):
        print(x)

    STATION = 'LA ROCHELLE-ILE DE RE'
    df_input = data[data['NOM_USUEL'] == STATION]
    df_input = df_input[['date', 'FF2', 'DD2', 'FXI', 'DXI', 'LAT', 'LON', 'NOM_USUEL']]
    df_input.index = pd.to_datetime(df_input.date)
    df_input['month'] = df_input.index.month
    df_input = df_input[(df_input['month'] >= 10) & (df_input['month'] <= 12)]

    file_byte = meteo_test.gen_windrose_from_dataframe(
        df=df_input,
        column_wind_speed='FXI',
        column_wind_direction='DXI',
        show=True,
        write=True,
        station=STATION,
        url=url,
    )
