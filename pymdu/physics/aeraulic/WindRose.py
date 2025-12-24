import numpy as np
import pandas as pd
import plotly.express as px

from pymdu.meteo.Meteo import Meteo


class WindRose:
    """
    ===
    Classe qui permet
    - d'analyser les données de vent d'un fichier .epw
    ===
    """

    def __init__(
        self,
        epw_path: str = r'C:\Users\simon\python-scripts\pymdu\pymdu\meteo/FRA_AC_La.Rochelle.Intl.AP.073160_TMYx_UWG.epw',
        *args,
        **kwargs,
    ):
        self.epw_path = epw_path
        meteo_object = Meteo()
        self.weather = pd.read_csv(
            self.epw_path, names=meteo_object.epw_columns, skiprows=8
        )
        self.weather.index = pd.date_range(
            start='2022-01-01', end='2022-12-31 23:00:00', freq='h'
        )

        dat_txt = open(self.epw_path, 'r')
        myList = dat_txt.readlines()
        dat_txt.close()
        self.head = myList[:7]

        self.title = self.head[0].split(',')[1]

    def degToCompass(self, num):
        val = int((num / 22.5) + 0.5)
        arr = [
            'N',
            'NNE',
            'NE',
            'ENE',
            'E',
            'ESE',
            'SE',
            'SSE',
            'S',
            'SSW',
            'SW',
            'WSW',
            'W',
            'WNW',
            'NW',
            'NNW',
        ]
        return arr[(val % 16)]

    def borne(
        serie,
        bornesmoins=[0, 1, 2, 3, 4, 5],
        bornesplus=[1, 2, 3, 4, 5, 6],
        cat=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6'],
    ):
        liste = []
        for val in serie.values:
            for k in range(6):
                toAdd = '6+'
                if bornesmoins[k] <= val < bornesplus[k]:
                    toAdd = cat[k]
                    break
            liste.append(toAdd)
        return liste

    def borne_adapt(self, my_range=2):
        bornesmoins = list(
            range(0, int(self.weather['Wind Speed'].max()) + 2, my_range)
        )
        bornesplus = list(
            range(0, int(self.weather['Wind Speed'].max()) + 2, my_range)
        )[1:]
        bornesplus.append(
            list(range(0, int(self.weather['Wind Speed'].max()) + 2, my_range))[-1] + 2
        )
        liste = []
        for val in self.weather['Wind Speed'].values:
            for k in range(len(bornesmoins)):
                if bornesmoins[k] <= val < bornesplus[k]:
                    toAdd = str(bornesmoins[k]) + '-' + str(bornesplus[k])
                    break
            liste.append(toAdd)
        return liste

    def corresp_bornes(self, my_range=2):
        bornesmoins = list(
            range(0, int(self.weather['Wind Speed'].max()) + 2, my_range)
        )
        bornesplus = list(
            range(0, int(self.weather['Wind Speed'].max()) + 2, my_range)
        )[1:]
        bornesplus.append(
            list(range(0, int(self.weather['Wind Speed'].max()) + 2, my_range))[-1] + 2
        )
        dico = {}
        for k in range(len(bornesmoins)):
            toAdd = str(bornesmoins[k]) + '-' + str(bornesplus[k])
            dico[toAdd] = k * 100
        return dico

    def wind_analysis(self, path_to_save='test.csv', plot=False):
        windSpeed = self.weather['Wind Speed'].values
        windDir = self.weather['Wind Direction'].values
        self.weather['Wind Direction'] = [x for x in windDir]
        self.weather['Wind Speed'] = [x for x in windSpeed]
        self.correspBornes = self.corresp_bornes()

        data = pd.DataFrame()
        data['dir'] = [self.degToCompass(x) for x in self.weather['Wind Direction']]
        data['sens'] = [x for x in self.weather['Wind Direction']]
        data['strenght'] = self.borne_adapt()
        data['name'] = data['dir'] + '_' + data['strenght']
        data['speed'] = self.weather['Wind Speed'].values
        self.data_wind = data

        listeFreq = []
        listeDir = []
        listeStrenght = []
        listeSens = []

        # test amélioration
        for name in list(dict.fromkeys(data['name'])):
            freq = 100 * len(data.loc[(data['name'] == name)]) / len(self.weather)
            listeFreq.append(round(freq, 3))
            listeDir.append(name.split('_')[0])
            listeStrenght.append(name.split('_')[1])
        df = pd.DataFrame()

        df['frequency1'] = listeFreq
        df['direction1'] = listeDir
        df['m/s'] = listeStrenght
        df['name1'] = df['direction1'] + '_' + df['m/s']
        order = [int(x[0][0]) for x in df['m/s']]
        df['order1'] = order
        # df1 = df1.sort_values(by=['order1'])
        # df.index = df1['name1']
        # result = pd.concat([df, df1], axis=1)
        # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"
        # weather = weather.dropna()
        listDir = [
            'N',
            'NNE',
            'NE',
            'ENE',
            'E',
            'ESE',
            'SE',
            'SSE',
            'S',
            'SSW',
            'SW',
            'WSW',
            'W',
            'WNW',
            'NW',
            'NNW',
        ]
        directionVentcorresp = {}
        for k, d in enumerate(listDir):
            directionVentcorresp[d] = k

        df['1'] = [directionVentcorresp[x] for x in df['direction1']]
        df['2'] = [self.correspBornes[x] for x in df['m/s']]
        df['order'] = df['1'] + df['2']
        df = df.sort_values(by=['order'])

        if plot == True:
            fig = px.bar_polar(
                df,
                r='frequency1',
                theta='direction1',
                color='m/s',
                template='none',
                color_discrete_sequence=px.colors.sequential.RdBu_r,
                title=self.title,
            )
            #     fig.write_image("windrose.svg")
            fig.show()
        else:
            pass

        liste_deg = np.arange(0, 360, 22.5)
        tableCorresp = {}
        for x in liste_deg:
            tableCorresp[self.degToCompass(x)] = x

        df['DIR'] = [tableCorresp[x] for x in df['direction1']]
        df['SPEED'] = [
            (float(vitesse.split('-')[0]) + float(vitesse.split('-')[1])) / 2
            for vitesse in df['m/s']
        ]

        # Pour lancer les calculs sur la rose des vents
        wind = df
        list_direction = list(dict.fromkeys(wind['DIR']))

        direction = []
        vitesse = []
        for dir in list_direction:
            df = wind[wind['DIR'] == dir]
            direction.append(int(dir))
            vitesse.append(
                np.sum((df['SPEED'] * df['frequency1']) / (df['frequency1'].sum()))
            )
        urock_calculs = pd.DataFrame()
        urock_calculs['direction'] = direction
        urock_calculs['vitesse'] = vitesse
        urock_calculs.to_csv(path_to_save)


if __name__ == '__main__':
    test = WindRose()
    test.wind_analysis(
        path_to_save=r'C:\Users\simon\python-scripts\POC\windrose_calculation.csv',
        plot=False,
    )
