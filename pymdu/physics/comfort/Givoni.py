from pymdu.GeoCore import GeoCore
import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)


class Givoni(GeoCore):
    def __init__(
        self,
        input_path=r'E:\RESULTATS_ATLANTECH\ATLANTECH_0\UMEP',
        weather_file=r'T:\FR_MDU/LaRochelle_historical_IPSL_bc_type_list_UMEP.txt',
        temperature_filed='Dry Bulb Temperature',
        humidity_field='Relative Humidity',
    ):
        self.weather_file = weather_file
        self.input_path = input_path
        self.temperature_filed = temperature_filed
        self.humidity_field = humidity_field
