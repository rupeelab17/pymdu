import numpy as np
import pandas as pd
from datetime import datetime
from pvlib.solarposition import get_solarposition
from pvlib import irradiance
from pymdu.meteo import Meteo

class EPWConverter:

    def __init__(self):
        # You can initialize any attributes here if needed
        pass

    @staticmethod
    def calculate_HorzIRSky(df):
        """
        Calculates the horizontal infrared radiation intensity (field 21 in EPW).
        If a value is present, it is converted to Wh/m², otherwise, the value is calculated
        from the following fields:
          - Dry Air Temperature (T) in °C (tens of degrees not included)
          - Dew Point Temperature (TD) in °C (tens of degrees not included)
          - Opaque Sky Cover (N in tenths)

        Output unit is Wh/m².
        """
        J_CM2_TO_WM2_HOURLY = 1e4 / 3600

        # Use provided INFRAR values if present, otherwise calculate it.
        from_infrar = df['INFRAR'] * J_CM2_TO_WM2_HOURLY

        # Conversion of temperatures to Kelvin
        T_air_K = df["T"] + 273.15
        T_dp_K = df["TD"] + 273.15

        # Stefan-Boltzmann constant
        sigma = 5.67e-8  # [W/m²/K⁴]

        # Emissivity of the sky: ε = 0.787 + 0.764 * ln(T_dp / 273.15)
        with np.errstate(divide='ignore', invalid='ignore'):
            epsilon_sky = 0.787 + 0.764 * np.log(T_dp_K / 273.15)
            epsilon_sky = epsilon_sky.clip(lower=0.1, upper=1.0)  # Limit emissivity range
            epsilon_sky = epsilon_sky.fillna(0.7)  # Default value if NaN

        # Cloud cover calculation
        if "N" in df.columns:
            N_cld = EPWConverter.okta_to_tenths(df["N"])  # Assuming `N` is in Okta
        else:
            N_cld = 0  # If not available, default to no cloud cover.

        # Cloud correction factor based on N_cld
        cloud_correction = 1.0 + 0.0224 * N_cld - 0.0035 * N_cld ** 2 + 0.00028 * N_cld ** 3

        # Horizontal Infrared Radiation calculation (W/m²)
        I_ir = epsilon_sky * cloud_correction * sigma * T_air_K ** 4

        # Use INFRAR where available, otherwise use the calculated value
        result = from_infrar.where(~df['INFRAR'].isna(), I_ir)

        return result

    @staticmethod
    def calculate_dhi_dif (df: pd.DataFrame):
      J_CM2_TO_WM2_HOURLY = 1e4 / 3600  # Conversion factor: 1 J/cm² = 2.77778 Wh/m²

      df['datetime'] = pd.to_datetime(df['AAAAMMJJHH'], format='%Y%m%d%H')
      df = df.set_index("datetime")
      if not isinstance(df.index, pd.DatetimeIndex):
              raise TypeError("The index is not a DatetimeIndex. Please ensure 'datetime' is properly parsed.")

      df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
      df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
      df['ALTI'] = pd.to_numeric(df['ALTI'], errors='coerce')
      df['PSTAT'] = pd.to_numeric(df['PSTAT'], errors='coerce')  # Pressure in Pa
      df['T'] = pd.to_numeric(df['T'], errors='coerce')  # Temperature in Celsius
      df['TD'] = pd.to_numeric(df['TD'], errors='coerce')  # Temperature in Celsius
      df['GLO'] = pd.to_numeric(df['GLO'], errors='coerce')  #


      # Extract location info
      latitude = df.iloc[0]['LAT']
      longitude = df.iloc[0]['LON']
      altitude = df.iloc[0]['ALTI']

          # Ensure that PSTAT and T are Series aligned with the DatetimeIndex
      pressure_series = df['PSTAT'] * 100  # Convert PSTAT to Pa (if it's in hPa)
      temperature_series = df['T']  # Temperature in Celsius
      ghi_series = df['GLO']* J_CM2_TO_WM2_HOURLY
      # Check that these Series are correctly aligned with the DatetimeIndex
      # print(f"GHI Series:\n{ghi_series.head()}")
      # print(f"Temperature Series:\n{temperature_series.head()}")


      solpos = get_solarposition(df.index - pd.Timedelta(minutes=30), latitude, longitude, altitude, pressure=pressure_series, temperature=temperature_series) #shift(freq="-30T") df['PSTAT']*100, df['T']
      solpos.index = df.index

      dni_dirint = irradiance.dirint(ghi_series, solpos.zenith, df.index, pressure=pressure_series, temp_dew=df['TD'])

      # # use "complete sum" AKA "closure" equation: DHI = GHI - DNI * cos(zenith)
      df_dirint = irradiance.complete_irradiance(solar_zenith = solpos.apparent_zenith, ghi = ghi_series, dni = dni_dirint, dhi = None)

      dni_dirint = dni_dirint.fillna(0)
      df_dirint['dhi'] = df_dirint['dhi'].fillna(0)

      # chosen_day = (df.index.month == 8) & (df.index.day == 21)
      # if chosen_day.any():
      #   print(dni_dirint[chosen_day], df_dirint.dhi[chosen_day])

      return dni_dirint, df_dirint.dhi


    def is_leap_year(self,year):
      return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

    # Function to extract the first weekday of the year
    def get_first_weekday_of_year(self,year):
      first_day = datetime(year, 1, 1)
      return first_day.strftime('%a')

    def generate_epw_header(self, data: pd.DataFrame):


      city_name = data['NOM_USUEL'][0]
      latitude = data['LAT'][0]
      longitude = data['LON'][0]
      elevation = data['ALTI'][0]

      # Extract the year from the first date in the dataset
      first_date = pd.to_datetime(data['AAAAMMJJHH'][0], format='%Y%m%d%H')
      year = first_date.year

      # Set placeholders for the other required fields (you can update these values)
      ground_temps = "n/a"  # You can replace this with actual data if available
      time_zone = "UTC"  # Update with correct time zone if available

      # Generate the EPW header lines
      epw_header = [
          f"LOCATION,{city_name},,,MeteoFrance,n/a,{latitude:.2f},{longitude:.2f},{time_zone},{elevation}",
          "DESIGN CONDITIONS,0",
          "TYPICAL/EXTREME PERIODS,0",
          f"GROUND TEMPERATURES,{ground_temps}",
          f"HOLIDAYS/DAYLIGHT SAVINGS,{'Yes' if self.is_leap_year(year) else 'No'},0,0,0",
          "COMMENTS 1,Data from MeteoFrance",
          "COMMENTS 2,PyMDU",
          f"DATA PERIODS,1,1,Data,{self.get_first_weekday_of_year(year)},1/1,12/31",
      ]

      return epw_header

    @staticmethod
    def okta_to_tenths(oktas):
        """
        Converts Okta (0-8 octas) to tenths (0-10) for EPW format.
        Okta 9 is treated as missing/obscured in EPW.
        """
        # Convert Okta to tenths
        tenths = (oktas / 8.0) * 10
        tenths = np.clip(tenths, 0, 10)  # Ensure value is between 0 and 10

        # Handle missing data (Okta 9)
        tenths = np.where(oktas == 9, 99, tenths)  # 99 represents missing data in EPW format


        return np.rint(tenths).astype(int)  # Return as integer tenths

    def create_epw_from_meteo(self, df: pd.DataFrame):
        column_map = {
            # Obligatory fields
            "T": "drybulb",  # Air Temperature (°C)
            "TD": "dewpoint",  # Dew Point Temperature (°C)
            "U": "relhum",  # Relative Humidity (%)
            "PSTAT": "atmos_pressure",  # Atmospheric Pressure (Pa)
            "INFRAR": "horirsky",  # Infrared Radiation (Wh/m²)
            "DIR": "dirnorrad",  # Direct Normal Radiation (Wh/m²)
            "DIF": "difhorrad",  # Diffuse Horizontal Radiation (Wh/m²)
            "DD": "winddir",  # Wind Direction (°)
            "FF": "windspd",  # Wind Speed (m/s)
            "HNEIGEFI1": "snowdepth",  # Snow Depth (cm)
            "RR1": "liq_precip_depth",  # Liquid Precipitation Depth (mm)
            "GLO": "glohorrad",  # Global Horizontal Radiation (Wh/m²)
            "N": "totskycvr",  # Total Sky Cover (tenths)
            "NBAS": "opaqskycvr",  # Opaque Sky Cover (tenths)
            "VV": "visibility",  # Visibility (km)
        }

        # Constants
        J_CM2_TO_WM2_HOURLY = 1e4 / 3600  # Conversion factor: 1 J/cm² = 2.77778 Wh/m²

        # Initialize the EPW DataFrame
        epw_df = pd.DataFrame(index=df.index)


        # Conversion of columns based on column_map
        for src_col, epw_col in column_map.items():
            if src_col in df.columns:
                if src_col in ["T", "TD"]:  # Temperature in Celsius
                    epw_df[epw_col] = df[src_col].clip(lower=-70, upper=70).fillna(99.9).round(1)
                elif src_col == "U":  # Relative Humidity (%)
                    epw_df[epw_col] = df[src_col].clip(lower=0, upper=100).fillna(999.)
                elif src_col == "PSTAT":  # Atmospheric Pressure (Pa)
                    epw_df[epw_col] = df[src_col].mul(100).clip(lower=31000, upper=120000).fillna(999999.) # hPa to Pa
                elif src_col == "INFRAR":  # Infrared Radiation
                    epw_df[epw_col] = np.round(self.calculate_HorzIRSky(df), 1).fillna(9999.).clip(lower=0)
                elif src_col in ["GLO"]:  # Radiation conversion J/cm² to Wh/m² #, "DIF", "DIR"
                    epw_df[epw_col] = np.round(df[src_col] * J_CM2_TO_WM2_HOURLY, 1)
                elif src_col == "FF":  # Wind Speed (m/s)
                    epw_df[epw_col] = df[src_col].clip(lower=0, upper=40).fillna(999.)
                elif src_col == "DD":  # Wind Direction (°)
                    epw_df[epw_col] = df[src_col].clip(lower=0, upper=360).fillna(999.)
                elif src_col == "HNEIGEFI1":  # Snow Depth (cm)
                    epw_df[epw_col] = np.round(df[src_col], 0).fillna(999).clip(lower=0)
                elif src_col == "RR1":  # Precipitation (mm)
                    epw_df[epw_col] = np.round(df[src_col], 0).fillna(999)
                elif src_col == 'NBAS':  # Opaque Sky Cover
                    epw_df[epw_col] = epw_df[epw_col] = df[src_col].apply(lambda x: self.okta_to_tenths(x) if pd.notna(x) else 99)
                elif src_col == 'N':  # Total Sky Cover
                    epw_df[epw_col] = epw_df[epw_col] = df[src_col].apply(lambda x: self.okta_to_tenths(x) if pd.notna(x) else 99)
                elif src_col == "VV":  # Visibility (km)
                    epw_df[epw_col] = (df[src_col] / 1000).round(1).fillna(9999).clip(lower=0)

        # Optional fields with missing data
        dict_missing = {
            "exthorrad": 9999,
            "extdirrad": 9999,
            # "glohorrad": 9999,
            "glohorillum": 999999,
            "dirnorillum": 999999,
            "difhorillum": 999999,
            "zenlum": 9999,
            "ceiling_hgt": 99999,
            "precip_wtr": 999,
            "aerosol_opt_depth": 999,
            "days_last_snow": 99,
            "albedo": 999,
            "liq_precip_rate": 99,
            "liq_precip_quantity" : 1
        }

        # Ensure missing columns are populated with default values
        for col in dict_missing.keys():
            if col not in epw_df.columns:
                epw_df[col] = dict_missing[col]

        # Convert 'AAAAMMJJHH' to a datetime object
        epw_df['Datetime'] = pd.to_datetime(df['AAAAMMJJHH'], format='%Y%m%d%H')

        # Create columns for EPW format: Year, Month, Day, Hour, etc.
        epw_df['year'] = epw_df['Datetime'].dt.year
        epw_df['month'] = epw_df['Datetime'].dt.month
        epw_df['day'] = epw_df['Datetime'].dt.day
        epw_df['hour'] = epw_df['Datetime'].dt.hour + 1
        epw_df['minute'] = 0  # Minute is always 0 in this case (hourly data)
        epw_df['datasource'] = '?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9*9?9?9?9'
        epw_df['presweathobs'] = 0 # 0 = Weather observation made; 9 = Weather observation not made, or missing
        epw_df['presweathcodes'] = 999999999

        dni_dirint, dhi_dirint = self.calculate_dhi_dif(df)

        # Fill NaN with zeros before inserting
        epw_df['dirnorrad'] = dni_dirint.values
        epw_df['difhorrad'] = dhi_dirint.values
        # print(self.calculate_dhi_dif(df))

        # EPW column order
        epw_order = ["year", "month", "day", "hour", "minute", "datasource", "drybulb", "dewpoint", "relhum",
                    "atmos_pressure", "exthorrad", "extdirrad", "horirsky", "glohorrad", "dirnorrad", "difhorrad",
                    "glohorillum", "dirnorillum", "difhorillum", "zenlum", "winddir", "windspd", "totskycvr", "opaqskycvr",
                    "visibility", "ceiling_hgt", "presweathobs", "presweathcodes", "precip_wtr", "aerosol_opt_depth",
                    "snowdepth", "days_last_snow", "albedo", "liq_precip_rate", "liq_precip_quantity"]

        # Rearrange columns to match EPW format
        epw_df = epw_df[epw_order]
        chosen_day = (epw_df['month'] == 3) & ( epw_df['day']  == 31)
        if chosen_day.any():
            print(df.N[chosen_day], epw_df.totskycvr[chosen_day], df.NBAS[chosen_day], epw_df.opaqskycvr[chosen_day])

        return epw_df

    def convert_csv_to_epw(self, input_csv_path: str, output_epw_path: str, delimiter=';'):
        # Load the input CSV file
        df = pd.read_csv(input_csv_path, delimiter = delimiter)

        # Convert to EPW format
        epw_data = self.create_epw_from_meteo(df)

        # print(epw_data["difhorrad"])

        # Write the header to the output EPW file
        with open(output_epw_path, "w") as f:
          for line in self.generate_epw_header(df):
              f.write(line + "\n")

        # Write to EPW file
        epw_data.to_csv(output_epw_path, mode='a', header=False, index=False)

        print(f"EPW file saved at {output_epw_path}")


# Example usage:
if __name__ == "__main__":
    meteo_test = Meteo(output_path=r"./")
    meteo_test.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    # meteo_test.run(
    #     begin="2017-06-30 00:00:00",
    #     end="2018-06-30 23:00:00",
    #     weather_filename="FRA_AC_La.Rochelle.Intl.AP.073160_TMYx_UWG.epw",
    # )
    chosen_year = 2022
    url = meteo_test.find_url_meteo_france(year=chosen_year, dep=17)
    print('meteo_test', url)
    data = pd.read_csv(url, compression="gzip", header=0, sep=";", quotechar='"')
    data['datetime'] = pd.to_datetime(data['AAAAMMJJHH'], format='%Y%m%d%H')


    STATION = "LA ROCHELLE-ILE DE RE"
    df_input = data[(data["NOM_USUEL"] == STATION) & (data["datetime"].dt.year == chosen_year)]
    df_input = df_input.drop(columns=["datetime"])
    df_input.to_csv(f'{STATION}_{chosen_year}.csv')
    # Initialize the EPWConverter class
    converter = EPWConverter()

    # Define your input CSV file path and output EPW file path
    input_csv_path = f'{STATION}_{chosen_year}.csv'  # Path to your input CSV file
    output_epw_path = f'{STATION}_{chosen_year}.epw'  # Desired output EPW file path

    # Convert the CSV to EPW format
    converter.convert_csv_to_epw(input_csv_path, output_epw_path, delimiter=',')
