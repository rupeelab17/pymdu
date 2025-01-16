import glob
import os
import shutil
from datetime import datetime

import imageio
import netCDF4
import numpy as np
import pandas as pd
import pythermalcomfort
import rasterio
import rioxarray as rxr
import xarray as xr
from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.image.geotiff import reproject_resample_cropped_raster
from pymdu.physics.comfort.utils import trouver_nombre_plus_proche
from rasterio.transform import from_origin
from tqdm import tqdm


class UTCI(GeoCore):
    def __init__(
        self,
        output_path: os.path,
        tmrt_path: os.path,
        wind_path: os.path,
        shadow_path: os.path,
        hierarachie_path_urock: str = 'wind_direction*/z1_5/*TWS.GTiff',
        video: bool = True,
    ) -> None:
        self.output_path = output_path if output_path else TEMP_PATH
        self.tmr_path = tmrt_path
        self.wind_path = wind_path
        self.shadow_path = shadow_path
        self.video = video
        self.wind_path_output = os.path.join(output_path, 'data')
        if os.path.exists(self.wind_path_output):
            shutil.rmtree(self.wind_path_output, ignore_errors=True)
        os.mkdir(self.wind_path_output)

        self.save_path_dataset = os.path.join(self.output_path, 'datasets')
        if os.path.exists(self.save_path_dataset):
            shutil.rmtree(self.save_path_dataset, ignore_errors=True)
        os.mkdir(self.save_path_dataset)

        self.wind_final_path = os.path.join(self.output_path, 'WindFinal')
        if os.path.exists(self.wind_final_path):
            shutil.rmtree(self.wind_final_path, ignore_errors=True)
        os.mkdir(self.wind_final_path)

        self.all_wind_tif_files = glob.glob(
            os.path.join(self.wind_path, hierarachie_path_urock)
        )

    def __traitement_urock(self):
        model_file = glob.glob(os.path.join(self.tmr_path, 'Tmr*.tif'))[0]
        for input_file in self.all_wind_tif_files:
            direction = input_file.split('/')[-1].split('_')[1]
            print(direction)
            output_file = os.path.join(self.wind_path_output, f'{direction}.tif')
            reproject_resample_cropped_raster(
                model_file=model_file, src_tif=input_file, dst_tif=output_file
            )

    def __traitement_meteo(
        self,
        file_path_meteo: os.path = 'meteo/FRA_AC_La.Rochelle.073150_TMYx.2004-2018.txt',
    ):
        METEO_PATH = os.path.join(self.output_path, file_path_meteo)
        self.meteo = pd.read_csv(METEO_PATH, sep=' ')
        year = '2022'
        liste_of_date = [
            datetime.strptime(year + '-' + str(day), '%Y-%j')
            for day in self.meteo['id']
        ]
        date = [
            res.strftime(f'%Y-%m-%d {hour}:00:00')
            for (res, hour) in zip(liste_of_date, self.meteo['it'])
        ]
        date = [pd.to_datetime(x) for x in date]

        liste_correspondances_tif = []
        all_files = glob.glob(self.wind_path_output + '/*')
        print(all_files)
        liste_number_tif = [
            int(x.split(os.sep)[-1].split('.tif')[0]) for x in all_files
        ]
        for direction in self.meteo['Wd']:
            numero = trouver_nombre_plus_proche(liste_number_tif, int(direction))
            liste_correspondances_tif.append(
                os.path.join(self.wind_path_output, f'{numero}.tif')
            )
        print(liste_correspondances_tif)
        self.meteo['tif'] = liste_correspondances_tif
        self.meteo[['Wind', 'Wd', 'tif']].head()
        self.meteo.index = date

        for file, speed, annee, jour, heure in zip(
            self.meteo['tif'],
            self.meteo['Wind'],
            self.meteo['%iy'],
            self.meteo['id'],
            self.meteo['it'],
        ):
            my_data = rxr.open_rasterio(file)
            # attention ici, il faut que la simulation ait été fait pour V = 1 m/s
            my_data.values = my_data.values * (speed)
            heure_modif = '{:02d}'.format(heure)
            save_file = os.path.join(
                self.wind_final_path, f'Wind_{annee}_{jour}_{heure_modif}60.tif'
            )
            print(save_file)
            my_data.rio.to_raster(raster_path=save_file)

    def __construction_dataset(
        self,
    ):
        self.liste_Shadow = glob.glob(os.path.join(self.shadow_path, 'Shadow*.tif'))
        self.liste_Tmrt = glob.glob(os.path.join(self.tmr_path, 'Tmrt*.tif'))
        self.liste_Wind = glob.glob(os.path.join(self.wind_final_path, 'Wind*.tif'))

        self.dictionnaire_inputs = [
            {'variable': 'Shadow', 'paths': self.liste_Shadow},
            {'variable': 'Tmrt', 'paths': self.liste_Tmrt},
            {'variable': 'Wind', 'paths': self.liste_Wind},
        ]

        # Construction de datasets
        model_file = glob.glob(os.path.join(self.tmr_path, 'Tmr*.tif'))[0]
        model_file = rxr.open_rasterio(model_file)

        for my_input in self.dictionnaire_inputs:
            filenames = my_input['paths']
            name_variable = my_input['variable']
            # suppression du .tif "average" > TODO : empêcher qu'il soit généré

            self.time = xr.Variable(
                'time', self.__time_index_from_filenames(filenames, name=name_variable)
            )

            encoding = {
                'y': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
                'x': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
                'time': {'zlib': True, 'complevel': 9},
                f'{name_variable}': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
            }
            liste_variable = []
            for file in tqdm(filenames):
                if name_variable == 'Wind':
                    liste_variable.append(
                        rxr.open_rasterio(file)
                        .interp(
                            x=model_file.x.values,
                            y=model_file.y.values,
                            method='nearest',
                        )
                        .chunk(chunks={'x': 100, 'y': 100})
                    )
                else:
                    liste_variable.append(
                        rxr.open_rasterio(file)
                        .interp(
                            x=model_file.x.values,
                            y=model_file.y.values,
                            method='nearest',
                        )
                        .chunk(chunks={'x': 100, 'y': 100})
                    )

            da = xr.concat(liste_variable, dim=self.time, coords='minimal').chunk(
                {'time': 50}
            )
            print(f'{name_variable} = {da}')
            da = da.drop_vars('band')
            da = da.rename(name_variable)
            # da = da.rename({'band': name_variable})
            print(f'{name_variable} = {da}')

            if os.path.exists(
                os.path.join(self.save_path_dataset, f'{name_variable}.nc')
            ):
                os.remove(os.path.join(self.save_path_dataset, f'{name_variable}.nc'))

            if name_variable == 'Wind':
                self.da_Wind = da
                # enregistrement
                print(f'{name_variable} en cours de sauvegarde')

            elif name_variable == 'Tmrt':
                self.da_Tmrt = da
                # enregistrement
                print(f'{name_variable} en cours de sauvegarde')
            elif name_variable == 'Shadow':
                self.da_Shadow = da
                # enregistrement
                print(f'{name_variable} en cours de sauvegarde')

            da.load().to_netcdf(
                os.path.join(self.save_path_dataset, f'{name_variable}.nc'),
                engine='netcdf4',
                encoding=encoding,
            )
            da.close()

        liste_dataset = [self.da_Wind, self.da_Tmrt, self.da_Shadow]
        self.final = xr.merge(liste_dataset)

    def __calculate_utci(self):
        utci_data = []
        for idx, tdb, rh in tqdm(
            zip(self.meteo.index, self.meteo['Td'], self.meteo['RH'])
        ):
            confort = self.__calculate_confort_from_solweig_and_urock_dataarray(
                dataset=self.final.sel(time=idx.replace(year=2022)), tdb=tdb, rh=rh
            )

            utci = self.final['Tmrt'][0, 0, :, :].copy()
            utci = utci.rename('Utci')
            utci.values = confort
            utci_data.append(utci.chunk(chunks={'x': 100, 'y': 100}))

        da_Utci = xr.concat(utci_data, dim=self.time, coords='minimal').chunk(
            {'time': 50}
        )

        encoding = {
            'y': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
            'x': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
            'time': {'zlib': True, 'complevel': 9},
            'Utci': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        }

        da_Utci.load().to_netcdf(
            os.path.join(self.save_path_dataset, 'Utci.nc'),
            engine='netcdf4',
            encoding=encoding,
        )

        self.convert_netcdf_to_geotiff(
            input_nc=os.path.join(self.save_path_dataset, 'Utci.nc'), variable='Utci'
        )

        print(self.da_Shadow)
        merge_xr = xr.merge([da_Utci, self.da_Shadow])
        data_utci_soleil = merge_xr.where(merge_xr.Shadow > 0.5)
        print(data_utci_soleil)
        data_utci_soleil = data_utci_soleil.squeeze('band')

        data_utci_soleil.load().to_netcdf(
            os.path.join(self.save_path_dataset, 'SunUtci.nc'),
            engine='netcdf4',
            encoding=encoding,
        )

        liste_dataset = [
            self.da_Wind,
            self.da_Tmrt,
            self.da_Shadow,
            da_Utci,
            data_utci_soleil,
        ]
        final = xr.merge(liste_dataset)
        final = final.squeeze('band')

        ########################################################################################################################
        #                                       UTCI SHADOWS
        ########################################################################################################################
        utci_shadow_liste = []
        for x in tqdm(range(len(final['Utci']))):
            utci_shadows = final['Utci'][x].copy()
            utci_shadows = utci_shadows.rename('Utci_shadow')
            # l'ombre (1 à l'ombre)
            arr = np.array(final['Shadow'][x].data.copy())
            where_0 = np.where(arr < 0.5)
            where_1 = np.where(arr == 1)
            arr[where_0] = 0
            arr[where_1] = 1
            utci_shadows.data = np.ma.array(
                data=utci_shadows.data, mask=arr, fill_value=np.nan
            )
            utci_shadow_liste.append(utci_shadows.chunk(chunks={'x': 100, 'y': 100}))

        utci_shadow_dataset = xr.concat(
            [x for x in utci_shadow_liste], dim=final.time
        ).chunk({'time': 50})

        encoding = {
            'y': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
            'x': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
            'time': {'zlib': True, 'complevel': 9},
            'Utci_shadow': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        }

        print('Utci_shadow en cours de sauvegarde')

        utci_shadow_dataset.load().to_netcdf(
            os.path.join(self.save_path_dataset, 'ShdUtci.nc'), encoding=encoding
        )

    def convert_netcdf_to_geotiff_old(self, input_nc, variable):
        # Open the NetCDF file
        nc = netCDF4.Dataset(input_nc)
        # Extract data and metadata from the NetCDF file
        data = nc.variables[variable][
            :
        ]  # Replace 'your_variable' with the actual variable name
        lat = nc.variables['y'][:]
        lon = nc.variables['x'][:]
        time = nc.variables['time'][:]

        print(time)
        # Define the GeoTIFF metadata
        transform = from_origin(lon.min(), lat.max(), lon[1] - lon[0], lat[0] - lat[1])
        height, width = data.shape[1], data.shape[2]

        profile = {
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'float32',  # Adjust data type based on your NetCDF data type
            'width': width,
            'height': height,
            'crs': 'EPSG:2154',  # Adjust the CRS as needed
            'transform': transform,
        }

        colormap = {
            29: (48, 18, 59),  # Red for pixel value 1
            2: (0, 255, 0),  # Green for pixel value 2
            3: (0, 0, 255),  # Blue for pixel value 3
            # Add more entries as needed
        }

        for i in list(map(str, time)):
            # Create GeoTIFF file and write data
            with rasterio.open(
                os.path.join(self.save_path_dataset, f'{variable}_{i}.tif'),
                'w',
                **profile,
            ) as dst:
                dst.write(data[int(i), :, :], 1)
                # # Read the color map from the .clr file
                # with open("/Users/Boris/Downloads/style.txt", 'r') as clr_file:
                #     # Parse the color map from the file
                #     colormap = [tuple(map(float, line.split(','))) for line in clr_file]
                #
                # print(colormap)
                # # Apply the color map to the raster data
                # color_mapped_data = [colormap[val] for val in data.flatten()]
                # color_mapped_data = [tuple(float(value) for value in color) for color in color_mapped_data]
                #
                # # Reshape the color-mapped data back to the original shape
                # color_mapped_data = [color_mapped_data[i:i + width] for i in
                #                      range(0, len(color_mapped_data), width)]
                # print(color_mapped_data)
                # Set the colormap in the new file
                dst.write_colormap(
                    1, colormap
                )  # windows = [window for ij, window in dst.block_windows()]  # We cannot write to the same file from multiple threads  # without causing race conditions. To safely read/write  # from multiple threads, we use a lock to protect the  # Writer  # write_lock = threading.Lock()

                # def process(window):  #     with write_lock:  # dst.write(data[int(i), :, :], 1)  # # Read the color map from the .clr file  # with open("/Users/Boris/Downloads/style.txt", 'r') as clr_file:  #     # Parse the color map from the file  #     colormap = [tuple(map(int, line.split())) for line in clr_file]  #  # # Apply the color map to the raster data  # color_mapped_data = [colormap[val] for val in data.flatten()]  # color_mapped_data = [tuple(int(value) for value in color) for color in color_mapped_data]  #  # # Reshape the color-mapped data back to the original shape  # color_mapped_data = [color_mapped_data[i:i + width] for i in  #                      range(0, len(color_mapped_data), width)]  # print(color_mapped_data)  # # Set the colormap in the new file  # dst.write_colormap(1, color_mapped_data)

                # We map the process() function over the list of  # windows.  # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  #     executor.map(process, windows)

        if self.video:
            self.create_video(
                image_folder=os.path.join(self.save_path_dataset, f'{variable}_*.tif'),
                output_path=os.path.join(self.save_path_dataset, f'{variable}.mp4'),
            )

    def convert_netcdf_to_geotiff(self, input_nc, variable):
        xds = xr.open_dataset(input_nc)
        time = xds.time.values
        # Formater chaque chaîne de caractères dans la liste
        formatted_datetime_list = [
            datetime.strptime(
                np.datetime_as_string(numpy_datetime), '%Y-%m-%dT%H:%M:%S.000000000'
            ).strftime('%Y-%m-%d-%Hh')
            for numpy_datetime in time
        ]

        for i, t in enumerate(formatted_datetime_list):
            xds.isel(time=i).rio.to_raster(
                os.path.join(self.save_path_dataset, f'{variable}_{t}.tif'),
                tiled=False,
                windowed=False,
            )

        if self.video:
            self.create_video(
                image_folder=os.path.join(self.save_path_dataset, f'{variable}_*.tif'),
                output_path=os.path.join(self.save_path_dataset, f'{variable}.mp4'),
            )

    def create_video(self, image_folder, output_path='output.mp4', fps=1):
        images = []

        # Load all GeoTIFF files from the specified folder
        for filename in sorted(glob.glob(image_folder)):
            if filename.endswith('.tif'):
                images.append(imageio.imread(filename))
        # imageio.mimsave(output_path, images, fps=fps, bigtiff=True)
        # Create the video
        video_writer = imageio.get_writer(output_path, fps=fps)
        for image in images:
            video_writer.append_data(image)
        video_writer.close()

    def run(self):
        self.__traitement_urock()
        self.__traitement_meteo()
        self.__construction_dataset()
        self.__calculate_utci()

        pass

    @staticmethod
    def __calculate_confort_from_solweig_and_urock_dataarray(dataset, tdb=25, rh=50):
        tmr_as_array = dataset['Tmrt'].data[0]
        wind_as_array = dataset['Wind'].data[0]
        size1, size2 = tmr_as_array.shape
        output = np.zeros(shape=(size1, size2))
        for i in range(0, size1):
            output[i, :] = pythermalcomfort.utci(
                tdb=tdb,
                tr=tmr_as_array[i, :],
                v=wind_as_array[i, :],
                rh=rh,
                limit_inputs=False,
            )

        return output

    @staticmethod
    def __time_index_from_filenames(filenames, name='Tmrt'):
        print(filenames)

        return [
            datetime.strptime(
                date.split(os.sep)[-1].replace('N', '').replace('D', ''),
                f'{name}_%Y_%j_%H60.tif',
            )
            for date in filenames
        ]


if __name__ == '__main__':
    test = UTCI(
        output_path='./Ressources',
        tmrt_path='./Ressources',
        wind_path='./Ressources',
        shadow_path='./Ressources',
        hierarachie_path_urock='wind_direction*/z1_5/*TWS.GTiff',
    )
    test.run()
