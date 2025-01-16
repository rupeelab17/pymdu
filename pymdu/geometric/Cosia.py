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
import re
import os

import geopandas as gpd
import pandas as pd
from osgeo import gdal
# from osgeo_utils.samples import ogr2ogr
from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH

from shapely import box, Polygon
from sqlalchemy import create_engine, text
from tqdm import tqdm


# for i in *.gpkg; do ogr2ogr -f "PostgreSQL" "PG:host=10.17.36.50 user=postgres password=postgres dbname=cosia" $i -lco LAUNDER=NO; done
# Name              Code Alb  Emis Ts_deg Tstart TmaxLST
# Roofs(buildings)   2   0.18 0.95 0.58   -9.78  15.0
# Dark_asphalt       1   0.18 0.95 0.58   -9.78  15.0
# Cobble_stone_2014a 0   0.20 0.95 0.37   -3.41  15.0
# Water              7   0.05 0.98 0.00    0.00  12.0
# Grass_unmanaged    5   0.16 0.94 0.21   -3.38  14.0
# bare_soil          6   0.25 0.94 0.33   -3.01  14.0
# Walls             99   0.20 0.90 0.37   -3.41  15.0


class Cosia(GeoCore):
    """
    A class used to collect and process Cosia data.

    This class provides methods for importing GeoPackage files into a PostgreSQL database,
    retrieving geodata from specified departments within a bounding box, creating landcover rasters,
    and overlapping with pedestrian areas. It also includes functionalities for creating trees and
    their positions based on the Cosia data.

    Attributes:
        cosia_keys (dict): A dictionary containing Cosia keys.
        output_path (str): The output path for processed data.
        template_raster_path (str): The default height of each storey.
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the geodata.
    """

    def __init__(self, output_path: str = None, template_raster_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            template_raster_path (str): The default height of each storey.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt

            plt.clf()  # markdown-exec: hide
            cosia = Cosia(output_path='./')
            cosia.bbox = [-1.15643, 46.16123, -1.15127, 46.16378]
            cosia_gdf = cosia.run().to_gdf()
            cosia_gdf.plot(color=cosia_gdf['color'])
            cosia_gdf.plot(ax=plt.gca())
            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
            * For module TODOs
        """
        super().__init__()
        self.cosia_keys = {
            'Bâtiment': 2,
            'Zone imperméable': 1,
            'Zone perméable': 6,
            'Piscine': 7,
            'Serre': 1,
            'Sol nu': 6,
            'Surface eau': 7,
            'Neige': 7,
            'Conifère': 6,
            'Feuillu': 6,
            'Coupe': 5,
            'Broussaille': 5,
            'Pelouse': 5,
            'Culture': 5,
            'Terre labourée': 6,
            'Vigne': 5,
            'Autre': 1,
        }
        self.output_path = output_path if output_path else TEMP_PATH
        self.template_raster_path = template_raster_path
        self.gdf: gpd.GeoDataFrame = None
        self.table_color_cosia = {
            'Bâtiment': '#ce7079',
            'Zone imperméable': '#a6aab7',
            'Zone perméable': '#987752',
            'Piscine': '#62d0ff',
            'Serre': '#b9e2d4',
            'Sol nu': '#bbb096',
            'Surface eau': '#3375a1',
            'Neige': '#e9effe',
            'Conifère': '#216e2e',
            'Feuillu': '#4c9129',
            'Coupe': '#e48e4d',
            'Broussaille': '#b5c335',
            'Pelouse': '#8cd76a',
            'Culture': '#decf55',
            'Terre labourée': '#d0a349',
            'Vigne': '#b08290',
            'Autre': '#222222',
        }

    def __import_gpkg_to_postgres(self, file_gpkg: str):
        pgConnection = (
            'PG:host=10.17.36.50 user=postgres password=postgres dbname=cosia'
        )
        gpkgFile = file_gpkg
        # set GDAL config options
        gdal.SetConfigOption('OGR_TRUNCATE', 'YES')
        gdal.SetConfigOption('PG_USE_COPY', 'YES')
        gdal.SetConfigOption('LAUNDER', 'NO')
        # ogr2ogr.main(
        #     ['', '-append', '-preserve_fid', '-f', 'PostgreSQL', pgConnection, gpkgFile]
        # )

    def run(self, departement: str = '17'):
        self.get_geodata(departement=departement)
        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    @staticmethod
    def __create_square(xmin, ymax, length=10000):
        # Define the coordinates of the square's vertices
        vertices = [
            (xmin, ymax),
            (xmin + length, ymax),
            (xmin + length, ymax - length),
            (xmin, ymax - length),
        ]

        # Create a Shapely Polygon from the vertices
        square = Polygon(vertices)

        return square

    def overlay_cosia_with_pedestrian(self, pedestrian: gpd.GeoDataFrame):
        # TODO : faire un "raiseException"

        cosia_clip_zone_pietonne = gpd.clip(self.gdf, pedestrian)
        cosia_clip_zone_pietonne['classe'] = [
            'Zone imperméable' for x in cosia_clip_zone_pietonne.classe
        ]
        cosia_clip_zone_pietonne['key_umep'] = [
            1 for x in cosia_clip_zone_pietonne.classe
        ]
        difference = self.gdf.overlay(cosia_clip_zone_pietonne, how='difference')

        self.gdf = gpd.GeoDataFrame(
            pd.concat([cosia_clip_zone_pietonne, difference], ignore_index=True)
        )

        return self.gdf

    def get_geodata(
            self,
            departement: str = '17',
            ip_address: str = '127.0.0.1',
            # uri="postgresql+psycopg2://postgres:postgres@10.17.36.50:5432/cosia",
    ):
        gdf_project = gpd.GeoDataFrame(
            gpd.GeoSeries(box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])),
            columns=['geometry'],
            crs='epsg:4326',
        )
        gdf_project = gdf_project.to_crs(epsg=2154)
        envelope_polygon = gdf_project.envelope.bounds
        bbox = envelope_polygon.values[0]
        bbox_final = box(bbox[0], bbox[1], bbox[2], bbox[3])

        uri = f'postgresql+psycopg2://postgres:postgres@{ip_address}:{5432}/cosia'

        # Replace 'your_database_connection_string' with your actual database connection string
        engine = create_engine(uri)

        # SQL query to list all tables
        query = text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE';
        """
        )

        # Execute the query and fetch the results
        result = engine.execute(query)

        # Extract and print the table names
        table_names = [
            row['table_name']
            for row in result
            if re.compile(rf'^D0{departement}+').match(row['table_name'])
        ]

        table_names_filtered = []
        for name_tuile in tqdm(table_names):
            xmin = int(name_tuile.split('_')[2]) * 1000
            ymax = int(name_tuile.split('_')[3]) * 1000
            tuile = gpd.GeoDataFrame(
                index=[0], crs='epsg:2154', geometry=[self.__create_square(xmin, ymax)]
            )
            intersection = gpd.overlay(tuile, gdf_project, how='intersection')
            if not intersection.empty:
                table_names_filtered.append(name_tuile)

        gdf_list = []

        for table in tqdm(table_names_filtered):
            QUERY = f"""
            SELECT classe, ST_AsText(geom), geom, ST_Envelope(geom) AS bbox
            FROM public."{table}"
             WHERE ST_Intersects(geom, ST_MakeEnvelope({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}, 2154))
            """
            # ( select ST_Transform(ST_SetSRID(ST_Extent(ST_MakeEnvelope(-1.0935510737589595, 46.20155315742559, -1.0922700929992573, 46.20266107583956)), 4326), 2154) as bbox) as sub
            #        WHERE ST_Intersects(ST_SetSRID(ST_Point(349134,6582999), 2154), geom)
            #        WHERE ST_Intersects(sub.bbox, geom)
            #            WHERE ST_Intersects(geom, ST_Transform(ST_MakeEnvelope(-1.0935510737589595, 46.20155315742559, -1.0922700929992573, 46.20266107583956, 4326), 2154))
            #            WHERE ST_Intersects(geom,ST_MakeEnvelope(ST_GeomFromText("{bbox_final.wkt}"), 2154))

            gdf = gpd.read_postgis(
                sql=f'{QUERY}', con=create_engine(uri, echo=False), crs=2154
            )
            if gdf is not None:
                gdf_list.append(gdf)

        self.gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))



        self.gdf['key_umep'] = [self.cosia_keys[x] for x in self.gdf.classe]
        self.gdf = self.gdf.rename_geometry('geometry')
        self.gdf = gpd.clip(self.gdf, gdf_project)
        return self.gdf

    def create_trees_from_cosia(
            self,
            geom_col='geometry',
            height=6.0,
            type=2,
            trunk_zone=3.0,
            diameter=4.0,
            resolution=13,
    ):
        """ """
        hexagone_arbres = self.gdf[
            (self.gdf['classe'] == 'Conifère') | (self.gdf['classe'] == 'Feuillu')
            ].to_crs(4326)
        position_arbres = hexagone_arbres.h3.polyfill_resample(resolution=resolution)
        position_arbres['centre'] = [x.centroid for x in position_arbres[geom_col]]
        point_arbres = position_arbres.copy()
        point_arbres['geometry'] = position_arbres['centre']
        point_arbres['height'] = [height for x in position_arbres['centre']]
        point_arbres['type'] = [type for x in position_arbres['centre']]
        point_arbres['trunk_zone'] = [trunk_zone for x in position_arbres['centre']]
        point_arbres['diameter'] = [diameter for x in position_arbres['centre']]
        point_arbres.drop('centre', inplace=True, axis=1)
        return point_arbres




if __name__ == '__main__':
    # dem = Dem(output_path="./")
    # dem.bbox = [-1.15643, 46.16123, -1.15127, 46.16378]
    # ign_dem = dem.run()


    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pymdu.commons.BasicFunctions import (
        extract_coordinates_from_filenames,
        get_intersection_with_bbox_and_attributes,
    )

    bbox_coords = [-1.152704, 46.181627, -1.139893, 46.18699]
    # fixme : interroger une base accessible à tout le monde
    # chemoin des fichiers cosia
    directory_path = os.path.join('D:\\CoSIA_D017_2021\\CoSIA_D017_2021')
    gdf_coordinates = extract_coordinates_from_filenames(directory_path)
    # intersection_gdf = get_intersection_with_bbox(gdf_coordinates, bbox_coords)
    cosia = get_intersection_with_bbox_and_attributes(
        gdf_coordinates, bbox_coords, directory_path
    )
    table_color_cosia = Cosia().table_color_cosia
    cosia['color'] = [table_color_cosia[x] for x in cosia.classe]
    # Tracer le GeoDataFrame
    fig, ax = plt.subplots(figsize=(10, 10))
    cosia.plot(ax=ax, edgecolor=None, color=cosia['color'])


    def supprimer_caracteres_speciaux(chaine):
        # Utilise une expression régulière pour supprimer tout ce qui n'est pas un caractère alphanumérique
        return re.sub(r'[^a-zA-Z0-9]', '', chaine)


    # Créer les patches pour chaque couleur et sa description dans la légende
    patches = [
        mpatches.Patch(color=value, label=supprimer_caracteres_speciaux(label))
        for (value, label) in zip(table_color_cosia.values(), table_color_cosia.keys())
    ]

    # Ajouter la légende personnalisée
    plt.legend(
        handles=patches,
        loc='upper right',
        title='Cosia Legend',
        bbox_to_anchor=(1.0, 1.0),
    )

    # Afficher la carte avec la légende
    plt.show()
