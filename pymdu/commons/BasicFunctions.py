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
import re
from datetime import datetime, timedelta
from math import *

import folium
import geopandas as gpd
import geopy
import h3
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from folium.plugins import Draw
from geopy.distance import geodesic
from ipywidgets import Output
from shapely import wkb, box
from shapely.geometry import Point, MultiPolygon, Polygon

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scienceplots
from matplotlib import rcParams


class BasicFunctions(object):
    def __init__(self):
        self.date_format = "%y-%m-%d %H:%M:%S"

    @staticmethod
    def get_interval(a: int, mylist: list):
        for i in mylist:
            if a - i < 0:
                end = i
                break
        for ii in list(reversed(mylist)):
            if a - ii > 0:
                init = ii
                break
        return init, end

    @staticmethod
    def from_string_to_datetime(time_str: str):
        datetime_object = datetime.strptime(time_str, "%y-%m-%d %H:%M:%S")
        return datetime_object

    @staticmethod
    def generate_datetime_list(
        init: str = "2022-06-21 06:00:00",
        end: str = "2022-06-21 19:00:00",
        time_delta_hours: int = 3,
    ):
        liste_date = [
            datetime.strptime(init, "%Y-%m-%d %H:%M:%S"),
            datetime.strptime(end, "%Y-%m-%d %H:%M:%S"),
            timedelta(hours=time_delta_hours),
        ]
        return liste_date

    @staticmethod
    def trouver_nombre_plus_proche(liste, nombre):
        nombre_plus_proche = None
        difference_min = float("inf")

        for element in liste:
            difference = abs(element - nombre)
            if difference < difference_min:
                difference_min = difference
                nombre_plus_proche = element

        return nombre_plus_proche

    @staticmethod
    def drop_z(gdf: gpd.GeoDataFrame):
        _drop_z = lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))
        new_gdf = gdf.explode()
        new_gdf.geometry = new_gdf.geometry.transform(_drop_z)
        return new_gdf

    @staticmethod
    def epw_columns():
        columns = [
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Data Source and Uncertainty Flags",
            "Dry Bulb Temperature",
            "Dew Point Temperature",
            "Relative Humidity",
            "Atmospheric Station Pressure",
            "Extraterrestrial Horizontal Radiation",
            "Extraterrestrial Direct Normal Radiation",
            "Horizontal Infrared Radiation Intensity",
            "Global Horizontal Radiation",
            "Direct Normal Radiation",
            "Diffuse Horizontal Radiation",
            "Global Horizontal Illuminance",
            "Direct Normal Illuminance",
            "Diffuse Horizontal Illuminance",
            "Zenith Luminance",
            "Wind Direction",
            "Wind Speed",
            "Total Sky Cover",
            "Opaque Sky Cover (used if Horizontal IR Intensity missing)",
            "Visibility",
            "Ceiling Height",
            "Present Weather Observation",
            "Present Weather Codes",
            "Precipitable Water",
            "Aerosol Optical Depth",
            "Snow Depth",
            "Days Since Last Snowfall",
            "Albedo",
            "Liquid Precipitation Depth",
            "Liquid Precipitation Quantity",
            "",
        ]
        return columns

    @staticmethod
    def epw_index():
        initRun = "2021-01-01 01:00:00"
        endRun = "2022-01-01 00:00:00"
        myIndex = pd.date_range(start=initRun, end=endRun, freq="1h")
        return myIndex

    @staticmethod
    # TODO : construire avec une loop
    def lcz_color():
        tableCorresp = {
            1: ["LCZ 1: Compact high-rise", "#8b0101", 1.0, "LCZ_PRIMARY_1"],
            2: [
                "LCZ 2: Compact mid-rise",
                "#cc0200",
                0.9411764705882353,
                "LCZ_PRIMARY_2",
            ],
            3: [
                "LCZ 3: Compact low-rise",
                "#fc0001",
                0.8823529411764706,
                "LCZ_PRIMARY_3",
            ],
            4: [
                "LCZ 4: Open high-rise",
                "#be4c03",
                0.8235294117647058,
                "LCZ_PRIMARY_4",
            ],
            5: ["LCZ 5: Open mid-rise", "#ff6602", 0.7647058823529411, "LCZ_PRIMARY_5"],
            6: ["LCZ 6: Open low-rise", "#ff9856", 0.8823529411764706, "LCZ_PRIMARY_6"],
            7: [
                "LCZ 7: Lightweight low-rise",
                "#fbed08",
                0.6470588235294118,
                "LCZ_PRIMARY_7",
            ],
            8: [
                "LCZ 8: Large low-rise",
                "#bcbcba",
                0.5882352941176471,
                "LCZ_PRIMARY_8",
            ],
            9: [
                "LCZ 9: Sparsely built",
                "#ffcca7",
                0.5294117647058824,
                "LCZ_PRIMARY_9",
            ],
            10: [
                "LCZ 10: Heavy industry",
                "#57555a",
                0.47058823529411764,
                "LCZ_PRIMARY_10",
            ],
            101: [
                "LCZ A: Dense trees",
                "#006700",
                0.4117647058823529,
                "LCZ_PRIMARY_101",
            ],
            102: [
                "LCZ B: Scattered trees",
                "#05aa05",
                0.35294117647058826,
                "LCZ_PRIMARY_102",
            ],
            103: [
                "LCZ C: Bush,scrub",
                "#648423",
                0.29411764705882354,
                "LCZ_PRIMARY_103",
            ],
            104: [
                "LCZ D: Low plants",
                "#bbdb7a",
                0.23529411764705882,
                "LCZ_PRIMARY_104",
            ],
            105: [
                "LCZ E: Bare rock or paved",
                "#010101",
                0.17647058823529413,
                "LCZ_PRIMARY_105",
            ],
            106: [
                "LCZ F: Bare soil or sand",
                "#fdf6ae",
                0.11764705882352941,
                "LCZ_PRIMARY_106",
            ],
            107: ["LCZ G: Water", "#6d67fd", 0.058823529411764705, "LCZ_PRIMARY_107"],
        }

        value_table = {
            "LCZ_PRIMARY_1": 1.0,
            "LCZ_PRIMARY_2": 0.8,
            "LCZ_PRIMARY_3": 0.7,
            "LCZ_PRIMARY_4": 0.6,
            "LCZ_PRIMARY_5": 0.5,
            "LCZ_PRIMARY_6": 0.4,
            "LCZ_PRIMARY_7": 0.3,
            "LCZ_PRIMARY_8": 0.2,
            "LCZ_PRIMARY_9": 0.1,
            "LCZ_PRIMARY_10": 0.01,
        }

        return tableCorresp, value_table


# def GroundGridIn3d(grid_geo):
#     gridin3d = GeomLib.forceZCoordinateToZ0(grid_geo, z0=0)
#     return gridin3d


def centroid(self):  # rectangle centroid
    coord = self.exterior.coords
    face_p0 = np.array(coord[0])
    face_p2 = np.array(coord[2])
    face_ce = (face_p0 + face_p2) / 2
    return Point(face_ce)


def Surface_diameter(surface):
    coord = surface.exterior.coords
    d = distance3d(Point(coord[0]), Point(coord[2]))
    return d


def distance3d(point1, point2):
    d = sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
        + (point1.z - point2.z) ** 2
    )
    return d


# get area from 3d polygon


def area3d(self):
    coord = self.exterior.coords
    d1 = distance3d(Point(coord[0]), Point(coord[1]))
    d2 = distance3d(Point(coord[1]), Point(coord[2]))
    return d1 * d2


def points_from_polygons(polygons) -> list:
    points = []
    for mpoly in polygons:
        if isinstance(mpoly, MultiPolygon):
            polys = list(mpoly)
        else:
            polys = [mpoly]
        for polygon in polys:
            for point in polygon.exterior.coords:
                points.append(point)
            for interior in polygon.interiors:
                for point in interior.coords:
                    points.append(point)
    return points


def geo_lat_lon_from_h3(
    df: pd.DataFrame, from_h3_column: str, lat: str = "lat", lon: str = "lon"
) -> pd.DataFrame:
    """ """
    df[lat], df[lon] = zip(*df[from_h3_column].apply(lambda x: h3.h3_to_geo(x)))
    return df


def geo_boundary_to_polygon(x: str) -> Polygon:
    """
    Transform h3 geo boundary to shapely Polygon
    Parameters
    ----------
    x: str
        H3 hexagon index
    Returns
    -------
    polygon: Polygon
        Polygon representing H3 hexagon area
    """
    return Polygon(
        [bound[::-1] for bound in h3.h3_to_geo_boundary(x)]
    )  # format as x,y (lon, lat)


def _clean_str(string):
    return re.sub(r"{.*}", "", string)


def from_point_to_bbox(LATITUDE, LONGITUDE, meters=100):
    bearings = [0, 90]
    origin = geopy.Point(LATITUDE, LONGITUDE)
    l = []

    for bearing in bearings:
        destination = geodesic(meters=meters).destination(origin, bearing)
        coords = destination.longitude, destination.latitude
        l.extend(coords)
    # xmin, ymin, xmax, ymax
    return l


def draw_bbox_with_folium(lat=46.160329, lon=-1.151139, zoom_start=13):
    # Créer une carte centrée sur une position de votre choix
    m = folium.Map(location=[lat, lon], zoom_start=13)

    # Ajouter l'outil de dessin avec seulement l'option de dessin rectangle activée
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    )
    draw.add_to(m)

    # Ajouter un div pour afficher les coordonnées
    output = Output()

    # Afficher la carte
    with output:
        display(m)

    # Afficher la carte et le widget output
    display(output)

    # Ajouter le script JavaScript pour capturer les coordonnées et afficher dans l'interface
    js = """
    <script>
        var map = window.map;
        var drawControl = window.drawControl;
        var drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);

        map.on(L.Draw.Event.CREATED, function (event) {
            var layer = event.layer;
            drawnItems.addLayer(layer);

            // Obtenir les coordonnées de la bbox
            var bounds = layer.getBounds();

            // Formater les coordonnées en [minx, miny, maxx, maxy]
            var minx = bounds.getSouthWest().lng;
            var miny = bounds.getSouthWest().lat;
            var maxx = bounds.getNorthEast().lng;
            var maxy = bounds.getNorthEast().lat;

            var bboxCoords = [minx, miny, maxx, maxy];

            // Afficher les coordonnées dans la console
            console.log("Bounding Box Coordinates: ", bboxCoords);

            // Afficher les coordonnées dans une alerte
            alert('Bounding Box Coordinates: [minx, miny, maxx, maxy] = ' + bboxCoords);
        });
    </script>
    """

    # Afficher le script JavaScript dans le notebook
    display(HTML(js))


def extract_coordinates_from_filenames(
    directory_path: str, departement_code: str = "17", year : str = '2021'
) -> gpd.GeoDataFrame:
    """
    Extrait les informations X et Y à partir de noms de fichiers .gpkg respectant le format 'D0{departement_code}_2021_X_Y_vecto.gpkg'
    et multiplie les coordonnées extraites par 1000. Ajoute ensuite une colonne 'geometry' avec une géométrie carrée
    de 10km de côté en utilisant le coin supérieur gauche (X, Y) en EPSG:2154.

    Args:
        departement_code (str): Code postal du département.
        directory_path (str): Chemin vers le dossier contenant les fichiers .gpkg.

    Returns:
        gpd.GeoDataFrame: Un GeoDataFrame avec les noms de fichiers, les coordonnées (X, Y) et la géométrie associée.
    """
    # Dictionnaire pour stocker les coordonnées extraites et les géométries
    coordinates = {}

    # Pattern pour extraire les coordonnées X et Y à partir des noms de fichiers
    pattern = re.compile(rf"D0{departement_code}_{year}_(\d+)_(\d+)_vecto\.gpkg")

    # Parcourir tous les fichiers dans le dossier spécifié
    for filename in os.listdir(directory_path):
        if filename.endswith(".gpkg"):
            match = pattern.match(filename)
            if match:
                # Extraire X et Y, multiplier par 1000
                x_coord = int(match.group(1)) * 1000
                y_coord = int(match.group(2)) * 1000

                # Créer une géométrie de carré de 10 km x 10 km à partir du coin supérieur gauche (X, Y)
                square_size = 10000  # Taille du carré de 10 km
                geometry = Polygon(
                    [
                        (x_coord, y_coord),  # Coin supérieur gauche
                        (x_coord + square_size, y_coord),  # Coin supérieur droit
                        (
                            x_coord + square_size,
                            y_coord - square_size,
                        ),  # Coin inférieur droit
                        (x_coord, y_coord - square_size),  # Coin inférieur gauche
                        (x_coord, y_coord),  # Retour au coin supérieur gauche
                    ]
                )

                # Ajouter les coordonnées et la géométrie au dictionnaire
                coordinates[filename] = {
                    "X": x_coord,
                    "Y": y_coord,
                    "geometry": geometry,
                }

    # Convertir le dictionnaire en GeoDataFrame
    gdf = gpd.GeoDataFrame.from_dict(coordinates, orient="index", crs="EPSG:2154")

    return gdf


def get_intersection_with_bbox(
    gdf: gpd.GeoDataFrame, bbox_coords: list
) -> gpd.GeoDataFrame:
    """
    Calcule l'intersection entre les géométries du GeoDataFrame et une bounding box définie par bbox_coords.
    La bounding box est initialement en EPSG:4326 (WGS84) et est convertie en EPSG:2154 avant de calculer l'intersection.

    Args:
        gdf (gpd.GeoDataFrame): Le GeoDataFrame contenant les géométries, en EPSG:2154.
        bbox_coords (list): La bounding box sous la forme [xmin, ymin, xmax, ymax] en EPSG:4326.

    Returns:
        gpd.GeoDataFrame: Un GeoDataFrame avec les géométries qui intersectent la bbox reprojetée en EPSG:2154.
    """
    # Créer une géométrie de bounding box à partir des coordonnées (EPSG:4326)
    bbox = box(*bbox_coords)

    # Créer un GeoDataFrame pour la bbox avec le CRS EPSG:4326
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

    # Reprojeter la bbox en EPSG:2154 (le CRS du GeoDataFrame d'entrée)
    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

    # Calculer l'intersection entre le GeoDataFrame et la bbox reprojetée
    intersection_gdf = gpd.overlay(gdf, bbox_gdf, how="intersection")

    return intersection_gdf


def get_intersection_with_bbox_and_attributes(
    gdf: gpd.GeoDataFrame, bbox_coords: list, directory_path: str, bbox_crs="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Calcule l'intersection entre les géométries du GeoDataFrame et une bounding box définie par bbox_coords,
    relit les fichiers .gpkg correspondants, et extrait les attributs complets des géométries intersectées.

    Args:
        gdf (gpd.GeoDataFrame): Le GeoDataFrame contenant les informations des fichiers .gpkg (noms de fichiers, géométries).
        bbox_coords (list): La bounding box sous la forme [xmin, ymin, xmax, ymax]
        bbox_crs (str) : crs du bounding box
        directory_path (str): Chemin vers le dossier contenant les fichiers .gpkg.

    Returns:
        gpd.GeoDataFrame: Un GeoDataFrame avec les géométries qui intersectent la bbox et les attributs d'origine.
    """
    # Créer une géométrie de bounding box à partir des coordonnées
    bbox = box(*bbox_coords)

    # Créer un GeoDataFrame pour la bbox avec le CRS
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=bbox_crs)

    # Reprojeter la bbox en EPSG:2154 (le CRS du GeoDataFrame d'entrée)
    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

    # Extraire uniquement les géométries qui intersectent la bbox, en conservant les attributs d'origine
    filtered_gdf = gdf[gdf.intersects(bbox_gdf.loc[0, "geometry"])]

    # Liste pour stocker les GeoDataFrames des fichiers .gpkg relus
    all_intersections = []
    result_files = []
    # Parcourir chaque ligne du GeoDataFrame filtré pour relire les .gpkg correspondants
    for index, row in filtered_gdf.iterrows():
        file_name = (
            index  # Le nom du fichier est utilisé comme index dans le gdf original
        )

        # Chemin complet du fichier .gpkg d'origine
        file_path = os.path.join(directory_path, file_name)

        # Lire le fichier .gpkg correspondant
        try:
            original_gdf = gpd.read_file(file_path)

            # Vérifier que les CRS correspondent, sinon, reprojeter le fichier .gpkg
            if original_gdf.crs != bbox_gdf.crs:
                original_gdf = original_gdf.to_crs(bbox_gdf.crs)

            # Extraire l'intersection entre le fichier original et la bbox reprojetée
            clipped_gdf = gpd.clip(original_gdf, bbox_gdf)

            # Si l'intersection contient des géométries, les ajouter à la liste
            if not clipped_gdf.empty:
                all_intersections.append(clipped_gdf)
                result_files.append(file_name)

        except Exception as e:
            print(f"Erreur lors de la lecture de {file_name}: {e}")

    # Combiner tous les GeoDataFrames obtenus à partir des .gpkg en un seul GeoDataFrame final
    if all_intersections:
        result_gdf = gpd.GeoDataFrame(
            pd.concat(all_intersections, ignore_index=True), crs=bbox_gdf.crs
        )
    else:
        result_gdf = gpd.GeoDataFrame(columns=original_gdf.columns, crs=bbox_gdf.crs)

    return result_gdf


def remove_linestring_from_geopandas(gdf: gpd.GeoDataFrame):
    crs = gdf.crs
    list_geometry = []
    list_index = []
    for i, row in gdf.iterrows():
        if row.geometry.type == "Polygon":
            list_geometry.append(row.geometry)
            list_index.append(i)
        else:
            pass
    data = {"index": list_index, "geometry": list_geometry}
    gdf_filtered = gpd.GeoDataFrame(data, crs="EPSG:2154")
    return gdf_filtered


def convert_crs(building: gpd.GeoDataFrame, crs=3857):
    building = building.to_crs(crs)
    return building


def trees_to_polygon(trees: gpd.GeoDataFrame, height: float = 6.0):
    new_geom = []
    for p in trees["geometry"]:
        circle = p.buffer(4.0)
        new_geom.append(Polygon(list(circle.exterior.coords)))
    trees["point"] = trees["geometry"]
    trees["geometry"] = new_geom
    trees["geometry"] = trees["geometry"].simplify(1)
    trees["hauteur"] = [height for x in trees["geometry"]]
    return trees


def union_trees_buildings(
    buildings: gpd.GeoDataFrame, trees: gpd.GeoDataFrame, height: float = 6.0
):
    union = trees.overlay(buildings, how="union")
    import numpy as np

    hauteur = []
    for x, y in zip(union["hauteur_1"], union["hauteur_2"]):
        if np.isnan(x):
            hauteur.append(y)
        elif np.isnan(y):
            hauteur.append(x)
        else:
            hauteur.append(height)
    union["hauteur"] = hauteur
    # bug sur la dernière ligne > TODO : mieux comprendre ce bug
    union.drop(union.tail(1).index, inplace=True)
    return union


def shadows_on_ground(shaders_gdf: gpd.GeoDataFrame, shades_gdf: gpd.GeoDataFrame):
    shadows_ground = shades_gdf.overlay(
        shaders_gdf, how="difference", keep_geom_type=True
    )
    shadows_ground = shadows_ground.explode(ignore_index=True)
    shadows_ground["id"] = [x for x in shadows_ground.index]
    return shadows_ground


def convert_3D_2D(geometry):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    """
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == "Polygon":
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == "MultiPolygon":
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo


def plot_sol_occupancy(gdf_before, gdf_after, **kwargs):
    """
    Plot sol occupancy histogram.
    Args:
        gdf_before: geodataframe before added new feautures, with Cosia class sol occupancy
        gdf_after: geodataframe after added new features, with Cosia class sol occupancy
        **kwargs:

    Returns:
        plotly.graph_objects.Figure histogram
    """
    gdf_before["area"] = gdf_before.area
    gdf_after["area"] = gdf_after.area

    class_before = (
        gdf_before.groupby("classe")["area"].sum().reset_index(name="area_before")
    )
    class_after = (
        gdf_after.groupby("classe")["area"].sum().reset_index(name="area_after")
    )

    merged_areas = pd.merge(class_before, class_after, on="classe", how="outer").fillna(
        0
    )
    merged_areas["percentage_before"] = (
        (merged_areas["area_before"]) / merged_areas["area_before"].sum()
    ) * 100
    merged_areas["percentage_after"] = (
        (merged_areas["area_after"]) / merged_areas["area_before"].sum()
    ) * 100

    plt.style.use(["science", "no-latex"])
    rcParams["font.family"] = "DejaVu Sans"

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(merged_areas["classe"]))  # x locations for the groups

    # Bar width
    width = 0.35

    before_patches = []
    after_patches = []

    from pymdu.geometric.Cosia import Cosia

    for i, classe in enumerate(merged_areas["classe"]):
        color = Cosia().table_color_cosia[classe]
        before_bar = ax.bar(
            x[i] - width / 2,
            merged_areas["percentage_before"][i],
            width,
            label="Before" if i == 0 else "",
            color=color,
            edgecolor="black",
        )
        after_bar = ax.bar(
            x[i] + width / 2,
            merged_areas["percentage_after"][i],
            width,
            label="After" if i == 0 else "",
            color=color,
            edgecolor="black",
            hatch="x",
        )
        before_patches.append(before_bar)
        after_patches.append(after_bar)
    # Add labels, title, and legend
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Pourcentage", fontsize=12)
    ax.set_title("Pourcentage avant et après par occupation du sol", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(merged_areas["classe"], rotation=45)
    # ax.legend()
    ax.legend(
        [before_patches[4], after_patches[4]], ["Avant", "Après (x)"], loc="upper left"
    )
    # Add grid for better readability
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    return fig


# Convertir les propriétés en types JSON compatibles
def process_datetime(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    # gdf = gdf[[column for column in gdf.columns if not is_datetime(gdf[column])]]
    for column in gdf.columns:
        if is_datetime(gdf[column]):
            gdf[column] = gdf[column].dt.year
    return gdf
