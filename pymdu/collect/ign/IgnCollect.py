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
import io
import json
import os
import os.path
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
from functools import lru_cache
from xml.etree import ElementTree

import pandas as pd
import requests
import urllib3
from owslib.fes2 import BBox
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
from shapely.geometry import Polygon

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.commons.BasicFunctions import _clean_str

pd.set_option("display.max_columns", None)

try:
    from osgeo import gdal, ogr
except ImportError:
    pass

warnings.filterwarnings("ignore")


# session = CachedSession(
#     'demo_cache',
#     use_cache_dir=False,  # Save files in the default user cache dir
#     cache_control=False,  # Use Cache-Control headers for expiration, if available
#     expire_after=timedelta(minutes=1),  # Otherwise expire responses after one minute
#     allowable_methods=['GET', 'POST'],  # Cache POST requests to avoid sending the same data twice
#     allowable_codes=[200, 400],  # Cache 400 responses as a solemn reminder of your failures
#     ignored_parameters=['api_key'],  # Don't match this param or save it in the cache
#     match_headers=True,  # Match all request headers
#     stale_if_error=True,  # In case of request errors, use stale cache data if possible
# )


class IgnCollect(GeoCore):
    """
    ===
    Classe qui permet
    - de construire une reqûete pour interroger l'API de l'IGN
    - enregistre les données dans le dossier ./demo/
    ===
    """

    # https://docs.geoserver.org/2.22.x/en/user/tutorials/cql/cql_tutorial.html

    _cql_filter: str | None = None
    _cutline_from_polygon: Polygon | None = None

    def __init__(self):
        self.content = None
        self.filter_xml = None
        self.ign_keys = {
            "buildings": "BDTOPO_V3:batiment",
            "cosia": "IGNF_COSIA_2021-2023_WMS",
            "water": "BDTOPO_V3:plan_d_eau",
            "road": "BDTOPO_V3:troncon_de_route",
            "irc": "ORTHOIMAGERY.ORTHOPHOTOS.IRC",
            # "ortho": "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO",
            "ortho": "HR.ORTHOIMAGERY.ORTHOPHOTOS",
            "dem": "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES",
            "dsm": "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES.MNS",
            "cadastre": "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle",
            "iris": "STATISTICALUNITS.IRIS:contours_iris",
            "hydrographique": "BDCARTO_V5:detail_hydrographique",
            "vegetation": "BDTOPO_V3:zone_de_vegetation",
            "isochrone": "bdtopo-valhalla",
            "altitude": "SERVICE_CALCUL_ALTIMETRIQUE",
        }

        # https://geoservices.ign.fr/bascule-vers-la-geoplateforme
        path_filename_ressource = str(
            self.collect_path.joinpath("ign/Tableau-suivi-services-web-23-05-2025.csv")
        )
        self.df_csv_file = pd.read_csv(
            path_filename_ressource,
            encoding="ISO-8859-1",
            sep=";",
            index_col=4,
            header=0,
        )
        print(self.df_csv_file.columns)

    @property
    def cql_filter(self):
        return self._cql_filter

    @cql_filter.setter
    def cql_filter(self, value):
        self._cql_filter = value

    @property
    def cutline_from_polygon(self):
        return self._cutline_from_polygon

    @cutline_from_polygon.setter
    def cutline_from_polygon(self, value):
        self._cutline_from_polygon = value

    @lru_cache(maxsize=None)
    def __get_full_list_wfs(self, topic, version="2.0.0"):
        raw_data_file = self.__get_capabilities(
            key=topic, version=version, service="wfs"
        )

        root = ET.parse(raw_data_file).getroot()

        list_var = ["FeatureTypeList"]

        find = False

        for i in range(len(root)):
            for var in list_var:
                if _clean_str(root[i].tag) == var:
                    data = root[i]
                    find = True
                    break
            if find:
                break

        list_df = []

        for i in range(len(data)):
            df = data[i]
            d = {}

            list_var0 = ["Name", "Title", "DefaultCRS", "Abstract"]
            list_var1 = ["WGS84BoundingBox", "Keywords"]
            list_subvar = ["LowerCorner", "UpperCorner", "Keyword"]

            for j in range(len(df)):
                for var in list_var0:
                    if _clean_str(df[j].tag) == var:
                        d[var] = df[j].text

                for var in list_var1:
                    if _clean_str(df[j].tag) == var:
                        for z in range(len(df[j])):
                            for subvar in list_subvar:
                                if _clean_str(df[j][z].tag) == subvar:
                                    d[subvar] = df[j][z].text

            df2 = pd.DataFrame(d, index=[0])

            list_df.append(df2)

        if len(list_df) > 0:
            data_all = (
                pd.concat(list_df).reset_index(drop=True).dropna(axis=0, how="all")
            )
        else:
            data_all = list_df

        return data_all

    @lru_cache(maxsize=None)
    def __get_capabilities(self, topic, version="1.0.0", service="wmts"):
        # https://geoservices.ign.fr/services-geoplateforme-diffusion
        service_upper = service.upper()

        link = f"https://data.geopf.fr/{service}/ows?SERVICE={service_upper}&VERSION={version}&REQUEST=GetCapabilities"

        try:
            proxies = {
                "http": os.environ["http_proxy"],
                "https": os.environ["https_proxy"],
            }
        except:
            proxies = {"http": "", "https": ""}

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        getCapabilities = requests.get(link, proxies=proxies, verify=False)

        # raw_data_file = tempfile.mkdtemp() + "\\" + "raw_data_file"
        # with open(raw_data_file, "wb") as f:
        #     f.write(results.content)
        #     f.close()
        file = io.BytesIO(getCapabilities.content)

        return file

    def get_geoportail_list(self, format="WFS", topic="essentiels", version="2.0.0"):
        """

        Args:
            format : environnement, essentiels, economie, clc, ...

        Examples:
        """

        data_full_list = self.__get_full_list_wfs(topic=topic, version=version)

        print(data_full_list)

        if len(data_full_list) > 0:
            list_var = [
                "Name",
                "Identifier",
                "Title",
                "DefaultCRS",
                "SupportedCRS",
                "TileMatrixSet",
                "Abstract",
                "LegendURL",
                "Format",
            ]

            list_col = [col for col in data_full_list.columns if col in list_var]

            data_list = data_full_list[list_col]
            data_list = data_list.drop_duplicates().reset_index(drop=True)

            if "Name" in data_list.columns:
                data_list.rename(columns={"Name": "Identifier"}, inplace=True)

            data_list["DataFormat"] = format
            data_list["Topic"] = topic
            data_list["ApiVersion"] = version

        data_all = data_list.reset_index(drop=True)

        # set column order
        first_col = [
            "Topic",
            "DataFormat",
            "ApiVersion",
            "Identifier",
            "Abstract",
            "Title",
            "ZoomRange",
        ]
        available_col = [col for col in first_col if col in data_all.columns]
        other_col = [col for col in data_all.columns if col not in available_col]

        data_all = data_all[available_col + other_col]

        return data_all

    def __execute_ign_old(self, key: str = "buildings"):
        row = self.get_row_ressource(key)
        name = row["Nom technique"].values[0]
        url = row["URL d'accès"].values[0].split("&REQUEST=GetCapabilities")[0]
        self._bbox = [round(num, 5) for num in self._bbox]
        print(url)
        payload = {
            "road": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                },
            },
            "water": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                },
            },
            "buildings": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                },
            },
            "irc": {
                "headers": {},
                "params": {
                    "LAYERS": name,
                    "EXCEPTIONS": "text/xml",
                    "FORMAT": "image/geotiff",
                    "SERVICE": "WMS",
                    "VERSION": "1.3.0",
                    "REQUEST": "GetMap",
                    "STYLES": "",
                    "CRS": "EPSG:4326",
                    "BBOX": f"{self._bbox[1]},{self._bbox[0]},{self._bbox[3]},{self._bbox[2]}",
                    "WIDTH": 1000,
                    "HEIGHT": 1000,
                    "DPI": 50,
                },
            },
            "dem": {
                "headers": {},
                "params": {
                    "LAYERS": name,
                    "FORMAT": "image/geotiff",
                    "SERVICE": "WMS",
                    "VERSION": "1.3.0",
                    "REQUEST": "GetMap",
                    "STYLES": "",
                    "CRS": "EPSG:4326",
                    "BBOX": f"{self._bbox[1]},{self._bbox[0]},{self._bbox[3]},{self._bbox[2]}",
                    "WIDTH": 1000,
                    "HEIGHT": 1000,
                    "DPI": 50,
                },
            },
            "cadastre": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                },
            },
            "iris": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                    # "CQL_FILTER": "code_iris ='172000102'"
                    # "CQL_FILTER": "nom_com LIKE 'Lago%'"
                },
            },
            "vegetation": {
                "headers": {"Content-type": "application/json"},
                "params": {
                    "REQUEST": "GetFeature",
                    "TYPENAME": name,
                    "SRSNAME": "EPSG:4326",
                    "BBOX": f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}",
                    "EPSG": 4326,
                    "STARTINDEX": 0,
                    "COUNT": 10000,
                    "outputFormat": "application/json",
                    "SERVICE": "WFS",
                    # "CQL_FILTER": "code_iris ='172000102'"
                    # "CQL_FILTER": "nom_com LIKE 'Lago%'"
                },
            },
        }

        if self._cql_filter:
            del payload[key]["params"]["BBOX"]
            del payload[key]["params"]["EPSG"]
            payload[key]["params"]["CQL_FILTER"] = self._cql_filter

        if self.cutline_from_polygon:
            del payload[key]["params"]["BBOX"]
            payload[key]["params"][
                "CQL_FILTER"
            ] = f"INTERSECTS(geometrie,{str(self.cutline_from_polygon.wkt)})"

            # new_url = f'https://wxs.ign.fr/essentiels/geoportail/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAME=BDTOPO_V3:batiment&SRSNAME=EPSG:4326,EPSG:4326&STARTINDEX=0&COUNT=10000&outputFormat=application%2Fjson&SERVICE=WFS&cql_filter={payload[key]["params"]["cql_filter"]}'  # print(new_url)  #  # response = requests.get(url=new_url, verify=False)

        new_url = self.__get_url_ign(url, payload[key]["params"])
        response = requests.get(
            url=new_url, headers=payload[key]["headers"], verify=False
        )

        # print('response content  : ', response.content)
        #
        # if "errors" in response.content:
        #     raise Exception(response.content)

        print("URL  : ", response.url)

        if key == "irc":
            with open(os.path.join(TEMP_PATH, "img.tiff"), "wb") as f:
                f.write(response.content)
            print("URL  : ", os.path.join(TEMP_PATH, "img.tiff"))
        elif key == "dem":
            with open(os.path.join(TEMP_PATH, "dem.tiff"), "wb") as f:
                f.write(response.content)
            print("URL  : ", os.path.join(TEMP_PATH, "dem.tiff"))

        self.content = response.content
        # self.content = json.loads(response.text)
        return self

    def execute_ign(self, key: str = "buildings", **kwargs):
        row = self.get_row_ressource(key=key)
        print(row.index.values)
        typename = row.index.values[0]
        print(row["URL d'acces Geoplateforme"].values[0])

        url = (
            row["URL d'acces Geoplateforme"]
            .values[0]
            .split("&REQUEST=GetCapabilities")[0]
        )

        if key in [
            "buildings",
            "road",
            "water",
            "cadastre",
            "iris",
            "vegetation",
            "hydrographique",
        ]:
            # url = "https://data.geopf.fr/wfs/ows"
            print("Geo url", url)
            wfs2 = WebFeatureService(url=url, version="2.0.0", timeout=130)

            title = wfs2.identification.title
            version = wfs2.identification.version
            type_wfs = wfs2.identification.type
            print("execute_ign", title, version, type_wfs)
            # requests_possible = [f"{url}" + "/" + operation.name for operation in wfs2.operations]
            # print(requests_possible)
            # print("\n\n")

            if self._cql_filter:
                # filter = PropertyIsEqualTo(propertyname='nom_com', literal='Lagord')
                # self.filter_xml = ElementTree.tostring(filter.toXML()).decode("utf-8")

                BBOX = BBox(bbox=self._bbox, crs="EPSG:4326")
                print(BBOX)
                self.filter_xml = ElementTree.tostring(
                    BBOX.toXML(), encoding="ascii", method="xml", xml_declaration=True
                ).decode("utf-8")
                print(self.filter_xml)

                # filter1 = fes2.PropertyIsEqualTo("nom_com", "Poitiers")
                # fr = fes2.FilterRequest()
                # self.filter_xml = fr.setConstraint(filter1, tostring=True)
                # print(self.filter_xml)
                # self.filter_xml = "<Filter><BBOX><PropertyName>Geometry</PropertyName> <Box srsName='EPSG:4326'><coordinates>%f,%f %f,%f</coordinates> </Box></BBOX></Filter>" % (
                # self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])

                self._bbox = None

            print("typename", typename)
            response_file = wfs2.getfeature(
                typename=typename,
                bbox=self._bbox,
                filter=self.filter_xml,
                startindex=0,
                maxfeatures=10000,
                # propertyname=["nom_com"],
                outputFormat="application/json",
            )

            self.content = response_file
        elif key == "isochrone":
            url = url.replace("/getcapabilities", "/isochrone")
            resource = f"{kwargs.get('resource')[0]}"
            costValue = f"{kwargs.get('costValue')[0]}"
            poi = f"{kwargs.get('point')[0][0]},{kwargs.get('point')[0][1]}"
            print("poi=>", poi)
            payload = {
                "resource": resource,
                "point": poi,
                "costValue": costValue,
                "costType": "time",
            }
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            response_file = requests.post(
                url=url, headers=headers, data=json.dumps(payload)
            )
            self.content = response_file.content

        else:
            # url = "https://data.geopf.fr/wms-r/wms"
            version = "1.3.0"
            crs = "EPSG:4326"
            wms = WebMapService(url=url, version=version, timeout=130)
            # GetMap (image/jpeg)

            from math import cos, radians

            # résolution au sol cible en mètres/pixel (~20 cm pour l'orthophoto IGN)

            resolution = kwargs.get("resolution") or 1.0
            # Taille bbox
            xmin, ymin, xmax, ymax = self._bbox
            lon_center = (xmin + xmax) / 2
            lat_center = (ymin + ymax) / 2

            # Conversion deg → m approximative (valable proche France métropolitaine)
            deg_to_m_lat = 111320
            deg_to_m_lon = 40075000 * cos(radians(lat_center)) / 360

            width_m = (xmax - xmin) * deg_to_m_lon
            height_m = (ymax - ymin) * deg_to_m_lat

            width_px = int(width_m / resolution)
            height_px = int(height_m / resolution)

            # print("width_px", width_px, "height_px", height_px)

            # Inversion bbox si nécessaire (EPSG:4326 + WMS 1.3.0)
            if key == "ortho" and version == "1.3.0" and crs == "EPSG:4326":
                bbox_str = [ymin, xmin, ymax, xmax]
            else:
                bbox_str = [xmin, ymin, xmax, ymax]

            response_file = wms.getmap(
                layers=[typename],
                srs=crs,
                crs=crs,
                bbox=bbox_str,
                # width=width_px,
                # height=height_px,
                size=(width_px, height_px),
                exceptions="text/xml",
                format="image/geotiff",
                transparent=True,
                styles=["normal"],
            )

            if key in ["irc", "dem", "cosia"]:
                with open(os.path.join(TEMP_PATH, f"{key}.tiff"), "wb") as f:
                    f.write(response_file.read())
                print("URL  : ", os.path.join(TEMP_PATH, f"{key}.tiff"))

            self.content = response_file.read()

        return self

    @staticmethod
    def __get_url_ign(url, payload):
        # new_url = url + '&' + urllib.parse.urlencode(payload, safe='():&,+').replace('&EPSG=', ',EPSG:').replace('&json', '/json').replace('&geotiff', '/geotiff')
        new_url = (
            url
            + "&"
            + urllib.parse.urlencode(payload)
            .replace("%3A", ":")
            .replace("%2F", "&")
            .replace("%2C", ",")
            .replace("&EPSG=", ",EPSG:")
            .replace("&json", "/json")
            .replace("&geotiff", "/geotiff")
            .replace("%28", "(")
            .replace("%29", ")")
            .replace("%2B", "+")
        )
        return new_url

    def get_row_ressource(self, key: str = "buildings"):
        print("key=>", key)
        row = self.df_csv_file.loc[(self.df_csv_file.index == self.ign_keys[key])]
        return row


if __name__ == "__main__":
    ign = IgnCollect()
    ign.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    content = ign.execute_ign(key="cosia", resolution=0.2).content
    file_tiff = io.BytesIO(content)
    # Write the stuff
    with open("cosia.tiff", "wb") as f:
        f.write(content)

    # content = ign.execute_ign(key="ortho", resolution=0.2).content
    # file_tiff = io.BytesIO(content)
    # # Write the stuff
    # with open("output.tiff", "wb") as f:
    #     f.write(content)
    #
    # import rioxarray as rxr
    # from rasterio.enums import Resampling
    #
    # dataarray = rxr.open_rasterio(file_tiff)
    # dataarray = dataarray.rio.reproject(
    #     dst_crs=ign._epsg,
    #     resolution=1,
    #     resampling=Resampling.nearest,
    #     # nodata=-9999, fill_value=-9999
    # )
    # dataarray.rio.to_raster(
    #     "./ortho.tiff",
    #     compress="lzw",
    #     bigtiff="YES",
    #     num_threads="all_cpus",
    #     tiled=True,
    #     driver="GTiff",
    #     predictor=2,
    #     discard_lsb=2,
    # )

    # for key in [
    #     "buildings",
    #     "dem",
    #     "cadastre",
    #     "iris",
    #     "vegetation",
    #     "water",
    #     "road",
    #     "hydrographique",
    # ]:
    #     print(key)
    #     if key == "iris":
    #         ign.cql_filter = "nom_com IN ('Lagord', 'Saint-Jean-de-Liversay', 'La-Rochelle', 'Dompierre-sur-Mer')"
    #     else:
    #         ign.cql_filter = None
    #
    #     ign.execute_ign(key=key)
    #
    #     if key != "dem":
    #         file = (
    #             ign.content
    #             if isinstance(ign.content, io.BytesIO)
    #             else io.BytesIO(ign.content)
    #         )
    #         gdf = gpd.read_file(file, driver="GeoJSON")
    #         print(gdf.head(100))
