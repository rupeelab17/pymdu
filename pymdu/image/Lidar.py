# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  Copyright                                                                   *
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
import os

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
import rasterio as rio
import requests
from osgeo import gdal, ogr, osr
from pandas import DataFrame
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from shapely import box
from shapely.geometry import Point
from pyproj import Transformer
from shapely.geometry.multipoint import MultiPoint

from pymdu.GeoCore import GeoCore
from pymdu._typing import FilePath
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.image import rasterize


# https://github.com/DTMilodowski/LiDAR_canopy/blob/3fa568548e2aa921cedce940636ae2536e8d6339/src/LiDAR_io.py
# https://laspy.readthedocs.io/en/latest/installation.html
# python3 -m pip install "laspy[lazrs,laszip]"


class Lidar(GeoCore):
    def __init__(
        self,
        output_path: FilePath = None,
        classification: int = None,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            building_gdf: gpd.GeoDataFrame = None,
            vegetation_gdf: gpd.GeoDataFrame = None,
            water_gdf: gpd.GeoDataFrame = None,
            pedestrian_gdf: gpd.GeoDataFrame = None,
            cosia_gdf: gpd.GeoDataFrame = None,
            dxf_gdf: gpd.GeoDataFrame = None,
            output_path: str = None,
            write_file: bool = True,

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            plt.clf()  # markdown-exec: hide

            from shapely.geometry.geo import box
            from pyproj import Transformer
            import matplotlib.pyplot as plt
            import rasterio.plot
            from pymdu.image.Lidar import Lidar

            lidar = Lidar(
                output_path="./",
                classification=6,
            )
            lidar.bbox = [-1.154894, 46.182639, -1.148361, 46.186820]
            # lidar_gdf = lidar.run().to_gdf()
            # # lidar_gdf.plot(ax=plt.gca(), edgecolor='black')
            # # plt.show()
            # lidar.to_shp(name='LidarTest')

            lidar_tif = lidar.to_tif(write_out_file=True)

            # Lire les données et les afficher avec rasterio.plot
            with lidar_tif.open() as src:
                fig, ax = plt.subplots(figsize=(10, 10))
                rasterio.plot.show(src, ax=ax, title="Lidar CDSM")

            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
            * For module TODOs
        """
        self.list_path_laz = None
        self.output_path = output_path if output_path else TEMP_PATH
        self.classification = classification

    def _get_lidar_points(self):
        url = "https://data.geopf.fr/private/wfs"

        # Créer le transformer pour la conversion de EPSG:4326 vers EPSG:2154
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        # Transformer chaque point de la bbox
        min_x, min_y = transformer.transform(self._bbox[0], self._bbox[1])
        max_x, max_y = transformer.transform(self._bbox[2], self._bbox[3])

        # Créer la bbox en EPSG:2154
        bbox_final = [min_x, min_y, max_x, max_y]
        bbox_string = ",".join(map(str, bbox_final))
        # Afficher la bbox transformée
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "apikey": "interface_catalogue",
            "typeName": "IGNF_LIDAR-HD_TA:nuage-dalle",
            "outputFormat": "application/json",
            "bbox": bbox_string,
        }

        response = requests.get(
            url, params=params, headers={"Accept": "application/json"}
        )
        print(response)
        response = response.json()
        # Extraire les URLs des features
        list_path_laz = [
            feature["properties"]["url"] for feature in response["features"]
        ]
        return min_x, min_y, max_x, max_y, list_path_laz

    def run(self):
        min_x, min_y, max_x, max_y, list_path_laz = self._get_lidar_points()
        list_lidar_gdf = []
        for las_path in list_path_laz:
            # Télécharger le fichier directement en mémoire
            response = requests.get(las_path)

            # Charger le contenu dans un BytesIO
            file_in_memory = io.BytesIO(response.content)

            las = laspy.read(file_in_memory)
            print("las.classification unique", np.unique(las.classification))

            # Read LAS file
            # point_format = las.point_format
            # ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'synthetic', 'key_point', 'withheld',
            # 'overlap', 'scanner_channel', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'user_data',
            # 'scan_angle', 'point_source_id', 'gps_time', 'Amplitude', 'Pulse width', 'Reflectance']

            UR, LR, UL, LL = self.get_las_bbox(las)

            # Filter points within the bounding box
            mask_project = (
                (las.x >= min_x)
                & (las.x <= max_x)
                & (las.y >= min_y)
                & (las.y <= max_y)
                & (las.z >= -10)
                & (las.z <= 5000)
            )
            # print("mask", min_x, max_x, min_y, max_y)
            # Create a new Laspy file with filtered points
            las_new = laspy.LasData(las.header)
            las_new.points = las.points[mask_project]
            # print("las.classification unique", np.unique(las.classification))

            ## Extracts planes and polygons,
            # # Import LAS into numpy array (X=raw integer value x=scaled float value)
            # lidar_points = np.array((las.X, las.Y, las.Z, las.intensity,
            #                          las.classification, las.scan_angle)).transpose()
            # from polylidar import MatrixDouble, Polylidar3D
            # polylidar_kwargs = dict(alpha=0.0, lmax=1.0, min_triangles=20, z_thresh=0.1, norm_thresh_min=0.94)
            # polylidar = Polylidar3D(**polylidar_kwargs)
            # points_mat = MatrixDouble(lidar_points, copy=False)
            # t1 = time.time()
            # mesh, planes, polygons = polylidar.extract_planes_and_polygons(points_mat)
            # t2 = time.time()
            # print("Took {:.2f} milliseconds".format((t2 - t1) * 1000))
            # print("Should see two planes extracted, please rotate.")
            # print("polygons", polygons)

            if self.classification:
                mask = las_new.classification == self.classification
                # Grab an array of all points which meet this threshold

                new_new_las = laspy.create(
                    point_format=las_new.header.point_format,
                    file_version=str(las_new.header.version),
                )
                new_new_las.points = las_new.points[mask]

                print(
                    "We kept %i points out of %i total"
                    % (len(new_new_las.points), len(las_new.points))
                )
            else:
                new_new_las = las_new

            # lidar_points = self.load_lidar_data_by_bbox()
            # Grab the scaled x, y, and z dimensions and stick them together
            # in an nx3 numpy array

            self.lidar_points = np.vstack(
                (
                    new_new_las.X,
                    new_new_las.Y,
                    new_new_las.Z,
                    new_new_las.intensity,
                    new_new_las.classification,
                    new_new_las.scan_angle,
                )
            ).transpose()

            # print("\t\tbuilding KD-trees...")
            # start = time.time()
            # starting_ids, trees = self.create_KDTree(self.lidar_points)
            # print(trees)
            # end = time.time()
            # print("\t\t\t...%.3f s" % (end - start))

            # Transform to pandas DataFrame
            lidar_df = DataFrame(
                data=self.lidar_points,
                columns=["X", "Y", "Z", "intensity", "classification", "scan_angle"],
            )

            # https://geoservices.ign.fr/sites/default/files/2022-05/DT_LiDAR_HD_1-0.pdf
            classification_dict = {
                1: "non_classe",
                2: "sol",
                5: "vegetation_haute",
                3: "vegetation_basse",
                4: "vegetation_moyenne",
                6: "batiment",
                9: "eau",
                17: "pont",
                64: "sursol_perenne",
                65: "artefact",
                66: "points_virtuels",
            }

            lidar_df["classification_name"] = lidar_df["classification"].map(
                classification_dict
            )

            print("classification unique", lidar_df.classification.unique())

            # Transform to geopandas GeoDataFrame
            geometry = [
                Point(xyz) for xyz in zip(new_new_las.X, new_new_las.Y, new_new_las.Z)
            ]
            gdf = gpd.GeoDataFrame(lidar_df, crs=self._epsg, geometry=geometry)
            # gdf.crs = {'init': f'epsg:{self._epsg}'}  # set correct spatial reference
            list_lidar_gdf.append(gdf)

        gdf = gpd.GeoDataFrame(pd.concat(list_lidar_gdf, ignore_index=True))

        polygons = []
        for cls, group in gdf.groupby(
            "classification"
        ):  # Remplacez "classification" par le nom de votre colonne
            multipoint = MultiPoint([pt for pt in group.geometry])
            # Choix de la méthode : convex hull ou alpha shape
            poly = multipoint.convex_hull  # ou alphashape.alphashape(multipoint, alpha)
            polygons.append({"classification": cls, "geometry": poly})

        self.gdf = gpd.GeoDataFrame(polygons, crs=gdf.crs)
        print(self.gdf.head(10))

        return self

    # a similar script, but now only loading points within bbox into memory
    def load_lidar_data_by_bbox(self, las, N, S, E, W, print_npts=True):
        # conditions for points to be included
        X_valid = np.logical_and((las.X <= E), (las.X >= W))
        Y_valid = np.logical_and((las.Y <= N), (las.Y >= S))
        Z_valid = las.Z >= 0
        ii = np.where(np.logical_and(X_valid, Y_valid, Z_valid))

        pts = np.vstack(
            (
                las.X[ii],
                las.Y[ii],
                las.Z[ii],
                las.return_number[ii],
                las.classification[ii],
                las.scan_angle[ii],
                las.gps_time[ii],
            )
        ).transpose()

        if print_npts:
            print("loaded ", pts[:, 0].size, " points")
        return pts

    def get_las_bbox(self, las):
        # get bounding box from las file
        max_xyz = las.header.max
        min_xyz = las.header.min
        UR = np.asarray([max_xyz[0], max_xyz[1]])
        LR = np.asarray([max_xyz[0], min_xyz[1]])
        UL = np.asarray([min_xyz[0], max_xyz[1]])
        LL = np.asarray([min_xyz[0], min_xyz[1]])

        print("UR LR UL LL", UR, LR, UL, LL)
        return UR, LR, UL, LL

    # ---------------------------------------------------------------------------
    # KD-Trees :-)
    # Creates kd-trees to host data.
    # RETURNS  - a second  array containing the starting indices of the points
    #            associated with a given tree for cross checking against the
    #            point cloud
    #          - a list of trees
    @staticmethod
    def create_KDTree(pts, max_pts_per_tree=10**6):
        npts = pts.shape[0]
        ntrees = int(np.ceil(npts / float(max_pts_per_tree)))
        trees = []
        starting_ids = []

        for tt in range(0, ntrees):
            i0 = tt * max_pts_per_tree
            i1 = (tt + 1) * max_pts_per_tree
            if i1 < pts.shape[0]:
                trees.append(cKDTree(pts[i0:i1, 0:2], leafsize=32, balanced_tree=True))
            else:
                trees.append(cKDTree(pts[i0:, 0:2], leafsize=32, balanced_tree=True))
            starting_ids.append(i0)

        return np.asarray(starting_ids, dtype="int"), trees

    def classify(self, las):
        point_records = las.points.copy()

        # filter all single return points of classes 5 (vegetation)
        single_veg = np.where(las.classification == 5)
        # pull out full point records of filtered points, and create an XYZ array for KDTree
        single_veg_points = las.points[single_veg]
        single_veg_points_xyz = np.vstack(
            (single_veg_points.X, single_veg_points.Y, single_veg_points.Z)
        ).transpose()

        # filter class 2 points (ground) create XYZ array for KDTree
        ground_only = np.where(las.classification == 2)
        ground_points = las.points[ground_only]
        ground_points_xyz = np.vstack(
            (ground_points.X, ground_points.Y, ground_points.Z)
        ).transpose()

        # create a KDTree to query against
        ctree = cKDTree(ground_points_xyz)

        # For every single return veg point query against all points within 20 meters.
        # If a higher elevation ground point is nearby, then change the classification
        # from vegetation to ground in the original point array.
        for idx, record in enumerate(single_veg_points_xyz):
            neighbors = ctree.query_ball_point(x=record, r=2000, p=2.0)
            for neighbor in neighbors:
                neighbor_z = ground_points[neighbor].Z
                record_z = single_veg_points[idx].Z
                if neighbor_z >= record_z:
                    single_veg_points[idx].classification = 2

        # update points just once
        point_records[single_veg] = single_veg_points

        out_file = laspy.file.File("new_las_file.las", mode="w", header=las.header)
        out_file.points = point_records
        out_file.close()

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_tif(
        self,
        dsm_out: FilePath = "cdsm.tif",
        write_out_file=False,
        classification_list=[1, 2, 6],
    ):
        _, _, _, _, list_path_laz = self._get_lidar_points()
        memfiles = []
        for las_path in list_path_laz:
            response = requests.get(las_path)
            # Charger le contenu dans un BytesIO
            file_in_memory = io.BytesIO(response.content)
            memfile = self._las2tif(
                las_path=file_in_memory,
                resolution=1.0,
                radius=1,
                sigma=None,
                classification_list=classification_list,
            )
            memfiles.append(memfile)

        memfile = self.merge_tifs(
            input_files=memfiles, dsm_out=dsm_out, write_out_file=write_out_file
        )
        return memfile

    def _las2tif(
        self,
        las_path: FilePath | io.BytesIO,
        weights_sum_out=None,
        clr_out=None,
        resolution=1.0,
        radius=None,
        sigma=None,
        roi=None,
        classification_list=[1, 2, 6],
    ) -> MemoryFile:
        """
        Convert point cloud las to dsm tif
        """
        gdf_project_4326 = gpd.GeoDataFrame(
            gpd.GeoSeries(
                box(self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])
            ),
            columns=["geometry"],
            crs="epsg:4326",
        )
        gdf_project = gdf_project_4326.to_crs(epsg=self._epsg)
        bbox = gdf_project.geometry.total_bounds
        min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]

        las = laspy.read(las_path)

        # Filter points within the bounding box
        mask_project = (
            (las.x >= min_x)
            & (las.x <= max_x)
            & (las.y >= min_y)
            & (las.y <= max_y)
            & (las.z >= -10)
            & (las.z <= 5000)
        )
        # Create a new Laspy file with filtered points
        new_las = laspy.create(
            point_format=las.header.point_format, file_version=str(las.header.version)
        )
        new_las.points = las.points[mask_project]

        # single_veg = np.where(np.logical_or(las.classification == 5))
        # single_veg = np.where(new_las.classification == 6)
        # liste = [1, 2, 6]  # Exemple de liste de classifications
        # Exemple de liste de classifications
        single_veg = np.where(np.isin(new_las.classification, classification_list))
        points = np.vstack((new_las.x[single_veg], new_las.y[single_veg]))
        # points = np.vstack((las.x, las.y))
        if clr_out is None:
            # values = new_las.z[single_veg]
            # values = values[np.newaxis, ...]

            values_height = new_las.z[single_veg][np.newaxis, ...]
            values_classification = new_las.classification[single_veg][np.newaxis, ...]
            values = np.vstack((values_height, values_classification))

        else:
            values = np.vstack((las.z, las.red, las.green, las.blue))

        valid = np.ones((1, points.shape[1]))

        if roi is None:
            roi = {
                "xmin": resolution
                * ((np.amin(new_las.x) - resolution / 2) // resolution),
                "ymax": resolution
                * ((np.amax(new_las.y) + resolution / 2) // resolution),
                "xmax": resolution
                * ((np.amax(new_las.x) + resolution / 2) // resolution),
                "ymin": resolution
                * ((np.amin(new_las.y) - resolution / 2) // resolution),
            }

            roi["xstart"] = roi["xmin"]
            roi["ystart"] = roi["ymax"]
            roi["xsize"] = (roi["xmax"] - roi["xmin"]) / resolution
            roi["ysize"] = (roi["ymax"] - roi["ymin"]) / resolution

        if sigma is None:
            sigma = resolution

        if radius is None:
            radius = 2 * sigma / resolution

        # pylint: disable=c-extension-no-member
        out, weights_sum, mean, stdev, nb_pts_in_disc, nb_pts_in_cell = (
            rasterize.pc_to_dsm(
                points,
                values,
                valid,
                roi["xstart"],
                roi["ystart"],
                int(roi["xsize"]),
                int(roi["ysize"]),
                resolution,
                radius,
                sigma,
            )
        )

        # reshape data as a 2d grid.
        shape_out = (int(roi["ysize"]), int(roi["xsize"]))
        out = out.reshape(shape_out + (-1,))
        mean = mean.reshape(shape_out + (-1,))
        stdev = stdev.reshape(shape_out + (-1,))
        weights_sum = weights_sum.reshape(shape_out)
        nb_pts_in_disc = nb_pts_in_disc.reshape(shape_out)
        nb_pts_in_cell = nb_pts_in_cell.reshape(shape_out)

        # save dsm
        # out: gaussian interpolation
        transform = Affine.translation(roi["xstart"], roi["ystart"])
        transform = transform * Affine.scale(resolution, -resolution)

        profile = DefaultGTiffProfile(
            count=2,  # 2 bandes : hauteur et classification
            dtype=out.dtype,
            width=roi["xsize"],
            height=roi["ysize"],
            transform=transform,
            nodata=np.nan,
        )

        def create_in_memory_file(profile, out) -> MemoryFile:
            # Créer et écrire dans le MemoryFile
            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                # dst.write(out[..., 0], 1)  # Écrire la première bande
                for band in range(2):
                    dst.write(out[..., band], band + 1)

            # Retourner le MemoryFile
            return memfile

        # Retourne l'objet MemoryFile
        memfile = create_in_memory_file(profile, out)

        if weights_sum_out is not None:
            profile = DefaultGTiffProfile(
                count=1,
                dtype=out.dtype,
                width=roi["xsize"],
                height=roi["ysize"],
                transform=transform,
                nodata=np.nan,
            )

            with rio.open(weights_sum_out, "w", **profile) as dst:
                dst.write(weights_sum, 1)

        if clr_out is not None:
            # clr: color r, g, b
            transform = Affine.translation(roi["xstart"], roi["ystart"])
            transform = transform * Affine.scale(resolution, -resolution)

            profile = DefaultGTiffProfile(
                count=3,
                dtype=out.dtype,
                width=roi["xsize"],
                height=roi["ysize"],
                transform=transform,
                nodata=np.nan,
            )

            with rio.open(clr_out, "w", **profile) as dst:
                for band in range(3):
                    dst.write(out[..., band + 1], band + 1)

        return memfile

    def merge_tifs(
        self, input_files: list, dsm_out: FilePath, write_out_file: bool = False
    ) -> MemoryFile:
        """
        Merge multiple GeoTIFF files into a single GeoTIFF file.

        Parameters:
            dsm_out (FilePath): Path to the output GeoTIFF file.
            write_out_file (bool): If True, write the merged GeoTIFF file. Otherwise, return a MemoryFile.
            input_files (list): List of paths to input GeoTIFF files.
        """
        # Open input GeoTIFF files
        src_files_to_mosaic = [rasterio.open(file) for file in input_files]

        # Merge GeoTIFF files
        mosaic, out_trans = merge(src_files_to_mosaic, resampling=Resampling.nearest)

        # Get metadata of one of the input GeoTIFF files
        out_meta = src_files_to_mosaic[0].meta.copy()

        # Update metadata with the new information
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            }
        )

        # Write the merged GeoTIFF file
        if write_out_file:
            with rasterio.open(dsm_out, "w", **out_meta) as dest:
                dest.write(mosaic)

        # Créer et écrire dans le MemoryFile
        memfile = MemoryFile()
        with memfile.open(**out_meta) as dst:
            dst.write(mosaic)  # Écrire la première bande

        # Retourner le MemoryFile
        return memfile

    def process_polygonize(
        self, path_file_csdm: str = "./cdsm.tif"
    ) -> gpd.GeoDataFrame:
        lidar_shp_path = os.path.join(TEMP_PATH, "LidarTest.shp")

        # ================================================
        # open image:
        im = gdal.Open(path_file_csdm)
        print("Nombre de bandes:", im.RasterCount)

        srcband = im.GetRasterBand(1)
        srcband_classification = im.GetRasterBand(2)

        if srcband.GetNoDataValue() is None:
            mask = None
        else:
            mask = srcband

        # Get CRS from raster
        spatial_ref = im.GetSpatialRef()

        # If no CRS found
        if spatial_ref is None:
            spatial_ref = osr.SpatialReference()
            spatial_ref.ImportFromEPSG(2154)

        # create output vector:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(lidar_shp_path):
            driver.DeleteDataSource(lidar_shp_path)

        vector = driver.CreateDataSource(lidar_shp_path)
        layer = vector.CreateLayer(
            lidar_shp_path.replace(".shp", ""), spatial_ref, geom_type=ogr.wkbPolygon
        )

        # create field to write NDVI values:
        field_hauteur = ogr.FieldDefn("hauteur", ogr.OFTReal)
        field_classification = ogr.FieldDefn("class", ogr.OFTInteger)
        layer.CreateField(field_hauteur)
        layer.CreateField(field_classification)
        del field_hauteur, field_classification

        # Polygonize first band (Hauteur)
        gdal.Polygonize(
            srcband, mask, layer, 0, options=[], callback=gdal.TermProgress_nocb
        )

        # Polygonize second band (Classification)
        gdal.Polygonize(
            srcband_classification,
            mask,
            layer,
            1,
            options=[],
            callback=gdal.TermProgress_nocb,
        )

        # close files:
        del im, srcband, vector, layer, srcband_classification

        def clean_inner_polygons(input_shapefile, output_shapefile):
            # Read the input shapefile
            gdf = gpd.read_file(input_shapefile, driver="ESRI Shapefile")
            # Dissolve polygons while preserving classification
            dissolved_gdf = gdf.dissolve(
                by="class"
            )  # Utilise la classification pour dissoudre
            # Reset index to avoid MultiIndex issues
            dissolved_gdf = dissolved_gdf.reset_index()
            # Écriture dans un fichier de sortie si un chemin est fourni
            if output_shapefile:
                dissolved_gdf.to_file(output_shapefile, driver="ESRI Shapefile")

            return dissolved_gdf

        input_shapefile = "LidarTest.shp"
        output_shapefile = "cleaned_polygon.shp"
        self.gdf = clean_inner_polygons(lidar_shp_path, output_shapefile)
        return self.gdf


if __name__ == "__main__":
    from shapely.geometry.geo import box
    import matplotlib.pyplot as plt
    import rasterio.plot

    lidar = Lidar(
        output_path="./",
        # classification=6,
    )
    lidar.bbox = [-1.154894, 46.182639, -1.148361, 46.186820]
    # lidar_gdf = lidar.run().to_gdf()

    # fig, ax = plt.subplots(figsize=(12, 10))
    #
    # lidar_gdf.plot(
    #     cmap="hot_r",
    #     column="classification",
    #     legend=True,
    #     ax=ax,
    #     edgecolor=None,
    # )
    #
    # # Pour s'assurer que l'axe couvre bien l'extension de vos données
    # xmin, ymin, xmax, ymax = lidar_gdf.total_bounds
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_axis_off()
    #
    # plt.show()
    # exit()

    # # lidar_gdf.plot(ax=plt.gca(), edgecolor='black')
    # # plt.show()
    # lidar.to_shp(name='LidarTest')

    lidar_tif = lidar.to_tif(write_out_file=True, classification_list=[3, 4, 5, 9])

    # Lire les données et les afficher avec rasterio.plot
    with lidar_tif.open() as src:
        fig, ax = plt.subplots(figsize=(8, 8))
        rasterio.plot.show(src, ax=ax, title="Lidar CDSM")
        plt.show()

    cleaned_polygon_df = lidar.process_polygonize(path_file_csdm="./cdsm.tif")

    fig, ax = plt.subplots(figsize=(12, 10))

    cleaned_polygon_df.plot(
        cmap="hot_r",
        column="class",
        legend=True,
        ax=ax,
        edgecolor=None,
    )

    # Pour s'assurer que l'axe couvre bien l'extension de vos données
    xmin, ymin, xmax, ymax = cleaned_polygon_df.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Optionnel : masquer les axes pour un rendu plus "cartographique"
    ax.set_axis_off()

    plt.show()
