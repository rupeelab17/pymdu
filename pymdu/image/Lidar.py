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
from pyproj import Transformer
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from shapely import box
from shapely.geometry import Point
from shapely.geometry.multipoint import MultiPoint

from pymdu.GeoCore import GeoCore
from pymdu._typing import FilePath
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.image import rasterize

from scipy.stats import binned_statistic_2d

from rasterio.transform import from_origin, xy
import rasterio.features
from shapely.geometry import shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import gc
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
            output_path: str | None  = None,
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

    def load_lidar_points(self, laz_urls):
        """
        Download each LAZ file into memory and load its LiDAR points.

        Returns:
          NumPy array of shape (N, 4) with columns [x, y, z, classification].
        """
        all_points = []
        for url in laz_urls:
            print("Downloading LAZ file from:", url)
            r = requests.get(url)
            if r.status_code != 200:
                print("Error downloading:", url)
                continue
            file_obj = io.BytesIO(r.content)
            try:
                las = laspy.read(file_obj)
            except Exception as e:
                print("Error reading LAZ file:", e)
                continue
            pts = np.vstack((las.x, las.y, las.z, las.classification)).T
            all_points.append(pts)
        if not all_points:
            raise Exception("No LiDAR points were loaded.")
        return np.concatenate(all_points, axis=0)

    def process_lidar_points(self,points, bbox, classification_list = [3, 4, 5], resolution=1.0):
        """
        Computes DSM, DTM, and CHM from LiDAR points.

        Arguments:
            points: Nx4 array of (x, y, z, classification)
            bbox: [x_min, x_max, y_min, y_max] in meters (projected CRS)
            resolution: Grid resolution in meters

        Returns:
            DSM, DTM, CHM (2D arrays)
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        classification = points[:, 3]

        x_min, x_max, y_min, y_max = bbox

        # Apply spatial mask: keep only points within the bbox.
        spatial_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x = x[spatial_mask]
        y = y[spatial_mask]
        z = z[spatial_mask]
        classification = classification[spatial_mask]

        # For DSM, use only vegetation points defined in classification_list.
        veg_mask = np.isin(classification, classification_list)
        if not np.any(veg_mask):
            raise ValueError("No vegetation points found within the bounding box for the provided classification_list.")

        # Define grid extents

        width = int((x_max - x_min) / resolution)
        height = int((y_max - y_min) / resolution)

        # Bin edges
        x_bins = np.linspace(x_min, x_max, width + 1)
        y_bins = np.linspace(y_min, y_max, height + 1)

        # Compute DSM: maximum elevation from vegetation returns.
        dsm, _, _, _ = binned_statistic_2d(x[veg_mask], y[veg_mask], z[veg_mask],
                                           statistic='max', bins=[x_bins, y_bins])

        # Compute DTM (min elevation for ground points)
        ground_mask = (classification == 2)
        if np.any(ground_mask):
            dtm, _, _, _ = binned_statistic_2d(x[ground_mask], y[ground_mask], z[ground_mask],
                                               statistic='min', bins=[x_bins, y_bins])
        else:
            dtm = np.full_like(dsm, np.nan)

        # Compute CHM (Canopy Height Model)
        chm = dsm - dtm
        chm[chm < 0] = 0  # Remove negative heights due to errors

        return dsm, dtm, chm

    def extract_tree_crowns(self, chm, transform, lidar_points=None,
                            min_tree_height = 2.0, min_distance = 5,
                            crown_shp="tree_crowns.shp",
                            tops_shp="tree_tops.shp",
                            lidar_shp="tree_lidar_points.shp"):
        """
        Extracts tree crown polygons and tree top points from a CHM using watershed segmentation.
        Optionally, it also filters LiDAR points (classes 3,4,5) and saves them.

        Parameters:
          chm : 2D numpy array
              Canopy Height Model (CHM) raster (oriented north-up).
          transform : Affine
              Affine transform associated with the CHM (e.g., from_origin).
          lidar_points : np.ndarray, optional
              LiDAR points array with columns [x, y, z, classification]. If provided, only points
              with classification 3,4,5 will be saved.
          min_tree_height : float, default 2.0
              Minimum canopy height (in meters) to consider a pixel as part of a tree.
          min_distance : int, default 5
              Minimum number of pixels separating local maxima (tree tops).
          crown_shp : str, default "tree_crowns.shp"
              Output filename for tree crown polygons shapefile.
          tops_shp : str, default "tree_tops.shp"
              Output filename for tree top points shapefile.
          lidar_shp : str, default "tree_lidar_points.shp"
              Output filename for filtered LiDAR points shapefile (if lidar_points is provided).

        Returns:
          gdf_crowns : GeoDataFrame
              GeoDataFrame containing tree crown polygons.
          gdf_tops : GeoDataFrame
              GeoDataFrame containing tree top points.
          gdf_lidar : GeoDataFrame or None
              GeoDataFrame containing filtered LiDAR points (if lidar_points was provided); otherwise None.
        """

        mask = chm >= min_tree_height


        # Note: peak_local_max returns (row, col) coordinates.
        local_max_coords = peak_local_max(chm, min_distance = min_distance,
                                          threshold_abs = min_tree_height)
        print(f"Detected {len(local_max_coords)} tree top candidates.")


        markers = np.zeros_like(chm, dtype=np.int32)
        for idx, (row, col) in enumerate(local_max_coords, start=1):
            markers[row, col] = idx


        segmentation = watershed(-chm, markers, mask=mask)

        crown_polygons = []
        crown_ids = []
        tree_heights = []

        for geom, val in rasterio.features.shapes(segmentation.astype(np.int32),
                                                  mask=(segmentation > 0),
                                                  transform=transform):
            if val == 0:
                continue
            poly = shape(geom)
            crown_polygons.append(poly)
            crown_ids.append(val)

            # Estimate tree height from CHM at detected peak
            tree_heights.append(chm[segmentation == val].max())

        print(f"Extracted {len(crown_polygons)} crown polygons.")

        tree_top_points = []
        tree_ids_tops = []
        estimated_trunk_heights = []
        estimated_diameters = []

        min_diameter = 1.0
        for idx, (row, col) in enumerate(local_max_coords, start=1):
            # xy returns (x, y) from row, col based on the transform.
            x, y = xy(transform, row, col)
            tree_top_points.append(Point(x, y))
            tree_ids_tops.append(idx)

            # Get height from crown segmentation
            tree_height = tree_heights[idx - 1]

            # Estimate trunk height
            trunk_height = tree_height * 0.3
            estimated_trunk_heights.append(trunk_height)

            # Estimate diameter (diamètre à hauteur de poitrine (DBH))
            dbh = max(0.1 * (tree_height ** 1.2), min_diameter)
            estimated_diameters.append(dbh)


        # Create GeoDataFrame for tree crowns
        gdf_crowns = gpd.GeoDataFrame({
            'tree_id': crown_ids,
            'tree_height': tree_heights,
            'trunk_height': estimated_trunk_heights,
            'diameter': estimated_diameters
        }, geometry=crown_polygons, crs="EPSG:2154")

        gdf_crowns.to_file(crown_shp)
        print(f"Tree crowns saved to '{crown_shp}'.")

        # Create GeoDataFrame for tree tops
        gdf_tops = gpd.GeoDataFrame({
            'tree_id': tree_ids_tops,
            'type': 1,
            'height': tree_heights,
            'trunk zone': estimated_trunk_heights,
            'diameter': estimated_diameters
        }, geometry=tree_top_points, crs="EPSG:2154")

        gdf_tops.to_file(tops_shp)
        print(f"Tree tops saved to '{tops_shp}'.")

        # 9. (Optional) Filter and save LiDAR points for tree vegetation classes 3,4,5.
        gdf_lidar = None
        # if lidar_points is not None:
        #     tree_classes = [3, 4, 5]
        #     mask_tree = np.isin(lidar_points[:, 3], tree_classes)
        #     tree_points = lidar_points[mask_tree]
        #     # Create point geometries from LiDAR x and y.
        #     tree_point_geoms = [Point(x, y) for x, y in zip(tree_points[:, 0], tree_points[:, 1])]
        #     gdf_lidar = gpd.GeoDataFrame({'z': tree_points[:, 2]}, geometry=tree_point_geoms, crs="EPSG:2154")
        #     gdf_lidar.to_file(lidar_shp)
        #     print(f"Filtered LiDAR tree points (classes 3,4,5) saved to '{lidar_shp}'.")

        return gdf_crowns, gdf_tops, gdf_lidar

    def run_trees(self):
        min_x, min_y, max_x, max_y, laz_urls = self._get_lidar_points()
        points = self.load_lidar_points(laz_urls)
        # Define transformation from EPSG:4326 (WGS84) to EPSG:2154 (Lambert 93)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

        # Convert bbox (lon_min, lat_min, lon_max, lat_max)
        x_min, y_min = transformer.transform(self._bbox[0], self._bbox[1])
        x_max, y_max = transformer.transform(self._bbox[2], self._bbox[3])

        # Projected bbox
        bbox_projected = [x_min, x_max, y_min, y_max]
        print("Projected BBOX (EPSG:2154):", bbox_projected)

        # (d) Compute DSM, DTM, CHM
        resolution = 1.0
        DSM, DTM, CHM = self.process_lidar_points(points, bbox_projected)

        DSM_adjusted = np.flipud(DSM.T)
        DTM_adjusted = np.flipud(DTM.T)
        CHM_adjusted = np.flipud(CHM.T)

        transform = from_origin(x_min, y_max, resolution, resolution)

        for raster, name in zip([DSM_adjusted, DTM_adjusted, CHM_adjusted], ["DSM.tif", "DTM.tif", "CHM.tif"]):
            with rasterio.open(
                    name,
                    "w",
                    driver="GTiff",
                    height=raster.shape[0],
                    width=raster.shape[1],
                    count=1,
                    dtype=raster.dtype,
                    crs="EPSG:2154",
                    transform=transform,
            ) as dst:
                dst.write(raster, 1)
            print(f"{name} saved successfully.")

        # free up memory
        del points, laz_urls
        gc.collect()

        gdf_crowns, gdf_tops, gdf_lidar = self.extract_tree_crowns(CHM_adjusted, transform, lidar_points = None,
                                                              min_tree_height=2.0, min_distance=5)
        return gdf_tops

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

        # Obtenir la géotransformation (pour convertir coordonnées monde → pixel)
        gt = im.GetGeoTransform()

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
        layer = vector.CreateLayer("", spatial_ref, geom_type=ogr.wkbPolygon)

        # cre_ate field to write NDVI values:
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
        # gdal.Polygonize(
        #     srcband_classification,
        #     mask,
        #     layer,
        #     1,
        #     options=[],
        #     callback=gdal.TermProgress_nocb,
        # )

        # =============================================
        # 4. Définir une fonction utilitaire pour convertir les coordonnées monde en coordonnées pixels
        def world_to_pixel(geo_transform, x, y):
            originX = geo_transform[0]
            originY = geo_transform[3]
            pixelWidth = geo_transform[1]
            pixelHeight = geo_transform[5]
            px = int((x - originX) / pixelWidth)
            py = int((y - originY) / pixelHeight)
            return px, py

        # =============================================
        # 5. Pour chaque polygone, lire la valeur de la deuxième bande au niveau du centroïde
        layer.ResetReading()
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            centroid = geom.Centroid()
            x = centroid.GetX()
            y = centroid.GetY()
            # Convertir les coordonnées du centroïde en indices de pixel
            px, py = world_to_pixel(gt, x, y)
            # Lire la valeur dans la deuxième bande (taille 1x1)
            classification_array = srcband_classification.ReadAsArray(px, py, 1, 1)

            if str(classification_array[0][0]) != "nan":
                classification_value = int(classification_array[0][0])
            else:
                classification_value = 3
            feature.SetField("class", classification_value)
            layer.SetFeature(feature)

        # =============================================
        # 6. Finaliser et fermer les ressources
        layer.SyncToDisk()
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

        output_shapefile = "cleaned_polygon.shp"
        self.gdf = clean_inner_polygons(
            lidar_shp_path, output_shapefile=output_shapefile
        )

        self.gdf = gpd.read_file(lidar_shp_path, driver="ESRI Shapefile")
        self.gdf.to_file("LidarTest.shp", driver="ESRI Shapefile")

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

    gdf_tops = lidar.run_trees()

    # Comment AJ from here
    lidar_tif = lidar.to_tif(write_out_file=True, classification_list=[3, 4, 5, 9])

    # Lire les données et les afficher avec rasterio.plot
    with lidar_tif.open() as src:
        fig, ax = plt.subplots(figsize=(8, 8))
        rasterio.plot.show(src, ax=ax, title="Lidar CDSM")
        plt.show()

    cleaned_polygon_df = lidar.process_polygonize(path_file_csdm="./cdsm.tif")

    print(cleaned_polygon_df.head())

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
