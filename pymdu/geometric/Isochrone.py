import os

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import rasterio as rio
from edsger.path import Dijkstra

# from networkit.distance import Dijkstra
from pyproj import transform
from shapely import concave_hull
from shapely.geometry import MultiPoint, Point
from shapely.ops import transform
from sklearn.neighbors import KDTree

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.geometric import Dem


class Isochrone(GeoCore):
    # https://aetperf.github.io/2024/03/01/Calculating_walking_isochrones_with_Python.html
    OUTPUT_NODES_FP = "./nodes_pedestrian_network.GeoJSON"
    OUTPUT_EDGES_FP = "./edges_pedestrian_network.GeoJSON"

    def __init__(
        self,
        output_path: str | None = None,
        poi: dict = None,
        write_geojson: bool = False,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            poi (dict): The point of interest.
            write_geojson (bool): If True, the processed data will be written to a GeoJSON file.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import contextily as cx
            import pymdu.geometric.Isochrone as Isochrone

            plt.clf()  # markdown-exec: hide
            poi = {'lon': -1.1491, 'lat': 46.1849}
            iso = Isochrone(poi=poi)
            iso.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            iso.run()
            iso_in_gdf, iso_out_gdf = iso.to_gdf()
            t = 15  # 15 minutes
            ax = iso_in_gdf.loc[[t]].plot(
                alpha=0.25, color='black', figsize=(8, 8), label='in'
            )
            ax = iso_out_gdf.loc[[t]].plot(alpha=0.25, color='b', ax=ax, label='out')
            cx.add_basemap(
                ax,
                source=cx.providers.CartoDB.VoyagerNoLabels,
                crs=iso_out_gdf.crs.to_string(),
            )
            _ = plt.plot(iso.poi_lam93.x, iso.poi_lam93.y, marker='o', color='red', alpha=1)
            _ = plt.axis('off')
            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
             * For module TODOs
        """
        self.output_path = output_path if output_path else TEMP_PATH
        self.write_geojson = write_geojson
        if poi is None:
            poi = {"lon": -1.152704, "lat": 46.181627}
        self.poi_wgs84 = Point([poi["lon"], poi["lat"]])
        self.poi_lam93 = None

    def run(self):
        nodes, edges = self.create_routable_pedestrian_network_with_elevation()
        nodes = nodes.set_index("id")
        edges = edges[["tail", "head", "travel_time_s"]]

        wgs84 = pyproj.CRS("EPSG:4326")
        lam93 = pyproj.CRS("EPSG:2154")  # Lambert 93
        project = pyproj.Transformer.from_crs(wgs84, lam93, always_xy=True).transform
        self.poi_lam93 = transform(project, self.poi_wgs84)

        nodes = nodes.to_crs("EPSG:2154")
        nodes["x_2154"] = nodes.geometry.x
        nodes["y_2154"] = nodes.geometry.y
        nodes = nodes.to_crs("EPSG:4326")
        X = nodes[["x_2154", "y_2154"]].values
        tree = KDTree(X)
        x, y = self.poi_lam93.x, self.poi_lam93.y
        n_connectors = 5
        dist, ind = tree.query([[x, y]], k=n_connectors)
        poi_index = nodes.index.max() + 1
        v_kms = 5.0
        v_ms = v_kms * 1000.0 / 3600.0

        connectors = pd.DataFrame(
            data={"tail": n_connectors * [poi_index], "head": ind[0], "length": dist[0]}
        )
        connectors["travel_time_s"] = connectors["length"] / v_ms
        connectors = connectors.drop("length", axis=1)

        connectors_reverse = connectors.copy(deep=True)
        connectors_reverse[["tail", "head"]] = connectors_reverse[["head", "tail"]]
        connectors = pd.concat([connectors, connectors_reverse], axis=0)
        graph_edges = pd.concat([edges, connectors], axis=0)
        graph_edges[["tail", "head"]] = graph_edges[["tail", "head"]].astype(np.uint32)

        sp_out = Dijkstra(
            graph_edges[["tail", "head", "travel_time_s"]],
            weight="travel_time_s",
            orientation="out",
            check_edges=False,
        )
        tt_out = sp_out.run(vertex_idx=poi_index, return_inf=True)

        sp_in = Dijkstra(
            graph_edges[["tail", "head", "travel_time_s"]],
            weight="travel_time_s",
            orientation="in",
            check_edges=False,
        )
        tt_in = sp_in.run(vertex_idx=int(poi_index), return_inf=True)

        # 5, 10 and 15 minutes “outward” isochrones.
        steps = np.arange(5, 16, 5)
        coords = nodes[["x_2154", "y_2154"]].copy(deep=True)
        coords["tt_out"] = tt_out[:-1]  # last index corresponds to POI
        coords["tt_in"] = tt_in[:-1]

        isochrones_out = self.create_isochrones(
            coords, tt_col="tt_out", x_col="x_2154", y_col="y_2154", steps_m=steps
        )

        # ax = isochrones_out.plot(alpha=0.25, color="b")
        # cx.add_basemap(ax, source=xyz.CartoDB.VoyagerNoLabels, crs=isochrones_out.crs.to_string(), alpha=0.8, )
        # _ = plt.plot(poi_lam93.x, poi_lam93.y, "bo")
        # _ = plt.axis("off")

        isochrones_in = self.create_isochrones(
            coords,
            tt_col="tt_in",
            x_col="x_2154",
            y_col="y_2154",
            steps_m=steps,
        )

        # ax = isochrones_in.plot(alpha=0.25, color="r", figsize=(8, 8))
        # cx.add_basemap(ax, source=xyz.CartoDB.VoyagerNoLabels, crs=isochrones_out.crs.to_string(), alpha=0.8, )
        # _ = plt.plot(poi_lam93.x, poi_lam93.y, "ro")
        # _ = plt.axis("off")

        # t = 15  # 15 minutes
        # ax = isochrones_in.loc[[t]].plot(alpha=0.25, color="r", figsize=(8, 8), label="in")
        # ax = isochrones_out.loc[[t]].plot(alpha=0.25, color="b", ax=ax, label="out")
        # cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerNoLabels, crs=isochrones_out.crs.to_string())
        # _ = plt.plot(poi_lam93.x, poi_lam93.y, marker="o", color="grey", alpha=0.7)
        # _ = plt.axis("off")

        self.gdf = isochrones_in, isochrones_out
        return self

    def create_isochrones(
        self,
        coords,
        tt_col="tt_out",
        x_col="x_2154",
        y_col="y_2154",
        steps_m=[10, 20, 30],
        ratio=0.3,
        allow_holes=False,
    ):
        """
        Create isochrones from travel time data.

        Parameters
        ----------
        coords : pandas.DataFrame
            DataFrame containing coordinates and travel time data.
        tt_col : str, optional
            Name of the column containing travel time data.
            The default is "tt_out".
        x_col : str, optional
            Name of the column containing x-coordinates.
            The default is "x_2154".
        y_col : str, optional
            Name of the column containing y-coordinates.
            The default is "y_2154".
        steps_m : list of int, optional
            List of travel times in minutes for which to create isochrones.
            The default is [10, 20, 30].
        ratio : float, optional
            Ratio of concavity for the isochrones.
            The default is 0.3.
        allow_holes : bool, optional
            Whether to allow holes in the isochrones.
            The default is False.

        Returns
        -------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the isochrones as polygons.
        """

        isochrones = {}
        for step in steps_m:
            t_s = 60.0 * step
            points = MultiPoint(
                coords.loc[coords[tt_col] <= t_s, [x_col, y_col]].values
            )
            isochrones[step] = concave_hull(
                points, allow_holes=allow_holes, ratio=ratio
            )

        df = pd.DataFrame.from_dict(isochrones, orient="index", columns=["geometry"])
        gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs="EPSG:2154")
        return gdf

    def create_routable_pedestrian_network_with_elevation(self):
        # create the graph
        self.__create_graph()
        nodes, edges = self.reindex_nodes(
            node_id_col="osmid", tail_id_col="tail", head_id_col="head"
        )

        print(nodes.head(10))
        print(edges.head(10))
        edges = pd.merge(
            edges,
            nodes[["z"]].rename(columns={"z": "tail_z"}),
            left_on="tail",
            right_index=True,
            how="left",
        )
        edges = pd.merge(
            edges,
            nodes[["z"]].rename(columns={"z": "head_z"}),
            left_on="head",
            right_index=True,
            how="left",
        )
        edges["slope_deg"] = edges[["tail_z", "head_z", "length"]].apply(
            self.compute_slope, raw=True, axis=1
        )
        edges["walking_speed_kmh"] = edges.slope_deg.map(
            lambda s: self.walking_speed_kmh(s)
        )
        edges["travel_time_s"] = (
            3600.0 * 1.0e-3 * edges["length"] / edges.walking_speed_kmh
        )
        # cleanup
        edges.drop(
            ["tail_z", "head_z", "length", "length", "slope_deg", "walking_speed_kmh"],
            axis=1,
            inplace=True,
        )

        if self.write_geojson:
            nodes.to_file(self.OUTPUT_NODES_FP, driver="GeoJSON", crs="EPSG:4326")
            edges.to_file(self.OUTPUT_EDGES_FP, driver="GeoJSON", crs="EPSG:4326")
        return nodes, edges

    @staticmethod
    def walking_speed_kmh(slope_deg):
        theta = np.pi * slope_deg / 180.0
        return 6.0 * np.exp(-3.5 * np.abs(np.tan(theta) + 0.05))

    def add_elevation_data_into_nodes(self):
        dem = Dem(output_path="./")
        dem.bbox = self.bbox
        ign_dem = dem.run()

        # extract point coordinates in Lambert 93
        nodes = self.nodes.to_crs("EPSG:2154")
        lon_lam93 = nodes.geometry.apply(lambda p: p.x)
        lat_lam93 = nodes.geometry.apply(lambda p: p.y)
        nodes = nodes.to_crs("EPSG:4326")
        point_coords = list(zip(lon_lam93, lat_lam93))
        dem = rio.open("DEM.tif")
        self.nodes["z"] = [x[0] for x in dem.sample(point_coords)]
        self.nodes.loc[self.nodes["z"] <= -99999, "z"] = np.nan

    def __create_graph(self):
        G = ox.graph_from_bbox(
            north=self.bbox[3],
            south=self.bbox[1],
            east=self.bbox[0],
            west=self.bbox[2],
            network_type="walk",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf = edges_gdf.loc[edges_gdf.highway != "trunk"]
        edges_gdf = edges_gdf.loc[edges_gdf.highway != "trunk_link"]
        # edge column selection and renaming
        edges = edges_gdf[["geometry"]].reset_index(drop=False)
        edges = edges.rename(columns={"u": "tail", "v": "head"})
        edges.drop("key", axis=1, inplace=True)
        # we select one direction between each pair couples
        edges["min_vert"] = edges[["tail", "head"]].min(axis=1)
        edges["max_vert"] = edges[["tail", "head"]].max(axis=1)

        edges = edges.to_crs("EPSG:2154")
        edges["length"] = edges.geometry.map(lambda g: g.length)
        edges = edges.to_crs("EPSG:4326")
        edges = edges.sort_values(by=["min_vert", "max_vert", "length"], ascending=True)
        edges = edges.drop_duplicates(subset=["min_vert", "max_vert"], keep="first")
        edges = edges.drop(["min_vert", "max_vert"], axis=1)
        edges_reverse = edges.copy(deep=True)
        edges_reverse[["tail", "head"]] = edges_reverse[["head", "tail"]]
        edges_reverse.geometry = edges_reverse.geometry.map(lambda g: g.reverse())
        edges = pd.concat((edges, edges_reverse), axis=0)
        edges = edges.sort_values(by=["tail", "head"])
        self.edges = edges.loc[edges["tail"] != edges["head"]]
        self.edges.reset_index(drop=True, inplace=True)
        self.nodes = nodes_gdf[["geometry"]].copy(deep=True)
        self.nodes = self.nodes.reset_index(drop=False)

    def reindex_nodes(
        self, node_id_col="osmid", tail_id_col="tail", head_id_col="head"
    ):
        if node_id_col == "id":
            node_id_col = "id_old"
            self.nodes = self.nodes.rename(columns={"id": node_id_col})

        assert "geometry" in self.nodes

        # reindex the nodes and update the edges
        self.nodes = self.nodes.reset_index(drop=True)
        if "id" in self.nodes.columns:
            self.nodes = self.nodes.drop("id", axis=1)
        self.nodes["id"] = self.nodes.index
        self.edges = pd.merge(
            self.edges,
            self.nodes[["id", node_id_col]],
            left_on=tail_id_col,
            right_on=node_id_col,
            how="left",
        )
        self.edges.drop([tail_id_col, node_id_col], axis=1, inplace=True)
        self.edges.rename(columns={"id": tail_id_col}, inplace=True)
        self.edges = pd.merge(
            self.edges,
            self.nodes[["id", node_id_col]],
            left_on=head_id_col,
            right_on=node_id_col,
            how="left",
        )
        self.edges.drop([head_id_col, node_id_col], axis=1, inplace=True)
        self.edges.rename(columns={"id": head_id_col}, inplace=True)

        # reorder the columns to have tail and head node vertices first
        cols = self.edges.columns
        extra_cols = [c for c in cols if c not in ["tail", "head"]]
        cols = ["tail", "head"] + extra_cols
        self.edges = self.edges[cols]

        # cleanup
        if node_id_col in self.nodes:
            self.nodes = self.nodes.drop(node_id_col, axis=1)

        self.add_elevation_data_into_nodes()
        return self.nodes, self.edges

    @staticmethod
    def compute_slope(triangle_att):
        """
        triangle_att must be [tail_z, head_z, length]
        """
        tail_z, head_z, length = triangle_att

        x = (head_z - tail_z) / length
        theta = np.arctan(x)
        theta_deg = theta * 180.0 / np.pi

        # Limits the slope angle to a maximum of 20.0 degrees and
        # a minimum of -20.0 degrees
        theta_deg = np.amin([theta_deg, 20.0])
        theta_deg = np.amax([theta_deg, -20.0])

        return theta_deg

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "isochrone"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":
    import contextily as cx
    import matplotlib.pyplot as plt

    poi = {"lon": -1.1491, "lat": 46.1849}
    iso = Isochrone(poi=poi)
    iso.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    iso.run()
    iso_in_gdf, iso_out_gdf = iso.to_gdf()
    t = 15  # 15 minutes
    ax = iso_in_gdf.loc[[t]].plot(alpha=0.25, color="black", figsize=(8, 8), label="in")
    ax = iso_out_gdf.loc[[t]].plot(alpha=0.25, color="b", ax=ax, label="out")
    cx.add_basemap(
        ax, source=cx.providers.CartoDB.VoyagerNoLabels, crs=iso_out_gdf.crs.to_string()
    )
    _ = plt.plot(iso.poi_lam93.x, iso.poi_lam93.y, marker="o", color="red", alpha=1)
    _ = plt.axis("off")
    plt.show()
