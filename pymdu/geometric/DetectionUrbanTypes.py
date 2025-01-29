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
import warnings

import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import momepy
import osmnx
import pandas
from clustergram import Clustergram
from shapely import box

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.geometric import Building


class DetectionUrbanTypes(GeoCore):
    def __init__(self, output_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.DetectionUrbanTypes as DetectionUrbanTypes

            plt.clf()  # markdown-exec: hide
            detection = DetectionUrbanTypes(output_path='./')
            detection.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            detection = detection.run()
            gdf = detection.to_gdf()
            gdf.plot(
                ax=plt.gca(),
                column='cluster',
                categorical=True,
                figsize=(16, 16),
                legend=True,
            )
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

        super().__init__()

    def run(self, nbr_cluster: int = 4):
        local_crs = self.epsg
        bbox = self.bbox
        gdf_project = gpd.GeoDataFrame(
            gpd.GeoSeries(box(bbox[0], bbox[1], bbox[2], bbox[3])),
            columns=["geometry"],
            crs="epsg:4326",
        )
        gdf_project = gdf_project.to_crs(crs=4326)
        gdf_project = gdf_project.scale(xfact=1.15, yfact=1.15)

        envelope_polygon = gdf_project.envelope.bounds
        bbox = envelope_polygon.values[0]
        bbox_final = [bbox[0], bbox[1], bbox[2], bbox[3]]

        envelope_polygon = box(
            bbox_final[0], bbox_final[1], bbox_final[2], bbox_final[3]
        )

        osmnx.config(log_console=True, overpass_endpoint="https://overpass-api.de/api")

        custom_filters = (
            '["area"!~"yes"]["highway"~"footway|pedestrian|cycleway"]["foot"!~"no"]["service"!~"private"]{}'
        ).format(osmnx.settings.default_access)

        osm_graph = osmnx.graph_from_polygon(
            envelope_polygon,
            network_type="drive",
            truncate_by_edge=True,
            simplify=True,
            custom_filter=custom_filters,
        )

        osm_graph = osmnx.project_graph(osm_graph, to_crs=local_crs)
        streets = osmnx.graph_to_gdfs(
            osm_graph,
            nodes=False,
            edges=True,
            node_geometry=False,
            fill_edge_geometry=True,
        )
        # print(streets.head())
        streets = momepy.remove_false_nodes(streets)
        streets = streets[["geometry"]]
        streets["nID"] = range(len(streets))

        buildings = Building()
        buildings.bbox = bbox_final
        buildings = buildings.run().to_gdf()
        buildings["uID"] = range(len(buildings))

        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        buildings = buildings[
            [
                column
                for column in buildings.columns
                if not is_datetime(buildings[column])
            ]
        ]

        # checkMultiPolygon = "MultiPolygon" in list(buildings["geometry"].geom_type)
        # if checkMultiPolygon:
        #     buildings = BasicFunctions.drop_z(buildings)

        buildings = buildings.to_crs(local_crs)
        limit = momepy.buffered_limit(buildings, 100)

        tessellation = momepy.Tessellation(
            buildings, "uID", limit, verbose=False, segment=1
        )
        tessellation = tessellation.tessellation
        buildings = buildings.sjoin_nearest(streets, max_distance=1000, how="left")
        buildings = buildings.drop_duplicates("uID").drop(columns="index_right")
        tessellation = tessellation.merge(
            buildings[["uID", "nID"]], on="uID", how="left"
        )
        # Dimensions
        buildings["area"] = buildings.area
        tessellation["area"] = tessellation.area
        streets["length"] = streets.length

        # Shape
        buildings["eri"] = momepy.EquivalentRectangularIndex(buildings).series
        buildings["elongation"] = momepy.Elongation(buildings).series
        tessellation["convexity"] = momepy.Convexity(tessellation).series
        streets["linearity"] = momepy.Linearity(streets).series

        # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        #
        # buildings.plot("eri", ax=ax[0], scheme="natural_breaks", legend=True)
        # buildings.plot("elongation", ax=ax[1], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # plt.show()

        # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        #
        # tessellation.plot("convexity", ax=ax[0], scheme="natural_breaks", legend=True)
        # streets.plot("linearity", ax=ax[1], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # plt.show()

        # Spatial distribution
        buildings["shared_walls"] = momepy.SharedWallsRatio(buildings).series
        # buildings.plot("shared_walls", figsize=(12, 12), scheme="natural_breaks", legend=True).set_axis_off()
        queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(
            tessellation, ids="uID", silence_warnings=True
        )

        tessellation["neighbors"] = momepy.Neighbors(
            tessellation, queen_1, "uID", weighted=True, verbose=False
        ).series
        tessellation["covered_area"] = momepy.CoveredArea(
            tessellation, queen_1, "uID", verbose=False
        ).series

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            buildings["neighbor_distance"] = momepy.NeighborDistance(
                buildings, queen_1, "uID", verbose=False
            ).series
        #
        # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        #
        # buildings.plot("neighbor_distance", ax=ax[0], scheme="natural_breaks", legend=True)
        # tessellation.plot("covered_area", ax=ax[1], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # plt.show()

        queen_3 = momepy.sw_high(k=3, weights=queen_1)
        buildings_q1 = libpysal.weights.contiguity.Queen.from_dataframe(
            buildings, silence_warnings=True
        )

        buildings["interbuilding_distance"] = momepy.MeanInterbuildingDistance(
            buildings, queen_1, "uID", queen_3, verbose=False
        ).series
        buildings["adjacency"] = momepy.BuildingAdjacency(
            buildings, queen_3, "uID", buildings_q1, verbose=False
        ).series

        # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        #
        # buildings.plot("interbuilding_distance", ax=ax[0], scheme="natural_breaks", legend=True)
        # buildings.plot("adjacency", ax=ax[1], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # plt.show()

        profile = momepy.StreetProfile(streets, buildings)
        streets["width"] = profile.w
        streets["width_deviation"] = profile.wd
        streets["openness"] = profile.o

        # fig, ax = plt.subplots(1, 3, figsize=(24, 12))
        #
        # streets.plot("width", ax=ax[0], scheme="natural_breaks", legend=True)
        # streets.plot("width_deviation", ax=ax[1], scheme="natural_breaks", legend=True)
        # streets.plot("openness", ax=ax[2], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # ax[2].set_axis_off()
        # plt.show()

        # Intensity
        tessellation["car"] = momepy.AreaRatio(
            tessellation, buildings, "area", "area", "uID"
        ).series
        # tessellation.plot("car", figsize=(12, 12), vmin=0, vmax=1, legend=True).set_axis_off()

        # Connectivity
        graph = momepy.gdf_to_nx(streets)
        graph = momepy.node_degree(graph)
        graph = momepy.closeness_centrality(graph, radius=400, distance="mm_len")
        graph = momepy.meshedness(graph, radius=400, distance="mm_len")
        nodes, streets = momepy.nx_to_gdf(graph)

        # fig, ax = plt.subplots(1, 3, figsize=(24, 12))
        #
        # nodes.plot("degree", ax=ax[0], scheme="natural_breaks", legend=True, markersize=1)
        # nodes.plot("closeness", ax=ax[1], scheme="natural_breaks", legend=True, markersize=1,
        #            legend_kwds={"fmt": "{:.6f}"})
        # nodes.plot("meshedness", ax=ax[2], scheme="natural_breaks", legend=True, markersize=1)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # ax[2].set_axis_off()
        # plt.show()

        buildings["nodeID"] = momepy.get_node_id(
            buildings, nodes, streets, "nodeID", "nID"
        )
        merged = tessellation.merge(
            buildings.drop(columns=["nID", "geometry"]), on="uID"
        )
        merged = merged.merge(streets.drop(columns="geometry"), on="nID", how="left")
        merged = merged.merge(nodes.drop(columns="geometry"), on="nodeID", how="left")

        # Understanding the context
        percentiles = []
        for column in merged.columns.drop(
            ["uID", "nodeID", "nID", "mm_len", "node_start", "node_end", "geometry"]
        ):
            try:
                perc = momepy.Percentiles(
                    merged, column, queen_1, "uID", verbose=False
                ).frame
                perc.columns = [f"{column}_" + str(x) for x in perc.columns]
                percentiles.append(perc)
            except Exception as e:
                print("percentiles =>", e)

        percentiles_joined = pandas.concat(percentiles, axis=1)
        percentiles_joined.head()

        # fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        #
        # tessellation.plot("convexity", ax=ax[0], scheme="natural_breaks", legend=True)
        # merged.plot(percentiles_joined['convexity_50'].values, ax=ax[1], scheme="natural_breaks", legend=True)
        #
        # ax[0].set_axis_off()
        # ax[1].set_axis_off()
        # plt.show()

        # Clustering
        standardized = (
            percentiles_joined - percentiles_joined.mean()
        ) / percentiles_joined.std()
        standardized.head()

        cgram = Clustergram(range(1, 12), n_init=10, random_state=42)
        cgram.fit(standardized.fillna(0))

        # show(cgram.bokeh())
        cgram.labels.head()
        merged["cluster"] = cgram.labels[nbr_cluster].values
        urban_types = buildings[["geometry", "uID"]].merge(
            merged[["uID", "cluster"]], on="uID"
        )
        # urban_types.plot("cluster", categorical=True, figsize=(16, 16), legend=True).set_axis_off()

        # Générer une palette de couleurs unique pour chaque valeur de cluster
        unique_clusters = urban_types["cluster"].unique()
        # Convertir les couleurs en format hexadécimal
        color_map = {
            cluster: "#" + "".join(f"{int(c * 255):02x}" for c in plt.cm.tab10(i)[:3])
            for i, cluster in enumerate(unique_clusters)
        }

        # Ajouter une colonne 'color' avec la couleur correspondante
        urban_types["color"] = urban_types["cluster"].map(color_map)

        self.gdf = urban_types

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = "detection"):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f"{os.path.join(self.output_path, name)}.gpkg", driver="GPKG")


if __name__ == "__main__":

    detection = DetectionUrbanTypes()
    detection.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    detection = detection.run()
    detection_gdf = detection.to_gdf()
    detection_gdf.plot("cluster", categorical=True, figsize=(16, 16), legend=True)
    plt.show()
