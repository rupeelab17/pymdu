import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.demos.Technoforum import Technoforum
from pymdu.geometric.Reshape import (
    trees_to_polygon,
    union_trees_buildings,
    shadows_on_ground,
)
from pymdu.geometric.SkyFactor import SkyFactor


class ShadowDetailed(GeoCore):
    """
    ===
    Classe qui permet
    - de calculer les facteurs de vue dans chaque ombre
    ===
    """

    def __init__(
        self,
        buildings: gpd.GeoDataFrame = Technoforum().buildings(),
        trees: gpd.GeoDataFrame = Technoforum().trees(),
        shadows: gpd.GeoDataFrame = Technoforum().shadows(),
        *args,
        **kwargs,
    ):
        self.buildings = buildings
        self.trees = trees_to_polygon(trees)
        self.shadows = shadows
        self.buildings_and_trees = union_trees_buildings(buildings, trees)
        self.ground_shaded = shadows_on_ground(self.buildings_and_trees, self.shadows)

    def run_all_view_factor_caculation(self):
        view_factor = {}
        for shader, name in zip(
            [self.buildings, self.trees, self.buildings_and_trees],
            ['bld', 'trees', 'bld&trees'],
        ):
            view_factor[name] = SkyFactor(shader).run()
            print(f'view factor on {name} > Done')
        with open(os.path.join(TEMP_PATH, 'viewfactors.pickle'), 'wb') as handle:
            pickle.dump(view_factor, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return view_factor

    def plot(self):
        _, basemap = plt.subplots(figsize=(10, 10))
        self.ground_shaded.plot(ax=basemap, color='grey', alpha=0.2)
        plt.show()
