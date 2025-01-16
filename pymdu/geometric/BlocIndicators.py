# # ******************************************************************************
# #  This file is part of pymdu.                                                 *
# #                                                                              *
# #  pymdu is free software: you can redistribute it and/or modify               *
# #  it under the terms of the GNU General Public License as published by        *
# #  the Free Software Foundation, either version 3 of the License, or           *
# #  (at your option) any later version.                                         *
# #                                                                              *
# #  pymdu is distributed in the hope that it will be useful,                    *
# #  but WITHOUT ANY WARRANTY; without even the implied warranty of              *
# #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
# #  GNU General Public License for more details.                                *
# #                                                                              *
# #  You should have received a copy of the GNU General Public License           *
# #  along with pymdu.  If not, see <https://www.gnu.org/licenses/>.             *
# # ******************************************************************************
# import os
#
# import numpy as np
# import shapely
# from geopandas import clip, GeoDataFrame
# from tqdm import tqdm
#
# from pymdu.GeoCore import GeoCore
# from pymdu.collect.GlobalVariables import TEMP_PATH
# from pymdu.demos.Technoforum import Technoforum
#
#
# class BlocIndicators(GeoCore):
#     def __init__(
#         self,
#         blocs_gdf: GeoDataFrame = None,
#         buildings_gdf: GeoDataFrame = None,
#         vegetation_gdf: GeoDataFrame = Technoforum.vegetation(),
#     ):
#         self.blocs = blocs_gdf.to_crs(self._epsg)
#         self.blocs = self.blocs[self.blocs['BUILDING_TOTAL_FRACTION'] > 0]
#         self.blocs = self.blocs.reset_index()
#         self.blocs['gid'] = self.blocs.index
#         self.vegetation = vegetation_gdf.to_crs(2154)
#         self.buildings = buildings_gdf.to_crs(2154)
#
#     def canyon_width(self) -> GeoDataFrame:
#         canWidth = []
#         for x, y, z in zip(
#             self.blocs['AVG_HEIGHT_ROOF_AREA_WEIGHTED'],
#             self.blocs['BUILDING_TOTAL_FRACTION'],
#             self.blocs['FREE_EXTERNAL_FACADE_DENSITY'],
#         ):
#             try:
#                 bldWidth = 4 * x * y / z
#                 d = bldWidth / np.sqrt(y)
#                 canWidth.append(d - bldWidth)
#             except:
#                 canWidth.append(999)
#         self.blocs['CANYON_WIDTH'] = canWidth
#         return self.blocs
#
#     def characteristic_length(self) -> GeoDataFrame:
#         liste_charac_length = []
#         for mybloc in tqdm(self.blocs['geometry']):
#             # gdf = gpd.GeoDataFrame(index=[0], crs='epsg:2154', geometry=[mybloc])
#             # gdf = gdf.to_crs(2154)
#             liste_charac_length.append(shapely.geometry.box(*mybloc.bounds).length)
#         self.blocs['charlength'] = liste_charac_length
#         return self.blocs
#
#     def uwg_rename(self) -> GeoDataFrame:
#         self.blocs = self.blocs.rename(
#             columns={
#                 'BUILDING_TOTAL_FRACTION': 'blddensity',
#                 'AVG_HEIGHT_ROOF_AREA_WEIGHTED': 'bldheight',
#                 'FREE_EXTERNAL_FACADE_DENSITY': 'vertohor',
#             }
#         )
#         self.blocs.to_csv(os.path.join(TEMP_PATH, '/uwg_blocs.csv'))
#         return self.blocs
#
#     def bloc_building(self) -> GeoDataFrame:
#         liste_batiments = []
#         for bloc_lcz in tqdm(self.blocs['geometry']):
#             intersection = clip(gdf=self.buildings, mask=bloc_lcz)
#             liste_batiments.append(intersection['id'].tolist())
#             print('intersection +++>', intersection)
#         self.blocs['id_bdtopo'] = liste_batiments
#         return self.blocs
#
#     def bloc_vegetation(self) -> GeoDataFrame:
#         liste_percent_veg = []
#         for bloc_lcz in tqdm(self.blocs['geometry']):
#             # gdf = gpd.GeoDataFrame(index=[0], crs='epsg:2154', geometry=[mybloc])
#             # gdf = gdf.to_crs(2154)
#             veg = clip(gdf=self.vegetation, mask=bloc_lcz)
#             percent_veg = veg.area.sum() / bloc_lcz.area
#             liste_percent_veg.append(percent_veg.tolist())
#         self.blocs['grasscover'] = liste_percent_veg
#         return self.blocs
