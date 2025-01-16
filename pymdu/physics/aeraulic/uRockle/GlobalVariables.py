#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:26:24 2021

@author: Jérémy Bernard, University of Gothenburg
"""

import pandas as pd
import tempfile
import os

# Wind input measurement height
Z_REF = 10
V_REF = 2.0
WIND_DIRECTION = 0
PROFILE_TYPE = 'power'
# PROFILE_TYPE = "urban"

# Option to remove any offset due to the initialisation step
REMOVE_INITIALIZATION_OFFSET = False

# If the solver should go descending order along y (does not work yet...)
DESCENDING_Y = False

# Street canyon scheme limitation (below this angle, no street canyon is created)
STREET_CANYON_ANGLE_THRESH = 0

# Temporary directory where are saved database and specific files exchanged between
# the H2Database and Python
TEMPO_DIRECTORY = tempfile.gettempdir()
INPUT_DIRECTORY = os.path.join('Resources', 'Inputs')
OUTPUT_DIRECTORY = os.path.join('Resources', 'Outputs')
BUILDING_TABLE_NAME = 'BUILDINGS'
VEGETATION_TABLE_NAME = 'VEGETATION'
CAD_TRIANGLE_NAME = 'ALL_TRIANGLES'
CAD_VEG_INTERSECTION = 'VEG_INTERSECTION'
BUILDING_FILENAME = 'buildings_3857_notnan.shp'
VEGETATION_FILENAME = ''
CAD_TRIANGLE_FILENAME = ''
CAD_VEG_INTERSECTION_FILENAME = ''
DELETE_OUTPUT_IF_EXISTS = True
# VEGETATION_FILENAME = "vegetation.shp"
# CAD_TRIANGLE_FILENAME = "AllTriangles.shp"
# CAD_VEG_INTERSECTION_FILENAME = "treesIntersection.shp"
TEMPO_HORIZ_WIND_FILE = 'tempo_horiz_wind.csv'
# Output files
OUTPUT_FILENAME = 'UROCK_OUTPUT'
OUTPUT_RASTER_EXTENSION = '.GTiff'
OUTPUT_VECTOR_EXTENSION = '.GeoJSON'
OUTPUT_NETCDF_EXTENSION = '.nc'
VECTOR_STYLE_FILENAME = 'vectorStyle.qml'

# List of vertical height for the output horizontal wind
Z_OUT = [1.5, 3, 6]

# Output field names (Horizontal wind speed, wind direction and vertical wind speed)
HORIZ_WIND_SPEED = 'HWS'
HORIZ_WIND_DIRECTION = 'HWD'
VERT_WIND_SPEED = 'VWS'
WIND_SPEED = 'WS'

# Informations to set the DB used for geographical calculations
INSTANCE_NAME = 'myDbH2'
INSTANCE_ID = 'sa'
INSTANCE_PASS = 'sa'
NEW_DB = True

# Where to save the current JAVA path
JAVA_PATH_FILENAME = 'JavaPath.csv'

# If debug is True, keep intermediate tables (within each process) and save
# intermediate tables (such as Röckle zones) as GIS file
DEBUG = False
ONLY_INITIALIZATION = True
SAVE_ROCKLE_ZONES = False
MAX_ITERATIONS = 500  # Based on QUIC-URB default values (2021)
THRESHOLD_ITERATIONS = 1e-4  # Based on QUIC-URB default values (2021)

# Note that the number of points of an ellipse is only used to identify whether
# the upper or lower part of an ellipse should be used (fro displacement zones),
# the number of points used to create an ellipse can not be chosen yet
# Need to create this variable in H2GIS
# https://github.com/locationtech/jts/blob/9d4097312d68cb8f9ae591bec69ce3b403e41e98/modules/core/uRockle/main/java/org/locationtech/jts/util/GeometricShapeFactory.java#L101
NPOINTS_ELLIPSE = 100
MESH_SIZE = 5
DZ = 5
ALONG_WIND_ZONE_EXTEND = 500
CROSS_WIND_ZONE_EXTEND = 500
VERTICAL_EXTEND = 20
CAV_N_WAKE_FACADE_NPOINTS = 10

# The "perpendicular vortex scheme" for rooftop and displacement zones is activated
# if the wind angle if more or less 'PERPENDICULAR_THRESHOLD_ANGLE' ° higher
# or lower than 90° (20° is given in Bagal et al. - 2004 and 15° in Pol et al. - 2006)
PERPENDICULAR_THRESHOLD_ANGLE = 15
# "Corner wind" rooftop recirculation is activated when a facade is 30 to 70° to
# the perpendicular to the wind direction (Bagal et al., 2004)
CORNER_THRESHOLD_ANGLE_MIN = 30
CORNER_THRESHOLD_ANGLE_MAX = 70
CORNER_THRESHOLD_ANGLE = [CORNER_THRESHOLD_ANGLE_MIN, CORNER_THRESHOLD_ANGLE_MAX]
ELLIPSOID_MIN_LENGTH = float(MESH_SIZE) / 4
GEOMETRY_MERGE_TOLERANCE = 0.05
SNAPPING_TOLERANCE = 0.3
GEOMETRY_SIMPLIFICATION_DISTANCE = 0.25

GEOM_FIELD = 'THE_GEOM'
ID_FIELD_BUILD = 'id'
ID_FIELD_BLOCK = 'ID_BLOCK'
ID_FIELD_STACKED_BLOCK = 'ID_STACKED_BLOCK'
ID_UPSTREAM_STACKED_BLOCK = ID_FIELD_STACKED_BLOCK
ID_DOWNSTREAM_STACKED_BLOCK = 'ID_DOWNSTREAM_STACKED_BLOCK'
ID_FIELD_CANYON = 'ID_CANYON'
ID_VEGETATION = 'ID_VEG'
ID_ZONE_VEGETATION = 'ID_ZONE_VEG'
ID_POINT = 'ID_POINT'
ID_POINT_X = 'ID_X'
ID_POINT_Y = 'ID_Y'
ID_POINT_Z = 'ID_Z'

HEIGHT_FIELD = 'hauteur'
BASE_HEIGHT_FIELD = 'BASE_HEIGHT'
CAVITY_BASE_HEIGHT_FIELD = 'CAVITY_BASE_HEIGHT'
VEGETATION_CROWN_BASE_HEIGHT = 'MIN_HEIGHT'
VEGETATION_CROWN_TOP_HEIGHT = 'MAX_HEIGHT'
TOP_CANOPY_HEIGHT_POINT = 'MAX_CANOPY_HEIGHT'
VEGETATION_ATTENUATION_FACTOR = 'ATTENUATIO'
UPSTREAM_HEIGHT_FIELD = 'UPSTREAM_HEIGHT'
DOWNSTREAM_HEIGHT_FIELD = 'DOWNSTREAM_HEIGHT'
MAX_CANYON_HEIGHT_FIELD = 'Hc'
CANYON_DELTAH_FIELD = 'CANYON_DELTAH'
UPWIND_FACADE_ANGLE_FIELD = 'THETA_WIND'
UPWIND_FACADE_FIELD = 'UPWIND_FACADE_ID'
DOWNWIND_FACADE_FIELD = 'DOWNWIND_FACADE_ID'
STACKED_BLOCK_WIDTH = 'BLOCK_WIDTH'
STACKED_BLOCK_X_MED = 'STACKED_BLOCK_X_MED'
STACKED_BLOCK_UPSTREAMEST_X = 'STACKED_BLOCK_UPSTREAMEST_X'
SIN_BLOCK_LEFT_AZIMUTH = 'SIN_BLOCK_LEFT_AZIMUTH'
COS_BLOCK_LEFT_AZIMUTH = 'COS_BLOCK_LEFT_AZIMUTH'
COS_BLOCK_RIGHT_AZIMUTH = 'COS_BLOCK_RIGHT_AZIMUTH'
SIN_BLOCK_RIGHT_AZIMUTH = 'SIN_BLOCK_RIGHT_AZIMUTH'
SIN_BLOCK_AZIMUTH = 'SIN_BLOCK_AZIMUTH'
COS_BLOCK_AZIMUTH = 'COS_BLOCK_AZIMUTH'
EFFECTIVE_LENGTH_FIELD = 'L_EFF'
EFFECTIVE_WIDTH_FIELD = 'W_EFF'
DISPLACEMENT_LENGTH_FIELD = 'Lf'
DISPLACEMENT_LENGTH_VORTEX_FIELD = 'Lfv'
CAVITY_LENGTH_FIELD = 'Lr'
WAKE_LENGTH_FIELD = 'Lw'
ROOFTOP_PERP_LENGTH = 'Lcp'
ROOFTOP_CORNER_FACADE_LENGTH = 'Lfc'
ROOFTOP_CORNER_LENGTH = 'Lcc'
ROOFTOP_PERP_HEIGHT = 'Hcm'
ROOFTOP_WIND_FACTOR = 'C1'
ROOFTOP_PERP_VAR_HEIGHT = 'Hr'
ROOFTOP_CORNER_VAR_HEIGHT = 'Hccp'

PREFIX_NAME = ''
Y_WALL = 'Y_WALL'
Y_POINT = 'Y_POINT'
DISTANCE_BUILD_TO_POINT_FIELD = 'DY'
LENGTH_ZONE_FIELD = 'D_0'
POINT_RELATIVE_POSITION_FIELD = 'DY_D0'
WAKE_RELATIVE_POSITION_FIELD = 'D0C_DY_15'
DOWNSTREAM_X_RELATIVE_POSITION = 'DXC_DX_DXC'

DISPLACEMENT_NAME = 'DISPLACEMENT'
DISPLACEMENT_VORTEX_NAME = 'DISPLACEMENT_VORTEX'
CAVITY_NAME = 'CAVITY'
WAKE_NAME = 'WAKE'
STREET_CANYON_NAME = 'STREET_CANYON'
ROOFTOP_PERP_NAME = 'ROOFTOP_PERPENDICULAR'
ROOFTOP_CORN_NAME = 'ROOFTOP_CORNER'
VEGETATION_BUILT_NAME = 'VEGETATION_BUILT'
VEGETATION_OPEN_NAME = 'VEGETATION_OPEN'
ALL_VEGETATION_NAME = 'VEGETATION_ALL'
CAVITY_BACKWARD_NAME = 'CAVITY_BACKWARD'
WAKE_BACKWARD_NAME = 'WAKE_BACKWARD'

UPPER_VERTICAL_THRESHOLD = 'Z_UP_THRESH'
LOWER_VERTICAL_THRESHOLD = 'Z_DOWN_THRESH'
U = 'U'
V = 'V'
W = 'W'
U_WEIGHT = 'U_WEIGHT'
V_WEIGHT = 'V_WEIGHT'
W_WEIGHT = 'W_WEIGHT'
VEGETATION_FACTOR = 'VEGETATION_FACTOR'

X = 'X'
Y = 'Y'
Z = 'Z'
# Coefficients for displacement zone calculation
C_DZ = 0.4
P_DZ = 0.16
# Coefficient for rooftop perpendicular
P_RTP = 0.16
# Default vegetation attenuation factor (value for Larch plantation - Cionco et al. (1978))
DEFAULT_VEG_ATTEN_FACT = 1.00
# Default vegetation crown base height (in % of crown top height)
DEFAULT_VEG_CROWN_BASE_HEIGHT_FRAC = 0.25

# Defines priorities (column "priority") when the zone comes from a same
# obstacle of same height. Also contains a column "ref_height" to
# know by which wind speed height should be multiplied a weigthing factor
# (1: "Associated building height",
#  2: "Reference wind speed measurement height Z_REF",
#  3: "Point height")
REF_HEIGHT_FIELD = 'REF_HEIGHT'
PRIORITY_FIELD = 'PRIORITY'
IS_UPSTREAM_FIELD = 'IS_UPSTREAM'
REF_HEIGHT_UPSTREAM_WEIGHTING = 3
REF_HEIGHT_DOWNSTREAM_WEIGHTING = 3
IS_UPSTREAM_UPSTREAM_WEIGHTING = 0
IS_UPSTREAM_DOWNSTREAM_WEIGHTING = 0
UPSTREAM_PRIORITY_TABLES = pd.DataFrame(
    {
        PRIORITY_FIELD: [1, 2, 3, 3, 3, 4, 5],
        REF_HEIGHT_FIELD: [1, 1, 2, 2, 1, 1, 3],
        IS_UPSTREAM_FIELD: [0, 0, 0, 0, 1, 1, 0],
    },
    index=[
        STREET_CANYON_NAME,
        CAVITY_NAME,
        ROOFTOP_PERP_NAME,
        ROOFTOP_CORN_NAME,
        DISPLACEMENT_VORTEX_NAME,
        DISPLACEMENT_NAME,
        WAKE_NAME,
    ],
)
UPSTREAM_BACKWARD_PRIORITY_TABLES = pd.DataFrame(
    {PRIORITY_FIELD: [1, 2], REF_HEIGHT_FIELD: [1, 1], IS_UPSTREAM_FIELD: [0, 0]},
    index=[CAVITY_BACKWARD_NAME, WAKE_BACKWARD_NAME],
)
UPSTREAM_WEIGHTING_TABLES = [WAKE_NAME]
UPSTREAM_BACKWARD_WEIGHTING_TABLES = [WAKE_BACKWARD_NAME]
UPSTREAM_WEIGHTING_INTRA_RULES = 'upstream'
UPSTREAM_WEIGHTING_INTER_RULES = 'upstream'
DOWNSTREAM_WEIGTHING_TABLE = ALL_VEGETATION_NAME

ID_3D_POINT = 'ID'
SELECTED_SUFFIX = '_SELECTION'

# Informations about the netcdf file
WIND_GROUP = '3D_wind'
VERT_WIND = 'vertWind'
RLAT = 'rlat'
RLON = 'rlon'
LAT = 'lat'
LON = 'lon'
WINDSPEED_X = 'windSpeed_x'
WINDSPEED_Y = 'windSpeed_y'
WINDSPEED_Z = 'windSpeed_z'
LEVELS = 'Levels'
WINDSPEED_PROFILE = 'windSpeed'
