#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:39:07 2021

@author: Jérémy Bernard, University of Gothenburg
"""

try:
    from GlobalVariables import (
        ALONG_WIND_ZONE_EXTEND,
        CROSS_WIND_ZONE_EXTEND,
        MESH_SIZE,
        PREFIX_NAME,
        GEOM_FIELD,
        ID_POINT,
        ID_POINT_X,
        ID_POINT_Y,
        CAVITY_NAME,
        WAKE_NAME,
        DISPLACEMENT_NAME,
        DISPLACEMENT_VORTEX_NAME,
        STREET_CANYON_NAME,
        ROOFTOP_PERP_NAME,
        UPWIND_FACADE_FIELD,
        ID_FIELD_STACKED_BLOCK,
        ID_FIELD_CANYON,
        ROOFTOP_CORN_NAME,
        HEIGHT_FIELD,
        UPSTREAM_HEIGHT_FIELD,
        DOWNSTREAM_HEIGHT_FIELD,
        ROOFTOP_CORNER_FACADE_LENGTH,
        ROOFTOP_CORNER_LENGTH,
        UPWIND_FACADE_ANGLE_FIELD,
        ROOFTOP_WIND_FACTOR,
        LENGTH_ZONE_FIELD,
        Y_WALL,
        MAX_CANYON_HEIGHT_FIELD,
        ROOFTOP_PERP_LENGTH,
        ROOFTOP_PERP_HEIGHT,
        BASE_HEIGHT_FIELD,
        UPPER_VERTICAL_THRESHOLD,
        POINT_RELATIVE_POSITION_FIELD,
        U,
        V,
        W,
        DISTANCE_BUILD_TO_POINT_FIELD,
        ROOFTOP_PERP_VAR_HEIGHT,
        WAKE_RELATIVE_POSITION_FIELD,
        ROOFTOP_CORNER_VAR_HEIGHT,
        DEBUG,
        VEGETATION_CROWN_TOP_HEIGHT,
        ID_VEGETATION,
        TOP_CANOPY_HEIGHT_POINT,
        VEGETATION_ATTENUATION_FACTOR,
        VEGETATION_CROWN_BASE_HEIGHT,
        DZ,
        ID_POINT_Z,
        C_DZ,
        P_DZ,
        Z,
        X,
        Y,
        Z_REF,
        P_RTP,
        VEGETATION_OPEN_NAME,
        VEGETATION_BUILT_NAME,
        VEGETATION_FACTOR,
        UPSTREAM_PRIORITY_TABLES,
        UPSTREAM_WEIGHTING_TABLES,
        UPSTREAM_WEIGHTING_INTER_RULES,
        UPSTREAM_WEIGHTING_INTRA_RULES,
        DOWNSTREAM_WEIGTHING_TABLE,
        REF_HEIGHT_UPSTREAM_WEIGHTING,
        REF_HEIGHT_FIELD,
        REF_HEIGHT_DOWNSTREAM_WEIGHTING,
        PRIORITY_FIELD,
        ID_3D_POINT,
        V_REF,
        TEMPO_DIRECTORY,
        Y_POINT,
        ID_UPSTREAM_STACKED_BLOCK,
        GEOMETRY_MERGE_TOLERANCE,
        SNAPPING_TOLERANCE,
        GEOMETRY_SIMPLIFICATION_DISTANCE,
        ID_DOWNSTREAM_STACKED_BLOCK,
        DOWNWIND_FACADE_FIELD,
        PROFILE_TYPE,
        HORIZ_WIND_SPEED,
        CAVITY_BACKWARD_NAME,
        WAKE_BACKWARD_NAME,
        UPSTREAM_BACKWARD_PRIORITY_TABLES,
        UPSTREAM_BACKWARD_WEIGHTING_TABLES,
        STACKED_BLOCK_UPSTREAMEST_X,
        COS_BLOCK_RIGHT_AZIMUTH,
        COS_BLOCK_LEFT_AZIMUTH,
        COS_BLOCK_AZIMUTH,
        SIN_BLOCK_RIGHT_AZIMUTH,
        SIN_BLOCK_LEFT_AZIMUTH,
        SIN_BLOCK_AZIMUTH,
        STACKED_BLOCK_WIDTH,
        DOWNSTREAM_X_RELATIVE_POSITION,
        V_WEIGHT,
        U_WEIGHT,
        W_WEIGHT,
        STACKED_BLOCK_X_MED,
        REMOVE_INITIALIZATION_OFFSET,
        IS_UPSTREAM_FIELD,
        IS_UPSTREAM_UPSTREAM_WEIGHTING,
    )
    import DataUtil

except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import (
        ALONG_WIND_ZONE_EXTEND,
        CROSS_WIND_ZONE_EXTEND,
        MESH_SIZE,
        PREFIX_NAME,
        GEOM_FIELD,
        ID_POINT,
        ID_POINT_X,
        ID_POINT_Y,
        CAVITY_NAME,
        WAKE_NAME,
        DISPLACEMENT_NAME,
        DISPLACEMENT_VORTEX_NAME,
        STREET_CANYON_NAME,
        ROOFTOP_PERP_NAME,
        UPWIND_FACADE_FIELD,
        ID_FIELD_STACKED_BLOCK,
        ID_FIELD_CANYON,
        ROOFTOP_CORN_NAME,
        HEIGHT_FIELD,
        UPSTREAM_HEIGHT_FIELD,
        DOWNSTREAM_HEIGHT_FIELD,
        ROOFTOP_CORNER_FACADE_LENGTH,
        ROOFTOP_CORNER_LENGTH,
        UPWIND_FACADE_ANGLE_FIELD,
        ROOFTOP_WIND_FACTOR,
        LENGTH_ZONE_FIELD,
        Y_WALL,
        MAX_CANYON_HEIGHT_FIELD,
        ROOFTOP_PERP_LENGTH,
        ROOFTOP_PERP_HEIGHT,
        BASE_HEIGHT_FIELD,
        UPPER_VERTICAL_THRESHOLD,
        POINT_RELATIVE_POSITION_FIELD,
        U,
        V,
        W,
        DISTANCE_BUILD_TO_POINT_FIELD,
        ROOFTOP_PERP_VAR_HEIGHT,
        WAKE_RELATIVE_POSITION_FIELD,
        ROOFTOP_CORNER_VAR_HEIGHT,
        DEBUG,
        VEGETATION_CROWN_TOP_HEIGHT,
        ID_VEGETATION,
        TOP_CANOPY_HEIGHT_POINT,
        VEGETATION_ATTENUATION_FACTOR,
        VEGETATION_CROWN_BASE_HEIGHT,
        DZ,
        ID_POINT_Z,
        C_DZ,
        P_DZ,
        Z,
        X,
        Y,
        Z_REF,
        P_RTP,
        VEGETATION_OPEN_NAME,
        VEGETATION_BUILT_NAME,
        VEGETATION_FACTOR,
        UPSTREAM_PRIORITY_TABLES,
        UPSTREAM_WEIGHTING_TABLES,
        UPSTREAM_WEIGHTING_INTER_RULES,
        UPSTREAM_WEIGHTING_INTRA_RULES,
        DOWNSTREAM_WEIGTHING_TABLE,
        REF_HEIGHT_UPSTREAM_WEIGHTING,
        REF_HEIGHT_FIELD,
        REF_HEIGHT_DOWNSTREAM_WEIGHTING,
        PRIORITY_FIELD,
        ID_3D_POINT,
        V_REF,
        TEMPO_DIRECTORY,
        Y_POINT,
        ID_UPSTREAM_STACKED_BLOCK,
        GEOMETRY_MERGE_TOLERANCE,
        SNAPPING_TOLERANCE,
        GEOMETRY_SIMPLIFICATION_DISTANCE,
        ID_DOWNSTREAM_STACKED_BLOCK,
        DOWNWIND_FACADE_FIELD,
        PROFILE_TYPE,
        HORIZ_WIND_SPEED,
        CAVITY_BACKWARD_NAME,
        WAKE_BACKWARD_NAME,
        UPSTREAM_BACKWARD_PRIORITY_TABLES,
        UPSTREAM_BACKWARD_WEIGHTING_TABLES,
        STACKED_BLOCK_UPSTREAMEST_X,
        COS_BLOCK_RIGHT_AZIMUTH,
        COS_BLOCK_LEFT_AZIMUTH,
        COS_BLOCK_AZIMUTH,
        SIN_BLOCK_RIGHT_AZIMUTH,
        SIN_BLOCK_LEFT_AZIMUTH,
        SIN_BLOCK_AZIMUTH,
        STACKED_BLOCK_WIDTH,
        DOWNSTREAM_X_RELATIVE_POSITION,
        V_WEIGHT,
        U_WEIGHT,
        W_WEIGHT,
        STACKED_BLOCK_X_MED,
        REMOVE_INITIALIZATION_OFFSET,
        IS_UPSTREAM_FIELD,
        IS_UPSTREAM_UPSTREAM_WEIGHTING,
    )
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil

import pandas as pd

idx = pd.IndexSlice

import math
import numpy as np
import os


def createGrid(
    cursor,
    dicOfInputTables,
    srid,
    alongWindZoneExtend=ALONG_WIND_ZONE_EXTEND,
    crossWindZoneExtend=CROSS_WIND_ZONE_EXTEND,
    meshSize=MESH_SIZE,
    prefix=PREFIX_NAME,
):
    """Creates a grid of points which will be used to initiate the wind
    speed field. The grid limits are defined by the enveloppe of a set of
    geometries ('dicOfInputTables') extended to a certain distance
    along wind ('alongWindZoneExtend') and cross wind ('crossWindZoneExtend')

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfInputTables: dictionary of String
                Dictionary of String with table names as values. The grid limits
                will be based on the enveloppe of this set of geometries
            srid: int
                SRID of the building data (useful for grid creation)
            alongWindZoneExtend: float, default ALONG_WIND_ZONE_EXTEND
                Distance (in meter) of the extend of the zone around the
                rotated obstacles in the along-wind direction
            crosswindZoneExtend: float, default CROSS_WIND_ZONE_EXTEND
                Distance (in meter) of the extend of the zone around the
                rotated obstacles in the cross-wind direction
            meshSize: float, default MESH_SIZE
                Resolution (in meter) of the grid
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            gridTable: String
                Name of the grid point table"""
    print('Creates the grid of points')

    # Output base name
    outputBaseName = 'GRID'

    # Name of the output table
    gridTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Gather all tables in one
    gatherQuery = [
        """SELECT {0} AS {0} FROM {1}""".format(GEOM_FIELD, dicOfInputTables[t])
        for t in dicOfInputTables.keys()
    ]

    # Calculate the extend of the envelope of all geometries
    finalQuery = """
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT   ST_SETSRID({1}, {9}) AS {1},
                        ID AS {6},
                        ID_COL AS {7},
                        ID_ROW AS {8},
                        ST_Y({1}) AS {10},
            FROM ST_MAKEGRIDPOINTS((SELECT ST_EXPAND(ST_EXTENT({1}),
                                                      {2},
                                                      {3}) FROM ({5})), 
                                    {4}, 
                                    {4})""".format(
        gridTable,
        GEOM_FIELD,
        crossWindZoneExtend,
        alongWindZoneExtend,
        meshSize,
        ' UNION ALL '.join(gatherQuery),
        ID_POINT,
        ID_POINT_X,
        ID_POINT_Y,
        srid,
        Y_POINT,
    )
    cursor.execute(finalQuery)

    return gridTable


def affectsPointToBuildZone(
    cursor, gridTable, dicOfBuildRockleZoneTable, prefix=PREFIX_NAME
):
    """Affects each point to a building Rockle zone and calculates relative
    point position within the zone for some of them.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            gridTable: String
                Name of the grid point table
            dicOfBuildRockleZoneTable: Dictionary of building Rockle zone tables
                Dictionary containing as key the building Rockle zone name and
                as value the corresponding table name
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfOutputTables: dictionary of table name
                Dictionary having as key the type of Rockle zone and as value
                the name of the table containing points corresponding to the zone
            verticalLineTable: String
                Vertical lines (not along z but along the grid "north-south")
                useful for future calculations"""
    print(
        """Affects each grid point to a building Rockle zone and calculates needed 
          variables for 3D wind speed"""
    )

    # Name of the output tables
    dicOfOutputTables = {
        t: DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix=prefix), suffix='INIT_POINTS'
        )
        for t in dicOfBuildRockleZoneTable
    }
    verticalLineTable = DataUtil.prefix(tableName='VERTICAL_LINES', prefix=prefix)

    # Temporary tables (and prefix for temporary tables)
    tempoCavity = DataUtil.postfix('TEMPO_CAVITY')
    dicOfTempoOutput = {
        t: DataUtil.postfix(
            DataUtil.prefix(tableName=dicOfOutputTables[t], prefix='TEMPO')
        )
        for t in dicOfBuildRockleZoneTable
    }
    dicOfPrefixZoneLim = {
        t: DataUtil.postfix(DataUtil.prefix(tableName=t, prefix='ZONE_LIMITS'))
        for t in dicOfBuildRockleZoneTable
    }

    # Tables that should keep y value (distance from upwind building)
    listTabYvalues = [
        CAVITY_NAME,
        WAKE_NAME,
        DISPLACEMENT_NAME,
        DISPLACEMENT_VORTEX_NAME,
        STREET_CANYON_NAME,
        ROOFTOP_PERP_NAME,
    ]

    # ID_ZONE field to use for join depending on zone type
    idZone = {
        DISPLACEMENT_NAME: UPWIND_FACADE_FIELD,
        DISPLACEMENT_VORTEX_NAME: UPWIND_FACADE_FIELD,
        CAVITY_NAME: DOWNWIND_FACADE_FIELD,
        WAKE_NAME: DOWNWIND_FACADE_FIELD,
        STREET_CANYON_NAME: ID_FIELD_CANYON,
        ROOFTOP_PERP_NAME: UPWIND_FACADE_FIELD,
        ROOFTOP_CORN_NAME: UPWIND_FACADE_FIELD,
    }

    query = [
        """{0};
                 DROP TABLE IF EXISTS {1}
             """.format(
            DataUtil.createIndex(
                tableName=gridTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            ','.join(dicOfOutputTables.values()),
        )
    ]
    # Construct a query to affect each point to a Rockle zone
    for i, t in enumerate(dicOfBuildRockleZoneTable):
        # The query differs depending on whether y value should be kept
        queryKeepY = 'b.{1}, b.{0}, b.{2},'.format(ID_POINT_X, Y_POINT, ID_POINT_Y)

        # The columns to keep are different in case of street canyon zone
        # or rooftop corner zone
        columnsToKeepQuery = """b.{0}, {1} a.{2}, a.{3}
                                """.format(
            ID_POINT, queryKeepY, idZone[t], HEIGHT_FIELD
        )
        if t == STREET_CANYON_NAME:
            columnsToKeepQuery = """b.{0}, {1} a.{2}, a.{3}, a.{4}, a.{5}, a.{6}
                                    """.format(
                ID_POINT,
                queryKeepY,
                idZone[t],
                UPSTREAM_HEIGHT_FIELD,
                DOWNSTREAM_HEIGHT_FIELD,
                UPWIND_FACADE_FIELD,
                ID_UPSTREAM_STACKED_BLOCK,
            )
        elif t == ROOFTOP_CORN_NAME:
            columnsToKeepQuery = """b.{0}, a.{1}, a.{2}, b.{3}, a.{4}, a.{5}, a.{6}, a.{7}, 
                                   ST_STARTPOINT(ST_TOMULTILINE(a.{3})) AS GEOM_CORNER_POINT,
                                   b.{8}
                                   """.format(
                ID_POINT,
                idZone[t],
                HEIGHT_FIELD,
                GEOM_FIELD,
                ROOFTOP_CORNER_FACADE_LENGTH,
                ROOFTOP_CORNER_LENGTH,
                UPWIND_FACADE_ANGLE_FIELD,
                ROOFTOP_WIND_FACTOR,
                ID_POINT_X,
            )

        query.append(
            """ 
            {5};
            DROP TABLE IF EXISTS {2};
            CREATE TABLE {2}
                AS SELECT {4}
                FROM    {0} AS a, {3} AS b
                WHERE   a.{1} && b.{1}
                        AND ST_INTERSECTS(a.{1}, b.{1})
                        """.format(
                dicOfBuildRockleZoneTable[t],
                GEOM_FIELD,
                dicOfTempoOutput[t],
                gridTable,
                columnsToKeepQuery,
                DataUtil.createIndex(
                    tableName=dicOfBuildRockleZoneTable[t],
                    fieldName=GEOM_FIELD,
                    isSpatial=True,
                ),
            )
        )

    # Get the ID of the lower grid point row
    cursor.execute(
        """
       SELECT MAX(DISTINCT {0}) AS {0} FROM {1};
                   """.format(ID_POINT_Y, gridTable)
    )
    idLowerGridRow = cursor.fetchall()[0][0]

    # For Rockle zones that needs relative point distance, extra calculation is needed
    # First creates vertical lines
    endOfQuery = """ 
        {6};
        {7};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0} 
            AS SELECT   a.{1},
                        ST_MAKELINE(b.{2}, a.{2}) AS {2},
                        b.{4},
                        ST_X(b.{2}) AS {9}
            FROM {3} AS a LEFT JOIN {3} AS b ON a.{1} = b.{1}
            WHERE a.{4} = 1 AND b.{4} = {5};
        {8};
            """.format(
        verticalLineTable,
        ID_POINT_X,
        GEOM_FIELD,
        gridTable,
        ID_POINT_Y,
        idLowerGridRow,
        DataUtil.createIndex(
            tableName=gridTable, fieldName=ID_POINT_X, isSpatial=False
        ),
        DataUtil.createIndex(
            tableName=gridTable, fieldName=ID_POINT_Y, isSpatial=False
        ),
        DataUtil.createIndex(tableName=gridTable, fieldName=GEOM_FIELD, isSpatial=True),
        X,
    )

    # Fields to keep in the zone table (zone dependent)
    varToKeepZone = {
        DISPLACEMENT_NAME: """b.{0},
                                    b.{1},
                                    a.{2},
                                    b.{6},
                                    ST_YMIN(ST_INTERSECTION(a.{3}, 
                                                            ST_TOMULTILINE(b.{3}))
                                            ) AS {5},
                                    ST_LENGTH(ST_INTERSECTION(a.{3}, b.{3})) AS {4}
                                    """.format(
            idZone[DISPLACEMENT_NAME],
            HEIGHT_FIELD,
            ID_POINT_X,
            GEOM_FIELD,
            LENGTH_ZONE_FIELD + DISPLACEMENT_NAME[0],
            Y_WALL,
            UPWIND_FACADE_ANGLE_FIELD,
        ),
        DISPLACEMENT_VORTEX_NAME: """b.{0},
                                    b.{1},
                                    a.{2},
                                    b.{6},
                                    ST_YMIN(ST_INTERSECTION(a.{3}, 
                                                            ST_TOMULTILINE(b.{3}))
                                            ) AS {5},
                                    ST_LENGTH(ST_INTERSECTION(a.{3}, b.{3})) AS {4}
                                    """.format(
            idZone[DISPLACEMENT_VORTEX_NAME],
            HEIGHT_FIELD,
            ID_POINT_X,
            GEOM_FIELD,
            LENGTH_ZONE_FIELD + DISPLACEMENT_VORTEX_NAME[0],
            Y_WALL,
            UPWIND_FACADE_ANGLE_FIELD,
        ),
        CAVITY_NAME: """b.{0},
                                    b.{1},
                                    a.{2},
                                    ST_YMAX(ST_INTERSECTION(a.{3}, 
                                                            b.{3})
                                            ) AS {5},
                                    a.{6},
                                    CASE WHEN ST_DIMENSION(ST_INTERSECTION(a.{3}, 
                                                                           b.{3}))=0
                                        THEN 0
                                        ELSE ST_LENGTH(ST_MAKELINE(ST_TOMULTIPOINT(ST_INTERSECTION(a.{3}, 
                                                                                          b.{3})
                                                                                   )
                                                                   )
                                                       )
                                        END AS {4},
                                    b.{7},
                                    CASE WHEN a.{8} > b.{9}
                                        THEN b.{10}
                                        ELSE b.{11} 
                                        END AS {12},
                                    CASE WHEN a.{8} > b.{9}
                                        THEN b.{13}
                                        ELSE b.{14} 
                                        END AS {15},
                                    POWER((b.{16}/2-ABS(a.{8}-b.{18}))/(b.{16}/2),0.5) AS {17}
                                    """.format(
            idZone[CAVITY_NAME],
            HEIGHT_FIELD,
            ID_POINT_X,
            GEOM_FIELD,
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            Y_WALL,
            ID_POINT_Y,
            ID_FIELD_STACKED_BLOCK,
            X,
            STACKED_BLOCK_UPSTREAMEST_X,
            COS_BLOCK_RIGHT_AZIMUTH,
            COS_BLOCK_LEFT_AZIMUTH,
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_RIGHT_AZIMUTH,
            SIN_BLOCK_LEFT_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            STACKED_BLOCK_WIDTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
            STACKED_BLOCK_X_MED,
        ),
        WAKE_NAME: """b.{0},
                                    b.{1},
                                    a.{2},
                                    ST_YMAX(ST_INTERSECTION(a.{3}, 
                                                            b.{3})
                                            ) AS {5},
                                    CASE WHEN ST_DIMENSION(ST_INTERSECTION(a.{3}, 
                                                                           b.{3}))=0
                                        THEN 0
                                        ELSE ST_LENGTH(ST_MAKELINE(ST_TOMULTIPOINT(ST_INTERSECTION(a.{3}, 
                                                                                          b.{3})
                                                                                   )
                                                                   )
                                                       )
                                        END AS {4},
                                    b.{6}
                                    """.format(
            idZone[WAKE_NAME],
            HEIGHT_FIELD,
            ID_POINT_X,
            GEOM_FIELD,
            LENGTH_ZONE_FIELD + WAKE_NAME[0],
            Y_WALL,
            ID_FIELD_STACKED_BLOCK,
        ),
        STREET_CANYON_NAME: """b.{0},
                                    b.{1},
                                    LEAST(b.{3}, b.{1}) AS {4},
                                    b.{5},
                                    a.{2},
                                    b.{8},
                                    ST_YMAX(ST_INTERSECTION(a.{6}, 
                                                            b.{6})
                                            ) AS {9},
                                    ST_LENGTH(ST_MAKELINE(ST_TOMULTIPOINT(ST_INTERSECTION(a.{6},
                                                                                          b.{6})
                                                                          )
                                                          )
                                              ) AS {7},
                                    b.{10},
                                    b.{11},
                                    a.{12},
                                    b.{13}
                                    """.format(
            idZone[STREET_CANYON_NAME],
            UPSTREAM_HEIGHT_FIELD,
            ID_POINT_X,
            DOWNSTREAM_HEIGHT_FIELD,
            MAX_CANYON_HEIGHT_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            GEOM_FIELD,
            LENGTH_ZONE_FIELD + STREET_CANYON_NAME[0],
            BASE_HEIGHT_FIELD,
            Y_WALL,
            UPWIND_FACADE_FIELD,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_POINT_Y,
            DOWNWIND_FACADE_FIELD,
        ),
        ROOFTOP_PERP_NAME: """b.{0},
                                    b.{1},
                                    a.{2},
                                    b.{3},
                                    b.{4},
                                    ST_YMAX(ST_INTERSECTION(a.{5}, 
                                                            b.{5})
                                            ) AS {6}
                                    """.format(
            idZone[ROOFTOP_PERP_NAME],
            HEIGHT_FIELD,
            ID_POINT_X,
            ROOFTOP_PERP_LENGTH,
            ROOFTOP_PERP_HEIGHT,
            GEOM_FIELD,
            Y_WALL,
        ),
    }

    # Calculates the coordinate of the upper and lower part of each of the
    # Röckle zones for each "north/south" line
    endOfQuery += ';'.join(
        [
            """
        {6};
        DROP TABLE IF EXISTS {0}, {4};
        CREATE TABLE {0}
            AS SELECT   {5}
            FROM    {3} AS a, {2} AS b
            WHERE   a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1}) AND 
                    ST_DIMENSION(ST_INTERSECTION(a.{1}, b.{1})) > 0;
                  """.format(
                dicOfPrefixZoneLim[t],
                GEOM_FIELD,
                dicOfBuildRockleZoneTable[t],
                verticalLineTable,
                dicOfOutputTables[t],
                varToKeepZone[t],
                DataUtil.createIndex(
                    tableName=dicOfBuildRockleZoneTable[t],
                    fieldName=GEOM_FIELD,
                    isSpatial=True,
                ),
            )
            for t in listTabYvalues
        ]
    )
    query.append(endOfQuery)
    cursor.execute(';'.join(query))

    # The cavity zone length is needed for the wind speed calculation of
    # wake zone points and for the upper height limit of the street canyon
    # while the wake zone length is needed for cavity zone wind speed calculation
    cursor.execute(
        """
       {0}{1}{2}{3}
       DROP TABLE IF EXISTS TEMPO_WAKE;
       CREATE TABLE TEMPO_WAKE 
           AS SELECT   a.*, 
                       b.{4}, b.{19}, b.{20}, b.{21}
           FROM     {5} AS a LEFT JOIN {6} AS b 
                    ON a.{7} = b.{7} AND a.{8} = b.{8};
       DROP TABLE IF EXISTS {5};
       ALTER TABLE TEMPO_WAKE RENAME TO {5};
       {9}{10}{11}{12}
       DROP TABLE IF EXISTS TEMPO_CANYON;
       CREATE TABLE TEMPO_CANYON 
           AS SELECT   a.*, 
                       b.{4}
           FROM     {13} AS a LEFT JOIN {6} AS b 
                    ON a.{7} = b.{7} AND a.{8} = b.{8};
       DROP TABLE IF EXISTS {13};
       ALTER TABLE TEMPO_CANYON RENAME TO {13};
       {14}{15}{16}{17}
       DROP TABLE IF EXISTS TEMPO_CAV;
       CREATE TABLE TEMPO_CAV 
           AS SELECT   a.*, 
                       b.{18}
           FROM     {6} AS a LEFT JOIN {5} AS b 
                    ON a.{7} = b.{7} AND a.{8} = b.{8};
       DROP TABLE IF EXISTS {6};
       ALTER TABLE TEMPO_CAV RENAME TO {6};
       """.format(
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME], fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[WAKE_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[WAKE_NAME], fieldName=ID_POINT_X, isSpatial=False
            ),
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            dicOfPrefixZoneLim[WAKE_NAME],
            dicOfPrefixZoneLim[CAVITY_NAME],
            DOWNWIND_FACADE_FIELD,
            ID_POINT_X,
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME], fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[STREET_CANYON_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[STREET_CANYON_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            dicOfPrefixZoneLim[STREET_CANYON_NAME],
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[CAVITY_NAME], fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[WAKE_NAME],
                fieldName=DOWNWIND_FACADE_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfPrefixZoneLim[WAKE_NAME], fieldName=ID_POINT_X, isSpatial=False
            ),
            LENGTH_ZONE_FIELD + WAKE_NAME[0],
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
        )
    )

    # Fields to keep in the point table (zone dependent)
    varToKeepPoint = {
        DISPLACEMENT_NAME: """b.{0},
                                    0.6*a.{4}*SQRT(1-POWER((b.{10}-a.{6})/
                                                                     a.{2}, 2)) AS {1},
                                    (b.{10}-a.{6})/a.{2} AS {5},
                                    b.{3},
                                    a.{4},
                                    0 AS {8},
                                    1 AS {9},
                                    CAST(a.{6} AS INTEGER) AS {6}""".format(
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            LENGTH_ZONE_FIELD + DISPLACEMENT_NAME[0],
            idZone[DISPLACEMENT_NAME],
            HEIGHT_FIELD,
            POINT_RELATIVE_POSITION_FIELD + DISPLACEMENT_NAME[0],
            Y_WALL,
            UPWIND_FACADE_ANGLE_FIELD,
            U,
            V,
            Y_POINT,
        ),
        DISPLACEMENT_VORTEX_NAME: """b.{0},
                                    0.5*a.{4}*SQRT(1-POWER((b.{7}-a.{6})/
                                                                     a.{2}, 2)) AS {1},
                                    (b.{7}-a.{6})/a.{2} AS {5},
                                    b.{3},
                                    a.{4},
                                    CAST(a.{6} AS INTEGER) AS {6}""".format(
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            LENGTH_ZONE_FIELD + DISPLACEMENT_VORTEX_NAME[0],
            idZone[DISPLACEMENT_VORTEX_NAME],
            HEIGHT_FIELD,
            POINT_RELATIVE_POSITION_FIELD + DISPLACEMENT_VORTEX_NAME[0],
            Y_WALL,
            Y_POINT,
        ),
        CAVITY_NAME: """a.{11},
                                    b.{0},
                                    a.{4}*SQRT(1-POWER((a.{7}-b.{9})/
                                                       a.{2}, 2)) AS {1},
                                    (a.{7}-b.{9})/a.{2} AS {5},
                                    (a.{7}-b.{9}) AS {8},
                                    a.{2},
                                    b.{3},
                                    a.{4},
                                    b.{6},
                                    CAST(a.{7} AS INTEGER) AS {7},
                                    b.{10},
                                    a.{12},
                                    a.{13},
                                    a.{14},
                                    a.{15},
                                    (a.{7}-b.{9})/a.{15} AS {16}""".format(
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            idZone[CAVITY_NAME],
            HEIGHT_FIELD,
            POINT_RELATIVE_POSITION_FIELD + CAVITY_NAME[0],
            ID_POINT_X,
            Y_WALL,
            DISTANCE_BUILD_TO_POINT_FIELD,
            Y_POINT,
            ID_POINT_Y,
            ID_FIELD_STACKED_BLOCK,
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
            LENGTH_ZONE_FIELD + WAKE_NAME[0],
            POINT_RELATIVE_POSITION_FIELD + WAKE_NAME[0],
        ),
        WAKE_NAME: """a.{9},
                                    b.{0},
                                    a.{4}*SQRT(1-POWER((a.{7}-b.{8})/
                                                       a.{2}, 2)) AS {1},
                                    (a.{7}-b.{8}) AS {5},
                                    b.{3},
                                    a.{4},
                                    b.{6},
                                    CAST(a.{7} AS INTEGER) AS {7},
                                    CASE  WHEN a.{7}-b.{8} > 0
                                          THEN POWER(a.{10}/(a.{7}-b.{8}),1.5)
                                          ELSE 0
                                          END AS {11},
                                    CASE  WHEN a.{7}-b.{8} <= a.{10}
                                          THEN a.{4}*SQRT(1-POWER((a.{7}-b.{8})/
                                                                 a.{10}, 2))
                                          ELSE 0
                                          END AS {12},
                                    a.{13},
                                    a.{14},
                                    a.{15},
                                    (a.{7}-b.{8})/a.{2} AS {16}
                                    """.format(
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            LENGTH_ZONE_FIELD + WAKE_NAME[0],
            idZone[WAKE_NAME],
            HEIGHT_FIELD,
            DISTANCE_BUILD_TO_POINT_FIELD,
            ID_POINT_X,
            Y_WALL,
            Y_POINT,
            ID_FIELD_STACKED_BLOCK,
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            WAKE_RELATIVE_POSITION_FIELD,
            UPPER_VERTICAL_THRESHOLD + CAVITY_NAME[0],
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
            POINT_RELATIVE_POSITION_FIELD + WAKE_NAME[0],
        ),
        STREET_CANYON_NAME: """b.{0},
                                    SIN(2*(a.{1}-PI()/2))*(0.5+(a.{9}-b.{13})*
                                    (a.{2}-(a.{9}-b.{13}))/
                                    (0.5*POWER(a.{2},2))) AS {3},
                                    1-POWER(COS(a.{1}-PI()/2),2)*(1+(a.{9}-b.{13})*
                                    (a.{2}-(a.{9}-b.{13}))/(POWER(0.5*a.{2},2))) AS {4},
                                    -ABS(0.5*(1-(a.{9}-b.{13})/(0.5*a.{2})))*
                                    (1-(a.{2}-(a.{9}-b.{13}))/(0.5*a.{2})) AS {5},
                                    a.{6},
                                    a.{7},
                                    a.{8},
                                    a.{10},
                                    CAST(a.{9} AS INTEGER) AS {9},
                                    a.{11},
                                    a.{12},
                                    a.{14},
                                    b.{15},
                                    a.{16},
                                    b.{13},
                                    a.{2},
                                    a.{7}*SQRT(1-POWER((a.{9}-b.{13})/a.{17}, 2)) AS {18}
                                    """.format(
            ID_POINT,
            UPWIND_FACADE_ANGLE_FIELD,
            LENGTH_ZONE_FIELD + STREET_CANYON_NAME[0],
            U,
            V,
            W,
            idZone[STREET_CANYON_NAME],
            UPSTREAM_HEIGHT_FIELD,
            BASE_HEIGHT_FIELD,
            Y_WALL,
            ID_POINT_X,
            UPWIND_FACADE_FIELD,
            MAX_CANYON_HEIGHT_FIELD,
            Y_POINT,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_POINT_Y,
            DOWNWIND_FACADE_FIELD,
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            UPPER_VERTICAL_THRESHOLD,
        ),
        ROOFTOP_PERP_NAME: """b.{0},
                                    a.{3}*SQRT(1-POWER(((a.{6}-b.{8})-a.{4}/2)/
                                                                     a.{4}, 2)) AS {5},
                                    b.{1},
                                    a.{2},
                                    CAST(a.{6} AS INTEGER) AS {6},
                                    a.{7}""".format(
            ID_POINT,
            idZone[ROOFTOP_PERP_NAME],
            HEIGHT_FIELD,
            ROOFTOP_PERP_HEIGHT,
            ROOFTOP_PERP_LENGTH,
            ROOFTOP_PERP_VAR_HEIGHT,
            Y_WALL,
            ID_POINT_X,
            Y_POINT,
        ),
    }

    # Last calculate the relative position of each point according
    # to the upper and lower part of the Rockle zones
    cursor.execute(
        ';'.join(
            [
                """
        {0};
        {1};
        {2};
        {3};
        DROP TABLE IF EXISTS {4};
        CREATE TABLE {4}
            AS SELECT   {5}
            FROM    {6} AS a RIGHT JOIN {7} AS b
                        ON a.{8} = b.{8} AND a.{9} = b.{9}
        """.format(
                    DataUtil.createIndex(
                        tableName=dicOfPrefixZoneLim[t],
                        fieldName=idZone[t],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfTempoOutput[t],
                        fieldName=idZone[t],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfPrefixZoneLim[t],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfTempoOutput[t],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    dicOfOutputTables[t],
                    varToKeepPoint[t],
                    dicOfPrefixZoneLim[t],
                    dicOfTempoOutput[t],
                    idZone[t],
                    ID_POINT_X,
                )
                for t in listTabYvalues
            ]
        )
    )

    # Special treatment for rooftop corners which have not been calculated previously
    cursor.execute(
        """DROP TABLE IF EXISTS {8};
                   CREATE TABLE {8}
                       AS SELECT {0},
                                ST_DISTANCE({7}, GEOM_CORNER_POINT)/
                                    COS(CASE WHEN   {6}<PI()/2
                                        THEN        {6}-ST_AZIMUTH({7}, GEOM_CORNER_POINT)
                                        ELSE        ST_AZIMUTH(GEOM_CORNER_POINT, {7})-{6}
                                        END
                                        )/
                                    {4}*{3} AS {5},
                                {1},
                                {2},
                                {10},
                                {11},
                                CAST(ST_Y(GEOM_CORNER_POINT) AS INTEGER) AS {12},
                                {13}
                        FROM {9}""".format(
            ID_POINT,
            idZone[ROOFTOP_PERP_NAME],
            HEIGHT_FIELD,
            ROOFTOP_CORNER_LENGTH,
            ROOFTOP_CORNER_FACADE_LENGTH,
            ROOFTOP_CORNER_VAR_HEIGHT,
            UPWIND_FACADE_ANGLE_FIELD,
            GEOM_FIELD,
            dicOfOutputTables[ROOFTOP_CORN_NAME],
            dicOfTempoOutput[ROOFTOP_CORN_NAME],
            UPWIND_FACADE_ANGLE_FIELD,
            ROOFTOP_WIND_FACTOR,
            Y_WALL,
            ID_POINT_X,
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS TEMPO_WAKE0, TEMPO_WAKE, {0},{1},{2}
                      """.format(
                ','.join(list(dicOfTempoOutput.values())),
                ','.join(list(dicOfPrefixZoneLim.values())),
                tempoCavity,
            )
        )

    return dicOfOutputTables, verticalLineTable


def affectsPointToVegZone(
    cursor, gridTable, dicOfVegRockleZoneTable, prefix=PREFIX_NAME
):
    """Affects each point to a vegetation Rockle zone and calculates the
    maximum vegetation height for each point.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            gridTable: String
                Name of the grid point table
            dicOfVegRockleZoneTable: Dictionary of vegetation Röckle zone tables
                Dictionary containing as key the vegetation Rockle zone name and
                as value the corresponding vegetation table name
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfOutputTables: dictionary of table name
                Dictionary having as key the type of vegetation Rockle zone and as value
                the name of the table containing points corresponding to the vegetation zone
    """
    print(
        """Affects each grid point to a vegetation Rockle zone and calculates 
          needed variables for 3D wind speed"""
    )

    # Name of the output tables
    dicOfOutputTables = {
        t: DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix=prefix), suffix='POINTS'
        )
        for t in dicOfVegRockleZoneTable
    }

    # Temporary tables (and prefix for temporary tables)
    maxHeightPointTable = 'MAX_VEG_HEIGHT_POINT_'

    # Calculate the max of the canopy height for each point and then keep each
    # intersection between point and zone
    cursor.execute(
        ';'.join(
            [
                """
        {12};
        {13};           
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT a.{1}, a.{2}, MAX(b.{3}) AS {7}
            FROM {5} AS a, {6} AS b
            WHERE    a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1})
            GROUP BY a.{2};
        {14};
        DROP TABLE IF EXISTS {11};
        CREATE TABLE {11}
            AS SELECT a.{1}, a.{2}, a.{7}, b.{4}, b.{8}, b.{9}, b.{10}
            FROM {0} AS a, {6} AS b
            WHERE    a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1})
           """.format(
                    maxHeightPointTable + t,
                    GEOM_FIELD,
                    ID_POINT,
                    VEGETATION_CROWN_TOP_HEIGHT,
                    ID_VEGETATION,
                    gridTable,
                    dicOfVegRockleZoneTable[t],
                    TOP_CANOPY_HEIGHT_POINT,
                    VEGETATION_ATTENUATION_FACTOR,
                    VEGETATION_CROWN_BASE_HEIGHT,
                    VEGETATION_CROWN_TOP_HEIGHT,
                    dicOfOutputTables[t],
                    DataUtil.createIndex(
                        gridTable, fieldName=GEOM_FIELD, isSpatial=True
                    ),
                    DataUtil.createIndex(
                        dicOfVegRockleZoneTable[t], fieldName=GEOM_FIELD, isSpatial=True
                    ),
                    DataUtil.createIndex(
                        maxHeightPointTable + t, fieldName=GEOM_FIELD, isSpatial=True
                    ),
                )
                for t in dicOfVegRockleZoneTable
            ]
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(','.join([maxHeightPointTable]))
        )

    return dicOfOutputTables


def removeBuildZonePoints(cursor, dicOfInitBuildZoneGridPoint, prefix=PREFIX_NAME):
    """Remove some of the Röckle zone points when there are specific
    zone overlapping. Currently, two major deletions are implemented:
        1. downwind building zone deletion: if a part of a building is entirely
        located within the vertical extend of an upward building cavity zone,
        its corresponding cavity, wake and street canyon zone are deleted
        (only the cavity part corresponding to the part of building covered by
         an other cavity is deleted)
        2. rooftop recirculation zone deletion: if the downward building of a street canyon
        is smaller or equal in size as the upward one, its rooftop recirculation
        zone is deleted.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfInitBuildZoneGridPoint: Dictionary of table name
                Dictionary containing as key the building Rockle zone name and
                as value the corresponding points
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfBuildZoneGridPoint: dictionary of table name
                Dictionary having as key the type of Rockle zone and as value
                the name of the table containing updated points corresponding
                to the zone"""
    print("""Remove some of the Röckle zone points""")

    # Name of the output tables
    dicOfBuildZoneGridPoint = {
        t: DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix=prefix), suffix='POINTS'
        )
        for t in dicOfInitBuildZoneGridPoint
    }

    # Temporary tables (and prefix for temporary tables)
    cavityFirstPointCoord = DataUtil.postfix('CAVITY_FIRST_POINT_COORD')
    cavityFirstPoint = DataUtil.postfix('CAVITY_FIRST_POINT')
    cavityRelations = DataUtil.postfix('CAVITY_RELATIONS')
    cavityRelationsAll = DataUtil.postfix('CAVITY_RELATIONS_ALL')
    cavityUpAndDown = DataUtil.postfix('CAVITY_UP_AND_DOWN')
    cavityWithoutUpAndDown = DataUtil.postfix('CAVITY_WITHOUT_UP_AND_DOWN')
    cavityWithoutDown = DataUtil.postfix('CAVITY_WITHOUT_DOWN')
    cavityPointsMinYwall = DataUtil.postfix('CAVITY_POINTS_MIN_YWALL')
    cavityFinalPoints = DataUtil.postfix('CAVITY_FINAL_POINTS')
    dicPointsToRemoveStreetCanyon = {
        ROOFTOP_PERP_NAME: 'TEMPO_POINTS_TO_REMOVE_' + ROOFTOP_PERP_NAME,
        ROOFTOP_CORN_NAME: 'TEMPO_POINTS_TO_REMOVE_' + ROOFTOP_CORN_NAME,
    }

    # 1. IDENTIFY BUILDINGS PARTS (X POSITION) HAVING THEIR DOWNSTREAM FACADE INCLUDED WITHIN AN
    #    UPSTREAM CAVITY ZONE
    # First identify the coordinate of the upstreamer point for each X coordinate to each cavity zone
    cursor.execute(
        """
           {5}{6}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT {1}, {2}, MAX({3}) AS {3}
               FROM {4}
               GROUP BY {1}, {2}
           """.format(
            cavityFirstPointCoord,
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            ID_POINT_Y,
            dicOfInitBuildZoneGridPoint[CAVITY_NAME],
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
        )
    )

    # Then identify the maximum height of the cavity zone for each of these points
    cursor.execute(
        """
           {9}{10}{11}{12}{13}{14}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT b.{1}, b.{2}, b.{3}, b.{6}, b.{7}
               FROM {4} AS a LEFT JOIN {5} AS b
               ON a.{6} = b.{6} AND a.{8} = b.{8}
           """.format(
            cavityFirstPoint,
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            Y_WALL,
            cavityFirstPointCoord,
            dicOfInitBuildZoneGridPoint[CAVITY_NAME],
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            ID_POINT_Y,
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT_Y,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                cavityFirstPointCoord, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityFirstPointCoord, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityFirstPointCoord, fieldName=ID_POINT_Y, isSpatial=False
            ),
        )
    )

    # Then identify potential relations between cavity zones (whether a cavity
    # zone is contained in an other)
    cursor.execute(
        """
           {13}{14}{15}{16}{17}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT b.{1}, b.{2} AS {3}, a.{2} AS {4}
               FROM {5} AS a LEFT JOIN {6} AS b
               ON a.{7} = b.{7}
               WHERE    a.{8} < b.{8} AND b.{9} > a.{9} + GREATEST({10}, {11}, {12})
               GROUP BY b.{1}, b.{2}, a.{2}
           """.format(
            cavityRelations,
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_DOWNSTREAM_STACKED_BLOCK,
            cavityFirstPoint,
            dicOfInitBuildZoneGridPoint[CAVITY_NAME],
            ID_POINT,
            UPPER_VERTICAL_THRESHOLD,
            Y_WALL,
            GEOMETRY_MERGE_TOLERANCE,
            SNAPPING_TOLERANCE,
            GEOMETRY_SIMPLIFICATION_DISTANCE,
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(cavityFirstPoint, fieldName=ID_POINT, isSpatial=False),
            DataUtil.createIndex(
                cavityFirstPoint, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfInitBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
    )

    # Add all remaining cavity zones (having no other cavity zone contained in their cavity zone)
    cursor.execute(
        """
           {7}{8}{9}{10}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.{1}, a.{2} AS {3}, NULL AS {4}
               FROM {5} AS a LEFT JOIN {6} AS b
               ON a.{1} = b.{1} AND a.{2} = b.{3}
               WHERE    b.{1} IS NULL AND b.{3} IS NULL
               UNION ALL
               SELECT {1}, {3}, {4}
               FROM {6}
           """.format(
            cavityRelationsAll,
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_DOWNSTREAM_STACKED_BLOCK,
            cavityFirstPoint,
            cavityRelations,
            DataUtil.createIndex(
                cavityFirstPoint, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityFirstPoint, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_UPSTREAM_STACKED_BLOCK, isSpatial=False
            ),
        )
    )

    # Identify cavity zones containing an other cavity BUT being also contained in an
    # upwind cavity
    cursor.execute(
        """
           {5}{6}{7}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.{1}, a.{2}
               FROM {3} AS a LEFT JOIN {3} AS b
               ON a.{1} = b.{4}
               WHERE b.{4} IS NOT NULL
               GROUP BY a.{1}, a.{2}
           """.format(
            cavityUpAndDown,
            ID_DOWNSTREAM_STACKED_BLOCK,
            ID_POINT_X,
            cavityRelations,
            ID_UPSTREAM_STACKED_BLOCK,
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_UPSTREAM_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_DOWNSTREAM_STACKED_BLOCK, isSpatial=False
            ),
        )
    )

    # Remove the previous identified cavity zones (ideally should be done with
    # a H2Network method since we do not take into account all cases...)
    cursor.execute(
        """
           {6}{7}{8}{9}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.{3}, a.{5}, a.{4}
               FROM {1} AS a LEFT JOIN {2} AS b
               ON a.{3} = b.{4} AND a.{5} = b.{5}
               WHERE b.{4} IS NULL
               GROUP BY a.{3}, a.{5}, a.{4}
           """.format(
            cavityWithoutUpAndDown,
            cavityRelationsAll,
            cavityUpAndDown,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_DOWNSTREAM_STACKED_BLOCK,
            ID_POINT_X,
            DataUtil.createIndex(
                cavityUpAndDown, fieldName=ID_DOWNSTREAM_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_UPSTREAM_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityUpAndDown, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityRelations, fieldName=ID_POINT_X, isSpatial=False
            ),
        )
    )

    # Remove the remaining cavity zones covered by an other cavity zone
    cursor.execute(
        """
           {5}{6}{7}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.{2}, a.{4}
               FROM {1} AS a LEFT JOIN {1} AS b
               ON a.{2} = b.{3} AND a.{4} = b.{4}
               WHERE b.{3} IS NULL
               GROUP BY a.{2}, a.{4}
           """.format(
            cavityWithoutDown,
            cavityWithoutUpAndDown,
            ID_UPSTREAM_STACKED_BLOCK,
            ID_DOWNSTREAM_STACKED_BLOCK,
            ID_POINT_X,
            DataUtil.createIndex(
                cavityUpAndDown, fieldName=ID_DOWNSTREAM_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityWithoutUpAndDown,
                fieldName=ID_UPSTREAM_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                cavityUpAndDown, fieldName=ID_POINT_X, isSpatial=False
            ),
        )
    )

    # 2. REMOVE ANY ZONE CORRESPONDING TO THESE PARTS OF BUILDINGS
    cavityJoinFields = {
        STREET_CANYON_NAME: [ID_UPSTREAM_STACKED_BLOCK, ID_POINT_X],
        WAKE_NAME: [ID_FIELD_STACKED_BLOCK, ID_POINT_X],
        CAVITY_NAME: [ID_UPSTREAM_STACKED_BLOCK, ID_POINT_X],
    }
    # Take all points from a 't' zone which have not been deleted by a cavity zone
    cursor.execute(
        ';'.join(
            [
                """
         {6}{7}{8}{9}
         DROP TABLE IF EXISTS {0};
         CREATE TABLE {0}
             AS SELECT a.*
             FROM {1} AS a RIGHT JOIN {2} AS b
             ON a.{5} = b.{4} AND a.{3} = b.{3}
             WHERE a.{10} IS NOT NULL
         """.format(
                    dicOfBuildZoneGridPoint[t],
                    dicOfInitBuildZoneGridPoint[t],
                    cavityWithoutDown,
                    ID_POINT_X,
                    ID_UPSTREAM_STACKED_BLOCK,
                    cavityJoinFields[t][0],
                    DataUtil.createIndex(
                        cavityWithoutDown,
                        fieldName=ID_UPSTREAM_STACKED_BLOCK,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=cavityJoinFields[t][0],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        cavityWithoutDown, fieldName=ID_POINT_X, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=cavityJoinFields[t][1],
                        isSpatial=False,
                    ),
                    ID_POINT,
                )
                for t in cavityJoinFields.keys()
            ]
        )
    )

    # For each point, several zones may overlay, we need to identify those
    # coming from the more downstream one (y_wall min)
    cursor.execute(
        """
           {4}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT {1}, MIN({2}) AS {2}
               FROM {3}
               GROUP BY {1}
           """.format(
            cavityPointsMinYwall,
            ID_POINT,
            Y_WALL,
            dicOfBuildZoneGridPoint[CAVITY_NAME],
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT,
                isSpatial=False,
            ),
        )
    )

    # At the end only one point per position is conserved, the points from the more downstream zone
    cursor.execute(
        """
           {5}{6}{7}{8}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.*
               FROM {1} AS a RIGHT JOIN {2} AS b
               ON a.{3} = b.{3} AND a.{4} = b.{4};
           DROP TABLE IF EXISTS {1};
           ALTER TABLE {0} RENAME TO {1};
           """.format(
            cavityFinalPoints,
            dicOfBuildZoneGridPoint[CAVITY_NAME],
            cavityPointsMinYwall,
            ID_POINT,
            Y_WALL,
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[CAVITY_NAME],
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                cavityPointsMinYwall, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[CAVITY_NAME], fieldName=Y_WALL, isSpatial=False
            ),
            DataUtil.createIndex(
                cavityPointsMinYwall, fieldName=Y_WALL, isSpatial=False
            ),
        )
    )

    # 3. REMOVE ROOFTOP ZONES WHICH ARE DOWNSTREAM A CANYON IF THE CANYON
    # IS AS HIGH AS THE BUILDING WHERE THE ROOFTOP ZONE SHOULD APPEAR
    # (note that street canyon have been previously updated)
    streetCanyonJoinFields = {
        ROOFTOP_PERP_NAME: [UPWIND_FACADE_FIELD, ID_POINT_X],
        ROOFTOP_CORN_NAME: [UPWIND_FACADE_FIELD, ID_POINT_X],
    }
    # Identify the rooftop zones to remove
    cursor.execute(
        ';'.join(
            [
                """ 
           {8};
           {9};
           {10};
           {11};
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT MIN(b.{1}) AS {1}, MIN(b.{2}) AS {2}, MIN(b.{3}) AS {3}
               FROM {4} AS a RIGHT JOIN {5} AS b ON a.{2} = b.{2} AND a.{3} = b.{3}
               WHERE b.{6} <= a.{7}
               GROUP BY b.{3}, b.{2}
           """.format(
                    dicPointsToRemoveStreetCanyon[t],
                    ID_POINT,
                    UPWIND_FACADE_FIELD,
                    ID_POINT_X,
                    dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                    dicOfInitBuildZoneGridPoint[t],
                    HEIGHT_FIELD,
                    MAX_CANYON_HEIGHT_FIELD,
                    DataUtil.createIndex(
                        dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                )
                for t in streetCanyonJoinFields.keys()
            ]
        )
    )

    # Remove points in rooftop zones
    cursor.execute(
        ';'.join(
            [
                """ 
           {5};
           {6};
           {7};
           {8};
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.*
               FROM {1} AS a LEFT JOIN {2} AS b ON a.{3} = b.{3} AND a.{4} = b.{4}
               WHERE b.{3} IS NULL AND b.{4} IS NULL
           """.format(
                    dicOfBuildZoneGridPoint[t],
                    dicOfInitBuildZoneGridPoint[t],
                    dicPointsToRemoveStreetCanyon[t],
                    streetCanyonJoinFields[t][0],
                    streetCanyonJoinFields[t][1],
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=streetCanyonJoinFields[t][0],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfInitBuildZoneGridPoint[t],
                        fieldName=streetCanyonJoinFields[t][1],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicPointsToRemoveStreetCanyon[t],
                        fieldName=streetCanyonJoinFields[t][0],
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicPointsToRemoveStreetCanyon[t],
                        fieldName=streetCanyonJoinFields[t][1],
                        isSpatial=False,
                    ),
                )
                for t in streetCanyonJoinFields.keys()
            ]
        )
    )

    # Rename tables which has not been modified to the "updated" name
    nonModifiedTables = (
        set(dicOfInitBuildZoneGridPoint.keys())
        - set(cavityJoinFields.keys())
        - set(streetCanyonJoinFields.keys())
    )
    cursor.execute(
        ';'.join(
            [
                """ 
           DROP TABLE IF EXISTS {0};
           ALTER TABLE {1} RENAME TO {0}
           """.format(dicOfBuildZoneGridPoint[t], dicOfInitBuildZoneGridPoint[t])
                for t in nonModifiedTables
            ]
        )
    )
    if not DEBUG:
        # Remove intermediate tables
        listToRemove = list(dicPointsToRemoveStreetCanyon.values()) + [
            cavityFirstPointCoord,
            cavityFirstPoint,
            cavityRelations,
            cavityRelationsAll,
            cavityUpAndDown,
            cavityWithoutUpAndDown,
            cavityWithoutDown,
            cavityPointsMinYwall,
            cavityFinalPoints,
        ]
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(','.join(listToRemove))
        )

    return dicOfBuildZoneGridPoint


def manageBackwardZones(
    cursor,
    dicOfBuildZoneGridPoint,
    cavity2dInitPoints,
    wake2dInitPoints,
    streetCanyonTable,
    gridTable,
    prefix,
    meshSize=MESH_SIZE,
    dz=DZ,
):
    """A building having a horizontal piece its upwind facade entirely (vertically)
    located within the cavity zone of an upwind taller building will create:
        -> a backward zone system within the 2 buildings (ie. the downwind building
                                                          cavity and wake zones will go
                                                          upstream - also downstream
                                                          if the building not entirely
                                                          in the cavity zone of upstream building),
        -> the removal of the street canyon at these locations (upstream the facade within cavity)
        -> the removal of the rooftop zones at these locations (downstream the facade within cavity)

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfBuildZoneGridPoint: Dictionary of table name
                Dictionary containing as key the building Rockle zone name and
                as value the corresponding points
            cavity2dInitPoints: String
                Cavity zone points before any zone removing
            wake2dInitPoints: String
                Wake zone points before any zone removing
            streetCanyonTable: String
                Name of the table containing the street canyon zones
            gridTable: String
                Name of the grid point table
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name
            meshSize: float, default MESH_SIZE
                Resolution (in meter) of the grid
            dz: float, default DZ
                Resolution (in meter) of the grid in the vertical direction

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfBuildZoneGridPoint: dictionary of table name
                Dictionary having as key the type of Rockle zone and as value
                the name of the table containing updated street canyon zones and
                new backward zone points
            facadeWithinCavity: string
                Name of the table containing the "facade" ID_POINT and height
                which will be useful for assigning a weighting to backward zones"""
    print("""Creates backward zones""")

    # Name of the output tables
    tables2calculate = [CAVITY_BACKWARD_NAME, WAKE_BACKWARD_NAME]
    for t in tables2calculate:
        dicOfBuildZoneGridPoint[t] = DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix=prefix), suffix='POINTS'
        )
    facadeWithinCavity = DataUtil.prefix(
        tableName='FACADE_WITHIN_CAVITY', prefix=prefix
    )

    # Temporary tables (and prefix for temporary tables)
    canyonLastPointCoord = DataUtil.postfix('CANYON_LAST_POINT_COORD')
    impactedStackedBlocs = DataUtil.postfix('IMPACTED_STACKED_BLOCKS')
    rooftop_tables = [ROOFTOP_PERP_NAME, ROOFTOP_CORN_NAME]
    tempoZoneTables = tables2calculate + rooftop_tables + [STREET_CANYON_NAME]
    dicOfTempoBackPoints = {
        t: DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix='TEMPO'), suffix='POINTS'
        )
        for t in tempoZoneTables
    }

    # 1. IDENTIFY BUILDINGS PARTS (X POSITION) HAVING THEIR UPSTREAM FACADE
    # INCLUDED WITHIN AN UPSTREAM CAVITY ZONE AND CREATE BACKWARD ZONES
    # First identify the coordinate of the downstreamer point for each X
    # coordinate of each street canyon zone and keep only those having the
    # cavity zone being higher than the downwind building height
    cursor.execute(
        """
           {6}{7}
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT    {1}, {2}, {3}, MIN({4}) AS {4}
               FROM {5}
               GROUP BY {1}, {3};
           {8}{9}{10}{11}{12}{13}
           DROP TABLE IF EXISTS {14};
           CREATE TABLE {14}
               AS SELECT   a.{1}, a.{2}, a.{3}, a.{4}, b.{17}-b.{18} AS {17}, b.{18}, b.{19}, 
                           TRUNC(b.{16} / {21}) + 1 AS {22}, b.{20}
               FROM {0} AS a LEFT JOIN {5} AS b
               ON a.{1} = b.{1} AND a.{3} = b.{3} AND a.{4} = b.{4}
               WHERE b.{15} > b.{16}
           """.format(
            canyonLastPointCoord,
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            ID_FIELD_CANYON,
            ID_POINT_Y,
            dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_FIELD_CANYON,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                canyonLastPointCoord, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                canyonLastPointCoord, fieldName=ID_POINT_Y, isSpatial=False
            ),
            DataUtil.createIndex(
                canyonLastPointCoord, fieldName=ID_FIELD_CANYON, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_POINT_Y,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_FIELD_CANYON,
                isSpatial=False,
            ),
            facadeWithinCavity,
            UPPER_VERTICAL_THRESHOLD,
            MAX_CANYON_HEIGHT_FIELD,
            Y_WALL,
            LENGTH_ZONE_FIELD + STREET_CANYON_NAME[0],
            UPWIND_FACADE_FIELD,
            ID_POINT,
            dz,
            ID_POINT_Z,
        )
    )

    # Then get the stacked blocks concerned by the backward cavity and wake zones
    # and revert the cavity and wake wind speed factors from downwind to upwind stacked block
    # and finally join the ID_POINT of each of the ID_X / ID_Y point
    var2keepSpe = {
        CAVITY_BACKWARD_NAME: """a.{0}, a.{1}
                                         """.format(
            LENGTH_ZONE_FIELD + CAVITY_NAME[0],
            POINT_RELATIVE_POSITION_FIELD + CAVITY_NAME[0],
        ),
        WAKE_BACKWARD_NAME: """a.{0}, a.{1}
                                       """.format(
            WAKE_RELATIVE_POSITION_FIELD, UPPER_VERTICAL_THRESHOLD + CAVITY_NAME[0]
        ),
    }
    tab2revert = {
        CAVITY_BACKWARD_NAME: cavity2dInitPoints,
        WAKE_BACKWARD_NAME: wake2dInitPoints,
    }
    cursor.execute(
        ';'.join(
            [
                """
           {0}{1}
           DROP TABLE IF EXISTS {2};
           CREATE TABLE {2}
               AS SELECT a.{3}, a.{4}, a.{5}, a.{6}, b.{7}, a.{8}, a.{30}
               FROM {9} AS a LEFT JOIN {10} AS b
               ON a.{11} = b.{11};
           {12}{13}{14}{15}
           DROP TABLE IF EXISTS {16};
           CREATE TABLE {16}
               AS SELECT b.{4}, a.{17}, {18}, a.{19}, a.{20}, b.{3}, b.{6},
                         b.{5} + TRUNC(a.{19}/{21}) AS {5}, b.{30}
               FROM {22} AS a LEFT JOIN {2} AS b
               ON a.{4} = b.{7} AND a.{3} = b.{3}
               WHERE a.{19} < b.{8} + {21};
           {23}{24}{25}{26}
           DROP TABLE IF EXISTS {27};
           CREATE TABLE {27}
               AS SELECT a.*, b.{28}
               FROM {16} AS a LEFT JOIN {29} AS b
               ON a.{3} = b.{3} AND a.{5} = b.{5}
           """.format(
                    DataUtil.createIndex(
                        facadeWithinCavity, fieldName=ID_FIELD_CANYON, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        streetCanyonTable, fieldName=ID_FIELD_CANYON, isSpatial=False
                    ),
                    impactedStackedBlocs,
                    ID_POINT_X,
                    ID_FIELD_STACKED_BLOCK,
                    ID_POINT_Y,
                    Y_WALL,
                    ID_DOWNSTREAM_STACKED_BLOCK,
                    LENGTH_ZONE_FIELD + STREET_CANYON_NAME[0],
                    facadeWithinCavity,
                    streetCanyonTable,
                    ID_FIELD_CANYON,
                    DataUtil.createIndex(
                        tab2revert[t], fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        tab2revert[t], fieldName=ID_POINT_X, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        impactedStackedBlocs,
                        fieldName=ID_DOWNSTREAM_STACKED_BLOCK,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        impactedStackedBlocs, fieldName=ID_POINT_X, isSpatial=False
                    ),
                    dicOfTempoBackPoints[t],
                    UPPER_VERTICAL_THRESHOLD,
                    var2keepSpe[t],
                    DISTANCE_BUILD_TO_POINT_FIELD,
                    HEIGHT_FIELD,
                    meshSize,
                    tab2revert[t],
                    DataUtil.createIndex(
                        dicOfTempoBackPoints[t], fieldName=ID_POINT_X, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        dicOfTempoBackPoints[t], fieldName=ID_POINT_Y, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        gridTable, fieldName=ID_POINT_X, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        gridTable, fieldName=ID_POINT_Y, isSpatial=False
                    ),
                    dicOfBuildZoneGridPoint[t],
                    ID_POINT,
                    gridTable,
                    UPWIND_FACADE_FIELD,
                )
                for t in tables2calculate
            ]
        )
    )

    # 2. REMOVE STREET CANYON POINTS WHERE THE DOWNWIND FACADE OF THE CANYON IS
    # ENTIRELY IN THE CAVITY ZONE OF THE UPSTREAM BUILDING
    cursor.execute(
        """
           {0}{1}{2}{3}
           DROP TABLE IF EXISTS {4};
           CREATE TABLE {4}
               AS SELECT a.*
               FROM {5} AS a LEFT JOIN {6} AS b
               ON a.{7} = b.{7} AND a.{8} = b.{8}
               WHERE b.{7} IS NULL AND b.{8} IS NULL;
           DROP TABLE IF EXISTS {5};
           ALTER TABLE {4} RENAME TO {5};
       """.format(
            DataUtil.createIndex(
                facadeWithinCavity, fieldName=ID_FIELD_CANYON, isSpatial=False
            ),
            DataUtil.createIndex(
                facadeWithinCavity, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_FIELD_CANYON,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
                fieldName=ID_POINT_X,
                isSpatial=False,
            ),
            dicOfTempoBackPoints[STREET_CANYON_NAME],
            dicOfBuildZoneGridPoint[STREET_CANYON_NAME],
            facadeWithinCavity,
            ID_FIELD_CANYON,
            ID_POINT_X,
        )
    )

    # 3. REMOVE ROOFTOP POINTS WHERE THE DOWNWIND FACADE OF THE CANYON IS
    # ENTIRELY IN THE CAVITY ZONE OF THE UPSTREAM BUILDING
    cursor.execute(
        ';'.join(
            [
                """
           {0}{1}{2}{3}
           DROP TABLE IF EXISTS {4};
           CREATE TABLE {4}
               AS SELECT a.*
               FROM {5} AS a LEFT JOIN {6} AS b
               ON a.{7} = b.{7} AND a.{8} = b.{8}
               WHERE b.{7} IS NULL AND b.{8} IS NULL;
           DROP TABLE IF EXISTS {5};
           ALTER TABLE {4} RENAME TO {5};
       """.format(
                    DataUtil.createIndex(
                        facadeWithinCavity,
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        facadeWithinCavity, fieldName=ID_POINT_X, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        dicOfBuildZoneGridPoint[t],
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        dicOfBuildZoneGridPoint[t],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    dicOfTempoBackPoints[t],
                    dicOfBuildZoneGridPoint[t],
                    facadeWithinCavity,
                    UPWIND_FACADE_FIELD,
                    ID_POINT_X,
                )
                for t in rooftop_tables
            ]
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        listToRemove = list(dicOfTempoBackPoints.values()) + [
            canyonLastPointCoord,
            impactedStackedBlocs,
        ]
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(','.join(listToRemove))
        )

    return dicOfBuildZoneGridPoint, facadeWithinCavity


def calculates3dBuildWindFactor(
    cursor, dicOfBuildZoneGridPoint, dz=DZ, prefix=PREFIX_NAME
):
    """Calculates the 3D wind speed factors for each building zone.

        Parameters
        _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    dicOfBuildZoneGridPoint: Dictionary of Rockle zone tables
        Dictionary having as key the type of Rockle zone and as value
        the name of the table containing points corresponding to the zone
    dz: float, default DZ
        Resolution (in meter) of the grid in the vertical direction
    prefix: String, default PREFIX_NAME
        Prefix to add to the output table name

        Returns
        _ _ _ _ _ _ _ _ _ _

    dicOfOutputTables: dictionary of table name
        Dictionary having as key the type of Rockle zone and as value
        the name of the table containing points corresponding to the zone
        and wind speed factor
    maxHeight: float
        Height of the highest Rôckle zone in the study area"""
    print(
        'Calculates the 3D wind speed factor value for each point of each BUILDING zone'
    )

    # Name of the output tables
    dicOfOutputTables = {
        t: DataUtil.postfix(
            tableName=DataUtil.prefix(tableName=t, prefix=prefix),
            suffix='POINTS_BUILD_3D',
        )
        for t in dicOfBuildZoneGridPoint
    }

    # Temporary tables (and prefix for temporary tables)
    zValueTable = DataUtil.postfix('Z_VALUES')

    # Identify the maximum height where wind speed may be affected by building obstacles
    maxHeightQuery = {
        DISPLACEMENT_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
        DISPLACEMENT_VORTEX_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(
            UPPER_VERTICAL_THRESHOLD
        ),
        CAVITY_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
        WAKE_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
        STREET_CANYON_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
        ROOFTOP_PERP_NAME: 'MAX({0}+{1}) AS MAX_HEIGHT'.format(
            ROOFTOP_PERP_VAR_HEIGHT, HEIGHT_FIELD
        ),
        ROOFTOP_CORN_NAME: 'MAX({0}+{1}) AS MAX_HEIGHT'.format(
            ROOFTOP_CORNER_VAR_HEIGHT, HEIGHT_FIELD
        ),
        CAVITY_BACKWARD_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
        WAKE_BACKWARD_NAME: 'MAX({0}) AS MAX_HEIGHT'.format(UPPER_VERTICAL_THRESHOLD),
    }
    cursor.execute(
        """ SELECT MAX(MAX_HEIGHT) AS MAX_HEIGHT
                       FROM (SELECT {0})
                       """.format(
            ' UNION ALL SELECT '.join(
                [
                    maxHeightQuery[t] + ' FROM ' + dicOfBuildZoneGridPoint[t]
                    for t in dicOfBuildZoneGridPoint
                ]
            )
        )
    )
    maxHeight = cursor.fetchall()[0][0]

    # Creates the table of z levels impacted by building obstacles (start at dz/2)
    if maxHeight:
        listOfZ = [
            str(i)
            for i in np.arange(
                float(dz) / 2, float(dz) / 2 + math.trunc(maxHeight / dz) * dz, dz
            )
        ]
        cursor.execute(
            """
                   DROP TABLE IF EXISTS {0};
                   CREATE TABLE {0}({2} SERIAL, {3} DOUBLE);
                   INSERT INTO {0} VALUES (NULL, {1})
                   """.format(zValueTable, '), (NULL, '.join(listOfZ), ID_POINT_Z, Z)
        )
    else:
        cursor.execute(
            """
                   DROP TABLE IF EXISTS {0};
                   CREATE TABLE {0}({1} SERIAL, {2} DOUBLE);
                   """.format(zValueTable, ID_POINT_Z, Z)
        )

    # Defines the calculation and columns to keep for each zone
    calcQuery = {
        DISPLACEMENT_NAME: """
                 b.{0},
                 {1}*POWER(b.{2}/a.{3},{4})*a.{7} AS {7},
                 {1}*POWER(b.{2}/a.{3},{4})*a.{5} AS {5},
                 CASE  WHEN a.{7} = 0 AND a.{5} = 0
                       THEN 0
                       ELSE -0*{1}*POWER((a.{3}-b.{2})/a.{3},0.5)*(ABS(a.{7})/POWER(POWER(a.{7},2)+POWER(a.{5},2),0.5))
                       END AS {8},
                 a.{6},
                 a.{3}
                 """.format(ID_POINT_Z, C_DZ, Z, HEIGHT_FIELD, P_DZ, V, ID_POINT, U, W),
        DISPLACEMENT_VORTEX_NAME: """
                 b.{0},
                 -(0.6*COS(PI()*b.{1}/(0.5*a.{2}))+0.05)*0.6*SIN(PI()*a.{3}) AS {4},
                 -0.1*COS(PI()*a.{3})-0.05 AS {5},
                 a.{6},
                 a.{2}
                 """.format(
            ID_POINT_Z,
            Z,
            HEIGHT_FIELD,
            POINT_RELATIVE_POSITION_FIELD + DISPLACEMENT_VORTEX_NAME[0],
            V,
            W,
            ID_POINT,
        ),
        CAVITY_NAME: """
                 b.{0},
                 -POWER(1-a.{1}/POWER(1-POWER(b.{2}/a.{3},2),0.5),2)*POWER(a.{6},0)*POWER(a.{8},0.5) AS {4},
                 a.{5},
                 a.{3},
                 0*POWER(a.{1}/POWER(1-POWER(b.{2}/a.{3},2),0.5),0.5) AS {9},
                 0*POWER(1-a.{1}/POWER(1-POWER(b.{2}/a.{3},2),0.5),2) AS {11}
                 """.format(
            ID_POINT_Z,
            POINT_RELATIVE_POSITION_FIELD + CAVITY_NAME[0],
            Z,
            HEIGHT_FIELD,
            V,
            ID_POINT,
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
            U,
            POINT_RELATIVE_POSITION_FIELD + WAKE_NAME[0],
            W,
        ),
        WAKE_NAME: """
                 b.{0},
                 1-a.{1}*POWER(POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {4},
                 1-a.{1}*POWER(POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {6},
                 1-a.{1}*POWER(POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {7},
                 a.{5},
                 a.{3},
                 1-a.{1}*POWER(POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5)*POWER(a.{8},0)*POWER(a.{10},0) AS {11},
                 0*a.{1}*POWER(POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5)*a.{9}*a.{10} AS {12}
                 """.format(
            ID_POINT_Z,
            WAKE_RELATIVE_POSITION_FIELD,
            Z,
            HEIGHT_FIELD,
            V_WEIGHT,
            ID_POINT,
            U_WEIGHT,
            W_WEIGHT,
            COS_BLOCK_AZIMUTH,
            SIN_BLOCK_AZIMUTH,
            DOWNSTREAM_X_RELATIVE_POSITION,
            V,
            U,
        ),
        STREET_CANYON_NAME: """
                 b.{0},
                 a.{1},
                 a.{2},
                 a.{3},
                 a.{4},
                 a.{5} AS {6}
                 """.format(
            ID_POINT_Z, U, V, W, ID_POINT, UPSTREAM_HEIGHT_FIELD, HEIGHT_FIELD
        ),
        ROOFTOP_PERP_NAME: """
                b.{0},
                -POWER((a.{1}+a.{2}-b.{3})/{4},{5})*ABS(a.{1}+a.{2}-b.{3})/a.{2} AS {6},
                a.{7},
                a.{1}
                """.format(
            ID_POINT_Z,
            HEIGHT_FIELD,
            ROOFTOP_PERP_VAR_HEIGHT,
            Z,
            Z_REF,
            P_RTP,
            V,
            ID_POINT,
        ),
        ROOFTOP_CORN_NAME: """
                b.{0},
                -a.{8}*SIN(2*a.{9})*POWER((a.{1}+a.{2}-b.{3})/{4},{5})
                *ABS(a.{1}+a.{2}-b.{3})/a.{2} AS {6},
                -a.{8}*POWER(SIN(a.{9}),2)*POWER((a.{1}+a.{2}-b.{3})/{4},{5})
                *ABS(a.{1}+a.{2}-b.{3})/a.{2} AS {10},
                a.{7},
                a.{1}
                """.format(
            ID_POINT_Z,
            HEIGHT_FIELD,
            ROOFTOP_CORNER_VAR_HEIGHT,
            Z,
            Z_REF,
            P_RTP,
            U,
            ID_POINT,
            ROOFTOP_WIND_FACTOR,
            UPWIND_FACADE_ANGLE_FIELD,
            V,
        ),
        CAVITY_BACKWARD_NAME: """
                 b.{0},
                 POWER(1-a.{1}/POWER(1-POWER(b.{2}/a.{3},2),0.5),2) AS {4},
                 a.{5},
                 a.{3},
                 a.{6},
                 a.{7}
                 """.format(
            ID_POINT_Z,
            POINT_RELATIVE_POSITION_FIELD + CAVITY_NAME[0],
            Z,
            HEIGHT_FIELD,
            V,
            ID_POINT,
            ID_POINT_X,
            UPWIND_FACADE_FIELD,
        ),
        WAKE_BACKWARD_NAME: """
                 b.{0},
                 -1+POWER(a.{1}*POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {4},
                 -1+POWER(a.{1}*POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {6},
                 -1+POWER(a.{1}*POWER(1-POWER(b.{2}/a.{3},2),0.5),1.5) AS {7},
                 a.{5},
                 a.{3},
                 a.{8},
                 a.{9}
                 """.format(
            ID_POINT_Z,
            WAKE_RELATIVE_POSITION_FIELD,
            Z,
            HEIGHT_FIELD,
            V,
            ID_POINT,
            U,
            W,
            ID_POINT_X,
            UPWIND_FACADE_FIELD,
        ),
    }

    # Defines the WHERE clause (on z-axis values) for each point of each zone
    whereQuery = {
        DISPLACEMENT_NAME: 'b.{0} < a.{1}'.format(Z, UPPER_VERTICAL_THRESHOLD),
        DISPLACEMENT_VORTEX_NAME: 'b.{0} < a.{1}'.format(Z, UPPER_VERTICAL_THRESHOLD),
        CAVITY_NAME: 'b.{0} < a.{1}'.format(Z, UPPER_VERTICAL_THRESHOLD),
        WAKE_NAME: 'b.{0} < a.{1} AND b.{0} >= a.{2}'.format(
            Z, UPPER_VERTICAL_THRESHOLD, UPPER_VERTICAL_THRESHOLD + CAVITY_NAME[0]
        ),
        STREET_CANYON_NAME: """b.{0} < a.{1} 
                                        AND b.{0} < a.{2}""".format(
            Z, UPPER_VERTICAL_THRESHOLD, MAX_CANYON_HEIGHT_FIELD
        ),
        ROOFTOP_PERP_NAME: """b.{0} < a.{1}+a.{2}
                                        AND b.{0} > a.{1}""".format(
            Z, HEIGHT_FIELD, ROOFTOP_PERP_VAR_HEIGHT
        ),
        ROOFTOP_CORN_NAME: """b.{0} < a.{1}+a.{2} 
                                        AND b.{0} > a.{1}""".format(
            Z, HEIGHT_FIELD, ROOFTOP_CORNER_VAR_HEIGHT
        ),
        CAVITY_BACKWARD_NAME: 'b.{0} < a.{1}'.format(Z, UPPER_VERTICAL_THRESHOLD),
        WAKE_BACKWARD_NAME: 'b.{0} < a.{1} AND b.{0} >= a.{2}'.format(
            Z, UPPER_VERTICAL_THRESHOLD, UPPER_VERTICAL_THRESHOLD + CAVITY_NAME[0]
        ),
    }
    # Execute the calculation
    cursor.execute(
        ';'.join(
            [
                """ DROP TABLE IF EXISTS {0};
            CREATE TABLE {0}
                AS SELECT {1}, a.{5}
                FROM {2} AS a, {3} AS b
                WHERE {4}
                """.format(
                    dicOfOutputTables[t],
                    calcQuery[t],
                    dicOfBuildZoneGridPoint[t],
                    zValueTable,
                    whereQuery[t],
                    Y_WALL,
                )
                for t in dicOfBuildZoneGridPoint
            ]
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(zValueTable)
        )

    return dicOfOutputTables, maxHeight


def calculates3dVegWindFactor(
    cursor, dicOfVegZoneGridPoint, sketchHeight, z0, d, dz=DZ, prefix=PREFIX_NAME
):
    """Calculates the 3D wind speed factors for each zone according to
    Nelson et al. (2009) method. Note that for vegetation located in
    open areas (Equations 8 and 9), the displacement height is defined by
    Equation 18a from Hanna and Britter (2002), considering that lambda_f=0.05
    and Hr being the height of the vegetation at this specific location.

    References:
        Hanna, SR, et RE Britter. « Wind flow and vapor cloud dispersion at
    industrial sites. Am. Inst ». Chem Eng, New York, 2002.

        Nelson, Matthew, Michael Williams, Dragan Zajic, Michael Brown, et
    Eric Pardyjak. Evaluation of an urban vegetative canopy scheme and
    impact on plume dispersion, 2009.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfVegZoneGridPoint: Dictionary of vegetation Rockle zone tables
                Dictionary having as key the type of vegetation Rockle zone and as value
                the name of the table containing points corresponding to the zone
            sketchHeight: float
                Height of the sketch (m)
            z0: float
                Value of the study area roughness height
            d: float
                Value of the study area displacement length
            dz: float, default DZ
                Resolution (in meter) of the grid in the vertical direction
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            vegetationWeightFactorTable: String
                Name of the table containing the weighting factor for each 3D point
                located in a vegetation zone"""
    print(
        'Calculates the 3D wind speed factor value for each point of each VEGETATION zone'
    )

    # Output base name
    outputBaseName = 'VEGETATION_WEIGHTING_FACTORS'

    # Name of the output table
    vegetationWeightFactorTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Temporary tables (and prefix for temporary tables)
    zValueTable = DataUtil.postfix('Z_VALUES')
    dicOfTempoTables = {
        t: DataUtil.postfix(tableName=t, suffix='TEMPO_3DPOINTS')
        for t in dicOfVegZoneGridPoint
    }
    tempoAllVeg = DataUtil.postfix('TEMPO_ALL_VEG')

    # Creates the table of z levels of the sketch
    listOfZ = [
        str(i)
        for i in np.arange(
            float(dz) / 2, float(dz) / 2 + math.trunc(sketchHeight / dz) * dz, dz
        )
    ]
    cursor.execute(
        """
            DROP TABLE IF EXISTS {0};
            CREATE TABLE {0}({2} SERIAL, {3} DOUBLE);
            INSERT INTO {0} VALUES (NULL, {1})
               """.format(zValueTable, '), (NULL, '.join(listOfZ), ID_POINT_Z, Z)
    )

    # Calculation of the wind speed depending on vegetation location (open or building zone)
    # d is actually calculated by Equation 18a from Hanna and Britter (2002)
    calcQuery = {
        VEGETATION_OPEN_NAME: """ CASE WHEN   a.{0}>b.{3}
                    THEN    LOG((a.{0} - 3 * 0.05 * {2}) / {1}) / LOG(a.{0} / {1})
                    ELSE    CASE WHEN   a.{0} > b.{3} OR a.{0} < b.{4}
                            THEN        LOG((b.{2} - 3 * 0.05 * {2}) / {1}) / LOG(a.{0} / {1})
                            ELSE        LOG((b.{3} - 3 * 0.05 * {2}) / {1}) / LOG(a.{0} / {1}) * EXP(b.{5} * (a.{0} / b.{2} - 1))
                    END
                END
            """.format(
            Z,
            z0,
            TOP_CANOPY_HEIGHT_POINT,
            VEGETATION_CROWN_TOP_HEIGHT,
            VEGETATION_CROWN_BASE_HEIGHT,
            VEGETATION_ATTENUATION_FACTOR,
        ),
        VEGETATION_BUILT_NAME: """ CASE WHEN   a.{0} > b.{3} OR a.{0} < b.{4}
                    THEN    LOG(b.{2} / {1}) / LOG(a.{0} / {1})
                    ELSE    LOG(b.{2} / {1}) / LOG(a.{0} / {1}) * EXP(b.{5} * (a.{0} / b.{2} - 1))
                END
            """.format(
            Z,
            z0,
            TOP_CANOPY_HEIGHT_POINT,
            VEGETATION_CROWN_TOP_HEIGHT,
            VEGETATION_CROWN_BASE_HEIGHT,
            VEGETATION_ATTENUATION_FACTOR,
        ),
    }

    whereQuery = {
        VEGETATION_OPEN_NAME: 'WHERE a.{0} > 0'.format(Z),
        VEGETATION_BUILT_NAME: """ WHERE a.{0} < b.{1} AND a.{0} > 0
                                          """.format(Z, TOP_CANOPY_HEIGHT_POINT),
    }

    # Initialize the wind speed field depending on vegetation type and height
    cursor.execute(
        ';'.join(
            [
                """
           {10};
           {11};           
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT b.{9}, a.{1}, {2} AS {3}
               FROM {4} AS a, {5} AS b
               {8};
           {12};
           UPDATE {0} SET {3} = 1 WHERE {3} > 1;
           UPDATE {0} SET {3} = 0 WHERE {3} < 0;
                   """.format(
                    dicOfTempoTables[t],
                    ID_POINT_Z,
                    calcQuery[t],
                    VEGETATION_FACTOR,
                    zValueTable,
                    dicOfVegZoneGridPoint[t],
                    Z,
                    TOP_CANOPY_HEIGHT_POINT,
                    whereQuery[t],
                    ID_POINT,
                    DataUtil.createIndex(
                        tableName=zValueTable, fieldName=Z, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfVegZoneGridPoint[t],
                        fieldName=TOP_CANOPY_HEIGHT_POINT,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfTempoTables[t],
                        fieldName=VEGETATION_FACTOR,
                        isSpatial=False,
                    ),
                )
                for t in dicOfTempoTables
            ]
        )
    )

    # Gather zone points in a single vegetation table and keep the minimum value
    # in case there are several vegetation layers
    unionAllQuery = [
        ' SELECT {0}, {1}, {2} FROM {3}'.format(
            ID_POINT, ID_POINT_Z, VEGETATION_FACTOR, dicOfTempoTables[t]
        )
        for t in dicOfTempoTables
    ]
    cursor.execute(
        """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS {1};
           {6};
           {7};
           DROP TABLE IF EXISTS {3};
           CREATE TABLE {3}
               AS SELECT {2}, {4}, MIN({5}) AS {5}
               FROM {0}
               GROUP BY {2}, {4}
           """.format(
            tempoAllVeg,
            ' UNION ALL '.join(unionAllQuery),
            ID_POINT,
            vegetationWeightFactorTable,
            ID_POINT_Z,
            VEGETATION_FACTOR,
            DataUtil.createIndex(
                tableName=tempoAllVeg, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoAllVeg, fieldName=ID_POINT_Z, isSpatial=False
            ),
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}, {1}
                      """.format(
                ','.join(dicOfTempoTables.values()),
                ','.join([zValueTable, tempoAllVeg]),
            )
        )

    return vegetationWeightFactorTable


def manageSuperimposition(
    cursor,
    dicAllWeightFactorsTables,
    facadeWithinCavity,
    upstreamPriorityTables=UPSTREAM_PRIORITY_TABLES,
    upstreamWeightingTables=UPSTREAM_WEIGHTING_TABLES,
    upstreamWeightingInterRules=UPSTREAM_WEIGHTING_INTER_RULES,
    upstreamWeightingIntraRules=UPSTREAM_WEIGHTING_INTRA_RULES,
    upstreamBackPriorityTables=UPSTREAM_BACKWARD_PRIORITY_TABLES,
    downstreamWeightingTable=DOWNSTREAM_WEIGTHING_TABLE,
    prefix=PREFIX_NAME,
):
    """Keep only one value per 3D point, dealing with superimposition from
    different Röckle zones. It is performed in three steps:
        - if a point is covered by several zones, keep the value only from
        a single zone based on the following priorities:
            1. the most upstream zone (if equal, use the next priority)
            2. the upper obstacle (if equal, use the next priority)
            3. a zone priority order (set in 'upstreamPriorityTables')
        - apply a weighting due to some upstream zones (such as wake zones)
        - add the backward zones weighted by cavity and wake zones
        - apply a weighting due to some downstream zones (such as vegetation)

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicAllWeightFactorsTables: Dictionary of Rockle zone tables
                Dictionary having as key the type of Rockle zone and as value
                the name of the table containing points corresponding to the zone
            facadeWithinCavity: string
                Name of the table containing the "facade" ID_POINT and height
                which will be useful for assigning a weighting to backward zones
            upstreamPriorityTables: pd.DataFrame, default UPSTREAM_PRIORITY_TABLES
                Defines which zones should be used in the priority algorithm and
                set priorities (column "priority") when the zone comes from a same
                upstream obstacle of same height. Also contains a column "ref_height" to
                set by which wind speed height the weigthing factor should be
                multiplied. The following values are possible:
                    -> 1: "upstream building height",
                    -> 2: "Reference wind speed measurement height Z_REF",
                    -> 3: "building height")
            upstreamBackPriorityTables: pd.DataFrame, default UPSTREAM_BACKWARD_PRIORITY_TABLES
                Defines which zones should be used in the priority algorithm and
                set priorities (column "priority") when the backward zone comes from a same
                upstream obstacle of same height. Also contains a column "ref_height" to
                set by which wind speed height the weigthing factor should be
                multiplied. The following values are possible:
                    -> 1: "upstream building height",
                    -> 2: "Reference wind speed measurement height Z_REF",
                    -> 3: "building height")
            upstreamWeightingTables: list, default UPSTREAM_WEIGHTING_TABLES
                Defines which upstream zones will be used to weight the wind speed factors
            upstreamWeightingInterRules: String, default UPSTREAM_WEIGHTING_INTER_RULES
                Defines how to deal with a point having several values from a
                same upstream weighting zone
                    -> "upstream":  use values from the most upstream and upper
                                    obstacles
            upstreamWeightingIntraRules: String, default UPSTREAM_WEIGHTING_INTRA_RULES
                Defines how to deal with a point having several values from
                several upstream weighting zones
                    -> "upstream":  use values from the most upstream and upper
                                    obstacles
            downstreamWeightingTable: String, default DOWNSTREAM_WEIGTHING_TABLES
                Name of the zone having the non-duplicated points used to weight
                the wind speed factors at the end
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            initializedWindFactorTable: String
                Name of the table containing the weighting factor for each 3D point
                (one value per point, means superimposition have been used)"""
    print('Deals with superimposition (keeps only 1 value per 3D point)')

    # Output base name
    outputBaseName = 'INITIALIZED_WIND_FACTOR_FIELD'

    # Name of the output table
    initializedWindFactorTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # name of the bacward weight field
    backWeightField = 'BACK_WEIGHT'

    # Backward zone names
    backwardZoneName = [CAVITY_BACKWARD_NAME, WAKE_BACKWARD_NAME]

    # Temporary tables (and prefix for temporary tables)
    tempoPrioritiesAll = DataUtil.postfix('TEMPO_PRIORITY_ALL')
    tempoPrioritiesWeighted = DataUtil.postfix('TEMPO_PRIORITY_WEIGHTED')
    tempoPrioritiesWeightedAll = DataUtil.postfix('TEMPO_PRIORITY_WEIGHTED_ALL')
    tempoBackwardWeights = DataUtil.postfix('TEMPO_BACWARD_WEIGHTS')
    dicBackwardWeighted = {
        t: DataUtil.postfix(DataUtil.prefix(t, prefix='TEMPO_WEIGHTED'))
        for t in backwardZoneName
    }
    tempoPrioritiesWeightedAllPlusBack = DataUtil.postfix(
        'TEMPO_PRIORITY_WEIGHTED_ALL_PLUS_BACK'
    )
    tempoUpstreamAndDownstream = DataUtil.postfix('TEMPO_UPSTREAM_AND_DOWNSTREAM')

    # Deal with superimposition when duplicated points in non backward zones
    tempoPrioritiesWeightedAll = manageUpstreamSuperimposition(
        cursor,
        dicAllWeightFactorsTables=dicAllWeightFactorsTables,
        upstreamPriorityTables=upstreamPriorityTables,
        upstreamWeightingTables=upstreamWeightingTables,
        upstreamWeightingInterRules=UPSTREAM_WEIGHTING_INTER_RULES,
        upstreamWeightingIntraRules=UPSTREAM_WEIGHTING_INTRA_RULES,
        backward=False,
        prefix=PREFIX_NAME,
    )

    # MANAGE THE BACKWARD ZONES
    # Get the weighting factor for the backward zones
    cursor.execute(
        """
          {0}{1}{2}{3}
          DROP TABLE IF EXISTS {4};
          CREATE TABLE {4}
              AS SELECT   a.{5}, a.{6}, ABS(b.{7}) AS {8}, b.{13}
              FROM     {9} AS a LEFT JOIN {10} AS b
                       ON a.{11} = b.{11} AND a.{12} = b.{12}
          """.format(
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAll,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAll,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=facadeWithinCavity, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=facadeWithinCavity, fieldName=ID_POINT_Z, isSpatial=False
            ),
            tempoBackwardWeights,
            ID_POINT_X,
            UPWIND_FACADE_FIELD,
            V,
            backWeightField,
            facadeWithinCavity,
            tempoPrioritiesWeightedAll,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
        )
    )

    # Apply the weighting factor to the backward zones
    backwardZoneQuery = {
        CAVITY_BACKWARD_NAME: """ a.{0}, a.{1}, b.{2}, NULL AS {3}, a.{5}*b.{4} AS {5},
                                 NULL AS {6}, {7} AS {8}, a.{9}
                             """.format(
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            U,
            backWeightField,
            V,
            W,
            UPSTREAM_PRIORITY_TABLES.loc[CAVITY_NAME, REF_HEIGHT_FIELD],
            REF_HEIGHT_FIELD,
            Y_WALL,
        ),
        WAKE_BACKWARD_NAME: """ a.{0}, a.{1}, b.{2}, NULL AS {3}, a.{5}*b.{4} AS {5},
                                 NULL AS {6}, {7} AS {8}, a.{9}
                             """.format(
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            U,
            backWeightField,
            V,
            W,
            UPSTREAM_PRIORITY_TABLES.loc[CAVITY_NAME, REF_HEIGHT_FIELD],
            REF_HEIGHT_FIELD,
            Y_WALL,
        ),
    }

    cursor.execute(
        ';'.join(
            [
                """
          {0}{1}{2}{3}
          DROP TABLE IF EXISTS {4};
          CREATE TABLE {4}
              AS SELECT {5}
              FROM     {6} AS a LEFT JOIN {7} AS b
                       ON a.{8} = b.{8} AND a.{9} = b.{9}
          """.format(
                    DataUtil.createIndex(
                        tableName=tempoBackwardWeights,
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=tempoBackwardWeights,
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicAllWeightFactorsTables[t],
                        fieldName=ID_POINT_X,
                        isSpatial=False,
                    ),
                    DataUtil.createIndex(
                        tableName=dicAllWeightFactorsTables[t],
                        fieldName=UPWIND_FACADE_FIELD,
                        isSpatial=False,
                    ),
                    dicBackwardWeighted[t],
                    backwardZoneQuery[t],
                    dicAllWeightFactorsTables[t],
                    tempoBackwardWeights,
                    ID_POINT_X,
                    UPWIND_FACADE_FIELD,
                )
                for t in backwardZoneQuery
            ]
        )
    )

    # Deal with superimposition when duplicated points between backward cavity and wake zones
    upstreamBackPrioritiesTempoTable = manageUpstreamSuperimposition(
        cursor,
        dicAllWeightFactorsTables=dicBackwardWeighted,
        upstreamPriorityTables=upstreamBackPriorityTables,
        upstreamWeightingTables=UPSTREAM_BACKWARD_WEIGHTING_TABLES,
        upstreamWeightingInterRules=UPSTREAM_WEIGHTING_INTER_RULES,
        upstreamWeightingIntraRules=UPSTREAM_WEIGHTING_INTRA_RULES,
        backward=True,
        prefix=PREFIX_NAME,
    )

    # Add backward zone points to the final table (replace points if exist)
    cursor.execute(
        """
          {0}{1}{2}{3}
          DROP TABLE IF EXISTS {4};
          CREATE TABLE {4}
              AS SELECT   a.* 
              FROM     {5} AS a LEFT JOIN {6} AS b
                       ON a.{7} = b.{7} AND a.{8} = b.{8}
              WHERE b.{7} IS NOT NULL AND b.{8} IS NOT NULL
              UNION ALL
              SELECT  a.*
              FROM     {6} AS a LEFT JOIN {5} AS b
                       ON a.{7} = b.{7} AND a.{8} = b.{8}
              WHERE b.{7} IS NULL AND b.{8} IS NULL
          """.format(
            DataUtil.createIndex(
                tableName=upstreamBackPrioritiesTempoTable,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamBackPrioritiesTempoTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAll,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAll,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            tempoPrioritiesWeightedAllPlusBack,
            upstreamBackPrioritiesTempoTable,
            tempoPrioritiesWeightedAll,
            ID_POINT,
            ID_POINT_Z,
        )
    )

    # MANAGE THE DOWNSTREAM WEIGHTING ZONES
    # Weight the wind speeds factors by the downstream weights (vegetation)
    cursor.execute(
        """
          {12};
          {13};
          {14};
          {15};
          DROP TABLE IF EXISTS {10};
          CREATE TABLE {10}
              AS SELECT   a.{2}, a.{3}, COALESCE(b.{4}, NULL) AS {4},
                          a.{5}*b.{6} AS {6},
                          COALESCE(a.{5}*b.{7}, a.{5}) AS {7},
                          a.{5}*b.{8} AS {8},
                          COALESCE(b.{9}, {11}) AS {9}
              FROM     {0} AS a LEFT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
          """.format(
            dicAllWeightFactorsTables[downstreamWeightingTable],
            tempoPrioritiesWeightedAllPlusBack,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            VEGETATION_FACTOR,
            U,
            V,
            W,
            REF_HEIGHT_FIELD,
            tempoUpstreamAndDownstream,
            REF_HEIGHT_DOWNSTREAM_WEIGHTING,
            DataUtil.createIndex(
                tableName=dicAllWeightFactorsTables[downstreamWeightingTable],
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=dicAllWeightFactorsTables[downstreamWeightingTable],
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAllPlusBack,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAllPlusBack,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
        )
    )

    # Join the downstream weigthted points to the non downstream weighted ones
    cursor.execute(
        """
          {12};
          {13};
          {10};
          {11};
          DROP TABLE IF EXISTS {9};
          CREATE TABLE {9}
              AS SELECT   a.{2}, a.{3}, a.{4}, a.{5}, a.{6}, a.{7}, a.{8}
              FROM     {0} AS a LEFT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
              WHERE    b.{2} IS NULL
              UNION ALL
              SELECT    c.{2}, c.{3}, c.{4}, c.{5}, c.{6}, c.{7}, c.{8}
              FROM     {1} AS c
          """.format(
            tempoPrioritiesWeightedAllPlusBack,
            tempoUpstreamAndDownstream,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            U,
            V,
            W,
            REF_HEIGHT_FIELD,
            initializedWindFactorTable,
            DataUtil.createIndex(
                tableName=tempoUpstreamAndDownstream,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoUpstreamAndDownstream,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAllPlusBack,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeightedAllPlusBack,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(
                ','.join(
                    [
                        tempoUpstreamAndDownstream,
                        tempoPrioritiesWeighted,
                        tempoPrioritiesWeightedAll,
                        tempoPrioritiesAll,
                        tempoBackwardWeights,
                        upstreamBackPrioritiesTempoTable,
                        ','.join(list(dicBackwardWeighted.values())),
                        tempoPrioritiesWeightedAllPlusBack,
                    ]
                )
            )
        )

    return initializedWindFactorTable


def manageUpstreamSuperimposition(
    cursor,
    dicAllWeightFactorsTables,
    upstreamPriorityTables=UPSTREAM_PRIORITY_TABLES,
    upstreamWeightingTables=UPSTREAM_WEIGHTING_TABLES,
    upstreamWeightingInterRules=UPSTREAM_WEIGHTING_INTER_RULES,
    upstreamWeightingIntraRules=UPSTREAM_WEIGHTING_INTRA_RULES,
    backward=False,
    prefix=PREFIX_NAME,
):
    """Keep only one value per 3D point, dealing with superimposition from
    different "all except downstream weighting tables". Can be applied for
    forward or backward wind zones.
    It is performed in three steps:
        - if a point is covered by several zones, keep the value only from
        a single zone based on the following priorities:
            1. the most upstream zone (if equal, use the next priority)
            2. the upper obstacle (if equal, use the next priority)
            3. a zone priority order (set in 'upstreamPriorityTables')
        - apply a weighting due to some upstream zones (such as wake zones)

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicAllWeightFactorsTables: Dictionary of Rockle zone tables
                Dictionary having as key the type of Rockle zone and as value
                the name of the table containing points corresponding to the zone
            upstreamPriorityTables: pd.DataFrame, default UPSTREAM_PRIORITY_TABLES
                Defines which zones should be used in the priority algorithm and
                set priorities (column "priority") when the zone comes from a same
                upstream obstacle of same height. Also contains a column "ref_height" to
                set by which wind speed height the weigthing factor should be
                multiplied. The following values are possible:
                    -> 1: "upstream building height",
                    -> 2: "Reference wind speed measurement height Z_REF",
                    -> 3: "building height")
            upstreamWeightingTables: list, default UPSTREAM_WEIGHTING_TABLES
                Defines which upstream zones will be used to weight the wind speed factors
            upstreamWeightingInterRules: String, default UPSTREAM_WEIGHTING_INTER_RULES
                Defines how to deal with a point having several values from a
                same upstream weighting zone
                    -> "upstream":  use values from the most upstream and upper
                                    obstacles
            upstreamWeightingIntraRules: String, default UPSTREAM_WEIGHTING_INTRA_RULES
                Defines how to deal with a point having several values from
                several upstream weighting zones
                    -> "upstream":  use values from the most upstream and upper
                                    obstacles
            backward: boolean, default False
                Whether or not the zones to deal with are backward zones
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            upstreamWindFactorTable: String
                Name of the table containing the weighting factor for each 3D point
                (one value per point, means superimposition have been used)"""
    print('Deals with superimposition (keeps only 1 value per 3D point)')

    # Output base name
    outputBaseName = 'UPSTREAM_WIND_FACTOR_TABLE'
    if backward:
        outputBaseName = 'BACKWARD_' + outputBaseName

    # Name of the output table
    upstreamWindFactorTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Temporary tables (and prefix for temporary tables)
    tempoPrioritiesAll = DataUtil.postfix('TEMPO_PRIORITY_ALL')
    tempoPrioritiesWeighted = DataUtil.postfix('TEMPO_PRIORITY_WEIGHTED')

    # Identify the points to keep for duplicates in upstream weigthing
    if backward:
        upstreamWeight = False
    else:
        upstreamWeight = True
    upstreamWeightingTempoTable = identifyUpstreamer(
        cursor=cursor,
        dicAllWeightFactorsTables=dicAllWeightFactorsTables,
        tablesToConsider=upstreamWeightingTables,
        prefix='TEMPO_WEIGHTING',
        upstream=upstreamWeight,
        weightingZone=True,
    )

    # Identify the points to keep to solve duplicated points in upstream priorities
    # (except zones being in weighting)
    if backward:
        upstreamPriorities = True
    else:
        upstreamPriorities = True
    upstreamPrioritiesTempoTable = identifyUpstreamer(
        cursor=cursor,
        dicAllWeightFactorsTables=dicAllWeightFactorsTables,
        tablesToConsider=upstreamPriorityTables.reindex(
            upstreamPriorityTables.index.difference(pd.Index(upstreamWeightingTables))
        ),
        prefix='TEMPO_PRIORITIES',
        upstream=upstreamPriorities,
    )

    # Identify the points to keep to solve duplicated points in upstream priorities
    # (only zones being in weighting - ie the most downstream wake zones...)
    if backward:
        upstreamPrioritiesWeight = True
    else:
        upstreamPrioritiesWeight = False
    upstreamPrioritiesWeightTempoTable = identifyUpstreamer(
        cursor=cursor,
        dicAllWeightFactorsTables=dicAllWeightFactorsTables,
        tablesToConsider=upstreamWeightingTables,
        prefix='TEMPO_PRIORITIES_WEIGHT',
        upstream=upstreamPrioritiesWeight,
    )

    # Join the points from the priority table to the downstreamest points of the
    # weighting table
    cursor.execute(
        """
          {12};
          {13};
          {14};
          {15};
          DROP TABLE IF EXISTS {10};
          CREATE TABLE {10}
              AS SELECT   {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {16}
              FROM        {1}
              UNION ALL   
              SELECT    a.{2}, a.{3}, a.{4}, a.{5}, a.{6}, a.{7},
                          NULL AS {8},
                          {11} AS {9},
                          {17} AS {16}
              FROM     {0} AS a LEFT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
              WHERE    b.{2} IS NULL AND b.{3} IS NULL
          """.format(
            upstreamPrioritiesWeightTempoTable,
            upstreamPrioritiesTempoTable,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            Y_WALL,
            U,
            V,
            W,
            REF_HEIGHT_FIELD,
            tempoPrioritiesAll,
            REF_HEIGHT_UPSTREAM_WEIGHTING,
            DataUtil.createIndex(
                tableName=upstreamPrioritiesWeightTempoTable,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamPrioritiesWeightTempoTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamPrioritiesTempoTable,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamPrioritiesTempoTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            IS_UPSTREAM_FIELD,
            IS_UPSTREAM_UPSTREAM_WEIGHTING,
        )
    )

    # Weight the wind speeds factors of the upstream priorities when the
    # weighting factors comes from more upstream and a higher position
    cursor.execute(
        """
          {12};
          {13};
          {14};
          {15};
          {16};
          {17};
          {18};
          {19};
          DROP TABLE IF EXISTS {10};
          CREATE TABLE {10}
              AS SELECT   a.{2}, a.{3}, a.{4}, COALESCE(a.{6}*b.{6}, a.{6}) AS {6},
                          COALESCE(a.{7}*b.{7}, a.{7}) AS {7},
                          COALESCE(a.{8}*b.{8}, 0) AS {8},
                          COALESCE(b.{9}, {11}) AS {9}
              FROM     {0} AS a RIGHT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
              WHERE    (a.{5} >= b.{5} AND a.{4} > b.{4}) OR (a.{5} > b.{5} AND b.{20} = 1)
              UNION ALL
              SELECT   a.{2}, a.{3}, a.{4}, a.{6}, a.{7}, NULL AS {8}, {11} AS {9}
              FROM     {0} AS a LEFT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
              WHERE    b.{2} IS NULL AND b.{3} IS NULL
          """.format(
            upstreamWeightingTempoTable,
            tempoPrioritiesAll,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            Y_WALL,
            U,
            V,
            W,
            REF_HEIGHT_FIELD,
            tempoPrioritiesWeighted,
            REF_HEIGHT_UPSTREAM_WEIGHTING,
            DataUtil.createIndex(
                tableName=upstreamWeightingTempoTable,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamWeightingTempoTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamWeightingTempoTable,
                fieldName=HEIGHT_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=upstreamWeightingTempoTable, fieldName=Y_WALL, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=ID_POINT_Z, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=Y_WALL, isSpatial=False
            ),
            IS_UPSTREAM_FIELD,
        )
    )

    # Join the upstream priority weigthted points to the upstream priority non-weighted ones
    cursor.execute(
        """
          {10};
          {11};
          {12};
          {13};
          DROP TABLE IF EXISTS {9};
          CREATE TABLE {9}
              AS SELECT   a.{2}, a.{3}, a.{4}, a.{5}, a.{6}, a.{7}, a.{8}
              FROM     {0} AS a LEFT JOIN {1} AS b
                       ON a.{2} = b.{2} AND a.{3} = b.{3}
              WHERE    b.{2} IS NULL
              UNION ALL
              SELECT    {2}, {3}, {4}, {5}, {6}, {7}, {8}
              FROM     {1}
          """.format(
            tempoPrioritiesAll,
            tempoPrioritiesWeighted,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            U,
            V,
            W,
            REF_HEIGHT_FIELD,
            upstreamWindFactorTable,
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesAll, fieldName=ID_POINT_Z, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeighted, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoPrioritiesWeighted, fieldName=ID_POINT_Z, isSpatial=False
            ),
        )
    )

    return upstreamWindFactorTable


def identifyUpstreamer(
    cursor,
    dicAllWeightFactorsTables,
    tablesToConsider,
    prefix=PREFIX_NAME,
    upstream=True,
    weightingZone=False,
):
    """If a point is covered by several zones, keep the value only from
    a single zone based on the following priorities:
        1. the most upstream zone (if equal, use the next priority)
        2. the upper obstacle (if equal, use the next priority)
        3. (optionnally) a zone priority order set in 'tablesToConsider'
    Note that the most downstream zone and lower obstacles (contrary of
                                                           conditions 1
                                                           and 2)
    may be conserved if upstream is set to False.

            Parameters
            _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    dicAllWeightFactorsTables: Dictionary of vegetation Rockle zone tables
        Dictionary having as key the type of vegetation Rockle zone and as value
        the name of the table containing points corresponding to the zone
    tablesToConsider: list (or pd.DataFrame if the 3rd step should be performed)
        Defines which zones should be used in the upstreamer identification.
        If priorities should be defined (in case the most upstream and the
        upper obstacle are not sufficient), then a dataframe containing
        the "priority" and "ref_height" columns should be passed. The
        column "ref_height" refers to the wind speed height by which
        the weigthing factor should be multiplied.
        The following values are possible:
            -> 1: "upstream building height",
            -> 2: "Reference wind speed measurement height Z_REF",
            -> 3: "building height"
    prefix: String, default PREFIX_NAME
        Prefix to add to the output table name
    upstream: Boolean, default True
        If False, the downstreamest zone coming from the lowest obstacles
        are used for calculation
    weightingZone: boolean, default False
        If True, use wind factor used for weighting (U_WEIGHT, V_WEIGHT, W_WEIGHT)
        instead of normal wind factors (U, V, W)

            Returns
            _ _ _ _ _ _ _ _ _ _

    uniqueValuePerPointTable: String
        Name of the table containing one value per point (without duplicate)"""
    print('Identify upstreamer points in {0} table'.format(prefix))

    # Wind factors
    if weightingZone:
        u_factor = U_WEIGHT
        v_factor = V_WEIGHT
        w_factor = W_WEIGHT
    else:
        u_factor = U
        v_factor = V
        w_factor = W
    wind_factor_names = {u_factor: U, v_factor: V, w_factor: W}

    # Output base name
    outputBaseName = 'UNIQUE_3D'

    # Name of the output table
    uniqueValuePerPointTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Temporary tables (and prefix for temporary tables)
    tempoAllPointsTable = DataUtil.postfix('TEMPO_3D_ALL', suffix=prefix)
    tempoUniquePointsTable = DataUtil.postfix('TEMPO_3D_UNIQUE', suffix=prefix)

    # If priorities should be used, recover list of tables and add columns to keep
    if type(tablesToConsider) == type(pd.DataFrame()):
        listOfTables = tablesToConsider.index
        defineCol2Add = '{0} INTEGER, {1} INTEGER, {2} INTEGER,'.format(
            REF_HEIGHT_FIELD, PRIORITY_FIELD, IS_UPSTREAM_FIELD
        )
    else:
        listOfTables = tablesToConsider
        defineCol2Add = ''

    # Set columns to keep in the final table
    selectQueryDownstream = {}
    considerPrioritiesQuery = ''

    for t in listOfTables:
        selectQueryDownstream[t] = """
                SELECT  NULL AS {0}, {1}, {2},
                        {3}, {4}, 
                """.format(ID_3D_POINT, ID_POINT, ID_POINT_Z, HEIGHT_FIELD, Y_WALL)

        # If priorities should be used, add columns to keep
        if type(tablesToConsider) == type(pd.DataFrame()):
            selectQueryDownstream[t] += """
                {0} AS {2}, 
                {1} AS {3},
                {4} AS {5},
                """.format(
                tablesToConsider.loc[t, REF_HEIGHT_FIELD],
                tablesToConsider.loc[t, PRIORITY_FIELD],
                REF_HEIGHT_FIELD,
                PRIORITY_FIELD,
                tablesToConsider.loc[t, IS_UPSTREAM_FIELD],
                IS_UPSTREAM_FIELD,
            )

            # Add the priority field as a decision criteria if two zones have
            # the same upstream obstacle
            considerPrioritiesQuery = ', b.{0} ASC'.format(PRIORITY_FIELD)

        #  Set to null wind speed factor for axis not set by upstream zones
        columns = DataUtil.getColumns(
            cursor=cursor, tableName=dicAllWeightFactorsTables[t]
        )
        for i in wind_factor_names:
            if i in columns:
                selectQueryDownstream[t] += ' {0} AS {1}, '.format(
                    i, wind_factor_names[i]
                )
            else:
                selectQueryDownstream[t] += ' NULL AS {0}, '.format(
                    wind_factor_names[i]
                )
        selectQueryDownstream[t] = (
            selectQueryDownstream[t][0:-2] + ' FROM ' + dicAllWeightFactorsTables[t]
        )

    # Gather all data for the upstream weighting into a same table
    cursor.execute(
        """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}({1} SERIAL, {2} INTEGER, {3} INTEGER,
                            {4} INTEGER, {5} INTEGER, {6} 
                            {7} DOUBLE, {8} DOUBLE, {9} DOUBLE)
               AS {10}
           """.format(
            tempoAllPointsTable,
            ID_3D_POINT,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            Y_WALL,
            defineCol2Add,
            U,
            V,
            W,
            ' UNION ALL '.join(selectQueryDownstream.values()),
        )
    )

    if upstream:
        order = 'DESC'
    else:
        order = 'ASC'
    # Identify which point should be conserved in the upstream weighting table
    cursor.execute(
        """
           {7};
           {8};
           {9};
           {10};
           DROP TABLE IF EXISTS {6};
           CREATE TABLE {6}
               AS SELECT   DISTINCT(a.{2}) AS {2},
                           (SELECT  b.{1}
                            FROM    {0} AS b
                            WHERE a.{2} = b.{2} AND a.{3} = b.{3}
                            ORDER BY (b.{5}, b.{4}) {11} {12} LIMIT 1) AS {1}
               FROM        {0} AS a;
           """.format(
            tempoAllPointsTable,
            ID_3D_POINT,
            ID_POINT,
            ID_POINT_Z,
            HEIGHT_FIELD,
            Y_WALL,
            tempoUniquePointsTable,
            DataUtil.createIndex(
                tableName=tempoAllPointsTable, fieldName=ID_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoAllPointsTable, fieldName=ID_POINT_Z, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoAllPointsTable, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoAllPointsTable, fieldName=Y_WALL, isSpatial=False
            ),
            order,
            considerPrioritiesQuery,
        )
    )

    # Recover the useful informations from the unique points kept
    cursor.execute(
        """
          {4};
          {5};
          DROP TABLE IF EXISTS {3};
          CREATE TABLE {3}
              AS SELECT a.*
              FROM     {0} AS a RIGHT JOIN {2} AS b
                       ON a.{1} = b.{1}
          """.format(
            tempoAllPointsTable,
            ID_3D_POINT,
            tempoUniquePointsTable,
            uniqueValuePerPointTable,
            DataUtil.createIndex(
                tableName=tempoAllPointsTable, fieldName=ID_3D_POINT, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoUniquePointsTable, fieldName=ID_3D_POINT, isSpatial=False
            ),
        )
    )

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(
                ','.join([tempoAllPointsTable, tempoUniquePointsTable])
            )
        )

    return uniqueValuePerPointTable


def getVerticalProfile(
    cursor,
    pointHeightList,
    z0,
    profileType=PROFILE_TYPE,
    v_ref=V_REF,
    z_ref=Z_REF,
    prefix=PREFIX_NAME,
    **kwargs,
):
    """Get the horizontal wind speed of a set of point heights. The
    wind speed profile used to set wind speed value can be:
        - power-law: proposed by Kuttler (2000) and initially
        used in QUIC-URB (Pardyjak et Brown, 2003),
        - urban: exponential below mean building height and log otherwise
        (used in Nelson et al., 2007)
        - user: the profile is defined by the user (need to pass the
                                                    'verticalProfileFile'
                                                    parameter)

    Note that:
        - the exponent p of the power-law is calculated according to the
    formulae p = 0.12*z0+0.18 (Matzarakis et al. 2009),
        - the attenuation coefficient for the exponential wind profile is calculated according to
    A = 9.6 * lambda_f (ratio of frontal and plot area - Hanna and Britter, 2002)
        - the stability is not taken into account yet

    References:
            Kuttler, Wilhelm. "Stadtklima." Umweltwissenschaften und
        Schadstoff-Forschung 16.3 (2004): 187-199.
            Matzarakis, A. and Endler, C., 2009: Physiologically Equivalent
        Temperature and Climate Change in Freiburg. Eighth Symposium on the
        Urban Environment. American Meteorological Society, Phoenix/Arizona,
        10. to 15. January 2009 4(2), 1–8.
            Nelson, M. A., B. Addepalli, D. Boswell, M. J. Brown, et Los
        Alamos National Laboratory. « QUIC Start Guide (V.45). », 2007.
        http://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-07-2799.
            Pardyjak, Eric R, et Michael Brown. « QUIC-URB v. 1.1: Theory and
        User’s Guide ». Los Alamos National Laboratory, Los Alamos, NM, 2003.



                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            pointHeightList: list
                Height (in meter) of the points for which we want the wind speed
            z0: float
                Value of the study area roughness height
            profileType: String, default PROFILE_TYPE
                Type of wind profile to use:
                    - "urban": exponential below building mean height and log otherwise
                    - "power": traditional power law profile
                    - "user": set by the user (in a text file)
            v_ref: float, default V_REF
                Wind speed (m/s) measured at measurement height z_ref
            z_ref: float, DEFAULT Z_REF
                Height of the wind speed sensor used to set the reference wind speed v_ref
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name
            (optional) d: float
                Value of the study area displacement length (only if profileType = "log" or "urban")
            (optional) H: float
                Value of the study area building geometric mean height (only if profileType = "urban")
            (optional) lambda_f: float
                Value of the study area frontal density (only if profileType = "urban")
            (optional) verticalProfileFile: string
                Path of the file where is stored the vertical wind profile
                (no header, comma separated, first column is the height, second column is the wind speed)

                Returns
                _ _ _ _ _ _ _ _ _ _

            verticalWindProfile: pd.DataFrame
                Values of the wind speed and height from ground for each vertical level
    """
    # Get kwargs arguments
    d = kwargs.get('d', None)
    H = kwargs.get('H', None)
    lambda_f = kwargs.get('lambda_f', None)
    verticalProfileFile = kwargs.get('verticalProfileFile', None)
    if profileType == 'power':
        verticalWindProfile = pd.Series(
            [v_ref * (z / z_ref) ** (0.12 * z0 + 0.18) for z in pointHeightList],
            index=pointHeightList,
        )

    elif profileType == 'urban':
        A = 9.6 * lambda_f
        pointHeightIndex = pd.Index(pointHeightList)
        pointHeighCanopy = pointHeightIndex[pointHeightIndex < H]
        pointHeighAbove = pointHeightIndex[pointHeightIndex >= H]
        speedAtCanopyHeight = v_ref * np.log((H - d) / z0) / np.log(z_ref / z0)

        verticalProfileWithin = pd.Series(
            [speedAtCanopyHeight * np.exp(A * (z / H - 1)) for z in pointHeighCanopy],
            index=pointHeighCanopy,
        )
        verticalProfileAbove = pd.Series(
            [
                v_ref * np.log((z - d) / z0) / np.log(z_ref / z0)
                for z in pointHeighAbove
            ],
            index=pointHeighAbove,
        )

        # verticalWindProfile = verticalProfileWithin \
        #     .append(verticalProfileAbove[verticalProfileAbove > verticalProfileWithin.max()]) \
        #     .reindex(pointHeightIndex).interpolate(method="index")

        verticalWindProfile = (
            pd.concat(
                [
                    verticalProfileWithin,
                    verticalProfileAbove[
                        verticalProfileAbove > verticalProfileWithin.max()
                    ],
                ],
                ignore_index=True,
            )
            .reindex(pointHeightIndex)
            .interpolate(method='index')
        )

    elif profileType == 'user':
        pointHeightIndex = pd.Index(pointHeightList)
        verticalWindProfile = pd.read_csv(
            verticalProfileFile, header=None, index_col=0, names=['z', 'v'], dtype=float
        )['v']
        verticalWindProfile = verticalWindProfile.reindex(
            pointHeightIndex.union(verticalWindProfile.index)
        )
        verticalWindProfile.loc[0] = 0
        verticalWindProfile = (
            verticalWindProfile.sort_values()
            .interpolate(method='polynomial', order=1)
            .reindex(pointHeightIndex)
        )

    # Add the height from ground as column instead of index
    verticalWindProfile.sort_index(inplace=True)
    verticalWindProfile = pd.DataFrame(
        {HORIZ_WIND_SPEED: verticalWindProfile.values, Z: verticalWindProfile.index},
        index=range(1, verticalWindProfile.size + 1),
    )

    return verticalWindProfile


def setInitialWindField(
    cursor,
    initializedWindFactorTable,
    gridPoint,
    df_gridBuil,
    z0,
    sketchHeight,
    profileType=PROFILE_TYPE,
    meshSize=MESH_SIZE,
    dz=DZ,
    z_ref=Z_REF,
    v_ref=V_REF,
    tempoDirectory=TEMPO_DIRECTORY,
    **kwargs,
):
    """Set the initial 3D wind speed according to the wind speed factor in
    the Röckle zones and to the initial vertical wind speed profile.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            initializedWindFactorTable: String
                Name of the table containing the weighting factor for each 3D point
                (one value per point, means superimposition have been used)
            gridPoint: String
                Name of the grid point table
            df_gridBuil: pd.DataFrame
                3D multiindex corresponding to grid points intersecting buildings
            z0: float
                Value of the study area roughness height
            sketchHeight: float
                Height of the sketch (m)
            profileType: String
                Type of wind profile to use:
                    - "urban": exponential below building mean height and log otherwise
                    - "log": traditional log profile
                    - "power": traditional power law profile
                    - "user": set by the user (in a text file)
            meshSize: float, default MESH_SIZE
                Resolution (in meter) of the grid
            dz: float, default DZ
                Resolution (in meter) of the grid in the vertical direction
            z_ref: float, DEFAULT Z_REF
                Height of the wind speed sensor used to set the reference wind speed v_ref
            v_ref: float, default V_REF
                Wind speed (m/s) measured at measurement height z_ref
            tempoDirectory: String, default TEMPO_DIRECTORY
                Path of the directory where will be stored the grid points
                having Röckle initial wind speed values (in order to exchange
                                                         data between H2 to Python)
            (optional) d: float
                Value of the study area displacement length (only if profileType = "log" or "urban")
            (optional) H: float
                Value of the study area building geometric mean height (only if profileType = "urban")
            (optional) lambda_f: float
                Value of the study area frontal density (only if profileType = "urban")
            (optional) verticalProfileFile: string
                Path of the file where is stored the vertical wind profile
                (no header, comma separated, first column is the height,
                 second column is the wind speed)



                Returns
                _ _ _ _ _ _ _ _ _ _

            initial3dWindSpeed: pd.DataFrame
                3D wind speed value used as "first guess" in the wind solver
            nPoints: dictionary
                Dimension of the 3D grid object with X, Y and Z as key and the
                number of grid point in the corresponding axis as value
            verticalWindSpeedProfile: pd.Series
                Initial wind speed profile along a vertical axis z"""

    print('Set the initial 3D wind speed field')
    # Get kwargs arguments
    d = kwargs.get('d', None)
    H = kwargs.get('H', None)
    lambda_f = kwargs.get('lambda_f', None)
    verticalProfileFile = kwargs.get('verticalProfileFile', None)

    # File name of the intermediate data saved on disk
    initRockleFilename = 'INIT_WIND_ROCKLE_ZONES.csv'

    # Temporary tables (and prefix for temporary tables)
    tempoVerticalProfileTable = DataUtil.postfix('TEMPO_VERTICAL_PROFILE_WIND')
    tempoBuildingHeightWindTable = DataUtil.postfix('TEMPO_BUILDING_HEIGHT_WIND')
    tempoZoneWindSpeedFactorTable = DataUtil.postfix('TEMPO_ZONE_WIND_SPEED_FACTOR')

    # Set a list of the level height and get their horizontal wind speed
    levelHeightList = [
        i
        for i in np.arange(
            float(dz) / 2, float(dz) / 2 + math.trunc(sketchHeight / dz) * dz, dz
        )
    ]
    verticalWindSpeedProfile = getVerticalProfile(
        cursor=cursor,
        pointHeightList=levelHeightList,
        z0=z0,
        v_ref=v_ref,
        z_ref=z_ref,
        profileType=profileType,
        d=d,
        H=H,
        lambda_f=lambda_f,
        verticalProfileFile=verticalProfileFile,
    )

    print('verticalWindSpeedProfile', verticalWindSpeedProfile)

    # Insert the initial vertical wind profile values into a table
    valuesForEachRowProfile = [
        str(i) + ',' + str(j)
        for i, j in verticalWindSpeedProfile[HORIZ_WIND_SPEED].items()
    ]
    print('valuesForEachRowProfile', valuesForEachRowProfile)

    cursor.execute(
        """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}({1} INTEGER, {2} DOUBLE);
           INSERT INTO {0} VALUES ({3});
           """.format(
            tempoVerticalProfileTable,
            ID_POINT_Z,
            V,
            '), ('.join(valuesForEachRowProfile),
        )
    )

    # Get the wind speed at each building height value...
    cursor.execute(
        """ SELECT DISTINCT({0}) AS {0}
                       FROM {1}
                       WHERE {0} IS NOT NULL;                   
                   """.format(HEIGHT_FIELD, initializedWindFactorTable)
    )
    buildingHeightList = cursor.fetchall()
    if len(buildingHeightList) > 0:
        df_buildingHeightList = pd.Series(pd.DataFrame(buildingHeightList)[0].values)
        buildingHeightWindSpeed = getVerticalProfile(
            cursor=cursor,
            pointHeightList=df_buildingHeightList,
            z0=z0,
            v_ref=v_ref,
            z_ref=z_ref,
            profileType=profileType,
            d=d,
            H=H,
            lambda_f=lambda_f,
            verticalProfileFile=verticalProfileFile,
        )

        # ... and insert it into a table
        valuesForEachRowBuilding = [
            str(i) + ',' + str(j)
            for i, j in buildingHeightWindSpeed.set_index(Z)[
                HORIZ_WIND_SPEED
            ].items()
        ]
        cursor.execute(
            """
               DROP TABLE IF EXISTS {0};
               CREATE TABLE {0}({1} INTEGER, {2} DOUBLE);
               INSERT INTO {0} VALUES ({3});
               """.format(
                tempoBuildingHeightWindTable,
                HEIGHT_FIELD,
                V,
                '), ('.join(valuesForEachRowBuilding),
            )
        )
    else:
        cursor.execute(
            """
               DROP TABLE IF EXISTS {0};
               CREATE TABLE {0}({1} INTEGER, {2} DOUBLE);
               """.format(tempoBuildingHeightWindTable, HEIGHT_FIELD, V)
        )

    if v_ref is None or z_ref is None:
        v_ref = verticalWindSpeedProfile.loc[
            verticalWindSpeedProfile.index[-1], HORIZ_WIND_SPEED
        ]
        z_ref = verticalWindSpeedProfile.loc[verticalWindSpeedProfile.index[-1], Z]

    # Calculates the initial wind speed field according to each point rule
    # and join to the table x and y coordinates
    cursor.execute(
        """
           {16};
           {17};
           {18};
           {19};
           {20};
           DROP TABLE IF EXISTS {4};
           CREATE TABLE {4}
               AS SELECT   a.{5},
                           a.{2},
                           CASE WHEN  a.{3}=1
                               THEN    (SELECT   c.{6} 
                                       FROM     {10} AS c
                                       WHERE    a.{11} = c.{11})
                           WHEN a.{3} = 2
                               THEN        {7}
                           WHEN a.{3} = 3
                               THEN      (SELECT   b.{6} 
                                         FROM     {1} AS b
                                         WHERE    a.{2} = b.{2})
                               ELSE      (SELECT   5 * b.{6} 
                                         FROM     {1} AS b
                                         WHERE    a.{2} = b.{2})
                           END AS WIND_SPEED,
                           a.{8},
                           a.{6},
                           a.{9}
               FROM {0} AS a;
           {21};
           {22};
           CALL CSVWRITE('{13}',
                         'SELECT b.{14} - 1 AS {14}_MINUS_1,
                                 b.{15} - 1 AS {15}_MINUS_1,
                                 a.{2},
                                 a.{8} * WIND_SPEED AS {8},
                                 a.{6} * WIND_SPEED AS {6},
                                 a.{9} * WIND_SPEED AS {9}
                          FROM {4} AS a LEFT JOIN {12} AS b
                          ON a.{5} = b.{5}',
                         'charset=UTF-8 fieldSeparator=,')
           """.format(
            initializedWindFactorTable,
            tempoVerticalProfileTable,
            ID_POINT_Z,
            REF_HEIGHT_FIELD,
            tempoZoneWindSpeedFactorTable,
            ID_POINT,
            V,
            v_ref,
            U,
            W,
            tempoBuildingHeightWindTable,
            HEIGHT_FIELD,
            gridPoint,
            os.path.join(tempoDirectory, initRockleFilename),
            ID_POINT_X,
            ID_POINT_Y,
            DataUtil.createIndex(
                tableName=initializedWindFactorTable,
                fieldName=HEIGHT_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=initializedWindFactorTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=initializedWindFactorTable,
                fieldName=REF_HEIGHT_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoVerticalProfileTable,
                fieldName=ID_POINT_Z,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoBuildingHeightWindTable,
                fieldName=HEIGHT_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoZoneWindSpeedFactorTable,
                fieldName=ID_POINT,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=gridPoint, fieldName=ID_POINT, isSpatial=False
            ),
        )
    )

    # Get the number of grid point for each axis x, y and z
    cursor.execute(
        """SELECT   MAX({0}) AS ID_POINT_X,
                               MAX({1}) AS ID_POINT_Y
                       FROM     {2}
                       """.format(ID_POINT_X, ID_POINT_Y, gridPoint)
    )
    nPointsResults = cursor.fetchall()
    nPoints = {
        X: nPointsResults[0][0],
        Y: nPointsResults[0][1],
        Z: verticalWindSpeedProfile.index.max() + 1,
    }

    # Initialize the 3D wind speed field considering no obstacles
    verticalWindSpeedProfile.loc[0] = [0, 0]
    verticalWindSpeedProfile.sort_index(inplace=True)
    df_wind0 = pd.DataFrame(
        {
            U: np.zeros(nPoints[X] * nPoints[Y] * nPoints[Z]),
            V: [
                val
                for j in range(nPoints[Y])
                for i in range(nPoints[X])
                for val in verticalWindSpeedProfile[HORIZ_WIND_SPEED]
            ],
            W: np.zeros(nPoints[X] * nPoints[Y] * nPoints[Z]),
        },
        index=pd.MultiIndex.from_product(
            [
                [i for i in range(0, nPoints[X])],
                [j for j in range(0, nPoints[Y])],
                [k for k in range(0, nPoints[Z])],
            ]
        ),
    )

    # Read the wind speed near obstacles (data coming from H2GIS database)
    df_wind0_rockle = pd.read_csv(
        os.path.join(tempoDirectory, initRockleFilename), header=0, index_col=[0, 1, 2]
    )

    # Update the 3D wind speed field with the initial guess near obstacles
    for c in df_wind0_rockle.columns:
        df_wind0.loc[df_wind0_rockle[c].sort_index().dropna().index, c] = (
            df_wind0_rockle[c].sort_index().dropna()
        )

        # Renormalize wind speed at each height to make sure there is no offset of
    # wind speed between the wind profile and the initialization before the balance of wind
    if REMOVE_INITIALIZATION_OFFSET:
        max_zi = df_wind0_rockle.index.get_level_values('ID_Z').max()
        for z_i in verticalWindSpeedProfile.index[1 : max_zi + 1]:
            df_wind0.loc[idx[:, :, z_i], :] = (
                df_wind0.loc[idx[:, :, z_i], :]
                * verticalWindSpeedProfile.loc[z_i, HORIZ_WIND_SPEED]
                / df_wind0.loc[idx[:, :, z_i], :].pow(2).sum(axis=1).pow(0.5).mean()
            )

    # Set to 0 wind speed within buildings...
    df_wind0.loc[df_gridBuil.index] = 0

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(
                ','.join(
                    [
                        tempoVerticalProfileTable,
                        tempoBuildingHeightWindTable,
                        tempoZoneWindSpeedFactorTable,
                    ]
                )
            )
        )

    return df_wind0, nPoints, verticalWindSpeedProfile


def identifyBuildPoints(
    cursor,
    gridPoint,
    stackedBlocksWithBaseHeight,
    meshSize=MESH_SIZE,
    dz=DZ,
    tempoDirectory=TEMPO_DIRECTORY,
):
    """Identify grid cells intersecting buildings.

        Parameters
        _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    gridPoint: String
        Name of the grid point table
    stackedBlocksWithBaseHeight: String
        Name of the table containing stacked blocks with block base
        height
    dz: float, default DZ
        Resolution (in meter) of the grid in the vertical direction
    tempoDirectory: String, default = TEMPO_DIRECTORY
        Path of the directory where will be stored the grid points
        intersecting with buildings (in order to exchange
                                     data between H2 to Python)


        Returns
        _ _ _ _ _ _ _ _ _ _

    df_gridBuil: pd.DataFrame
        3D multiindex corresponding to grid points intersecting buildings"""

    print('Identify grid points intersecting buildings')

    # File name of the intermediate data saved on disk
    buildPointsFilename = 'BUILDING_POINTS.csv'

    # Temporary tables (and prefix for temporary tables)
    tempoBuildPointsTable = DataUtil.postfix('BUILDING_POINTS')
    tempoLevelHeightPointTable = DataUtil.postfix('LEVEL_POINTS')

    # Identify 2D coordinates of points intersecting buildings
    cursor.execute(
        """
           {9};
           {10};
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT a.{1}, a.{8}, b.{2}, b.{3}, b.{4}
               FROM {5} AS a, {6} AS b
               WHERE a.{7} && b.{7} AND ST_INTERSECTS(a.{7}, b.{7})
           """.format(
            tempoBuildPointsTable,
            ID_POINT_X,
            ID_FIELD_STACKED_BLOCK,
            HEIGHT_FIELD,
            BASE_HEIGHT_FIELD,
            gridPoint,
            stackedBlocksWithBaseHeight,
            GEOM_FIELD,
            ID_POINT_Y,
            DataUtil.createIndex(
                tableName=gridPoint, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=stackedBlocksWithBaseHeight,
                fieldName=GEOM_FIELD,
                isSpatial=True,
            ),
        )
    )

    # Get the maximum building height
    cursor.execute(
        """
           SELECT MAX({0}) AS {0} FROM {1};
           """.format(HEIGHT_FIELD, stackedBlocksWithBaseHeight)
    )
    buildMaxHeight = cursor.fetchall()[0][0]

    # Set a list of the level height (and indice) which can intersect with buildings
    if buildMaxHeight:
        levelHeightList = [
            str(j + 1) + ',' + str(i)
            for j, i in enumerate(
                np.arange(
                    float(dz) / 2,
                    float(dz) / 2 + math.trunc(buildMaxHeight / dz) * dz,
                    dz,
                )
            )
        ]

        # ...and insert them into a table
        cursor.execute(
            """
               DROP TABLE IF EXISTS {0};
               CREATE TABLE {0}({1} INTEGER, {2} DOUBLE);
               INSERT INTO {0} VALUES ({3});
               """.format(
                tempoLevelHeightPointTable, ID_POINT_Z, Z, '), ('.join(levelHeightList)
            )
        )
    else:
        cursor.execute(
            """
               DROP TABLE IF EXISTS {0};
               CREATE TABLE {0}({1} INTEGER, {2} DOUBLE);
               """.format(tempoLevelHeightPointTable, ID_POINT_Z, Z)
        )

    # Identify the third dimension of points intersecting buildings and save it...
    cursor.execute(
        """
           {9};
           {10};
           {11};
           CALL CSVWRITE('{0}',
                         ' SELECT a.{1}-1, a.{8}-1, b.{2}
                           FROM {3} AS a, {4} AS b
                           WHERE b.{5} <= a.{6} AND b.{5} > a.{7}',
                         'charset=UTF-8 fieldSeparator=,')
           """.format(
            os.path.join(tempoDirectory, buildPointsFilename),
            ID_POINT_X,
            ID_POINT_Z,
            tempoBuildPointsTable,
            tempoLevelHeightPointTable,
            Z,
            HEIGHT_FIELD,
            BASE_HEIGHT_FIELD,
            ID_POINT_Y,
            DataUtil.createIndex(
                tableName=tempoBuildPointsTable, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoBuildPointsTable,
                fieldName=BASE_HEIGHT_FIELD,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoLevelHeightPointTable, fieldName=Z, isSpatial=False
            ),
        )
    )

    # ...in order to load it back into Python
    df_gridBuil = pd.read_csv(
        os.path.join(tempoDirectory, buildPointsFilename), header=0, index_col=[0, 1, 2]
    )

    # Remove potential duplicated indexes
    df_gridBuil = pd.DataFrame(index=df_gridBuil.index.drop_duplicates())

    if not DEBUG:
        # Remove intermediate tables
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0}
                      """.format(
                ','.join([tempoBuildPointsTable, tempoLevelHeightPointTable])
            )
        )

    return df_gridBuil
