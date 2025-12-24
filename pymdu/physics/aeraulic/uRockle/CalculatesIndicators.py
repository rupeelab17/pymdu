#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:34:12 2021

@author: Jérémy Bernard, University of Gothenburg
"""

try:
    from GlobalVariables import *
    import DataUtil as DataUtil

except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil


def obstacleProperties(cursor, obstaclesTable, prefix=PREFIX_NAME):
    """Calculates obstacle properties (effective width and length)
    for a wind coming from North (thus you first need to rotate your
                                  obstacles to make them facing north if you
                                  want to study a different wind direction).
    The calculation method is based on Figure 1 of Nelson et al. (2008). Note
    that it is however adapted to our specific geometries which are sometimes
    far from rectangles...

    References:
        Nelson, Matthew, Bhagirath Addepalli, Fawn Hornsby, Akshay Gowardhan,
        Eric Pardyjak, et Michael Brown. « 5.2 Improvements to a Fast-Response
        Urban Wind Model », 2008.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            obstaclesTable: String
                Name of the table containing the obstacle geometries to characterize
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            obstaclePropertiesTable: String
                Name of the table containing the properties of each obstacle"""
    print('Calculates obstacle properties')

    # Output base name
    outputBaseName = 'PROPERTIES'

    # Name of the output table
    obstaclePropertiesTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Calculates the effective width (Weff) and effective length (Leff)
    # of each obstacle, respectively  based on their maximum cross-wind and
    # along-wind extends of the obstacle, weighted by the area ratio between
    # obstacle area and envelope area
    query = """
       DROP TABLE IF EXISTS {0};
       CREATE TABLE {0}
           AS SELECT   {1},
                       {2},
                       {3},
                       {4},
                       (ST_XMAX(ST_ENVELOPE({3}))-ST_XMIN({3}))*ST_AREA({3})/ST_AREA(ST_ENVELOPE({3})) AS {6},
                       (ST_YMAX(ST_ENVELOPE({3}))-ST_YMIN({3}))*ST_AREA({3})/ST_AREA(ST_ENVELOPE({3})) AS {7}
           FROM {5}""".format(
        obstaclePropertiesTable,
        ID_FIELD_STACKED_BLOCK,
        ID_FIELD_BLOCK,
        GEOM_FIELD,
        HEIGHT_FIELD,
        obstaclesTable,
        EFFECTIVE_WIDTH_FIELD,
        EFFECTIVE_LENGTH_FIELD,
    )
    cursor.execute(query)

    return obstaclePropertiesTable


def zoneProperties(cursor, obstaclePropertiesTable, prefix=PREFIX_NAME):
    """Calculates properties of the "Röckle" (some are not) zones:
        - for displacement: length Lf and vortex length Lfv (Bagal et al. - 2004),
        - for cavity: length Lr (equation 3 in Kaplan et al. - 1996),
        - for wake: length Lw (3*Lr, Kaplan et al. - 1996)
        - for cavity and wake: x coordinate of the downstreamest stacked block point,
        downstream left facade angle and downstream right facade angle
        - for rooftop perpendicular: height Hcm and length Lc (Pol et al. 2006)
        - for rooftop corner: wind speed factor C1 (Bagal et al. 2004 "Implementation of rooftop...)
    Note that L and W in the equation are respectively replaced by the
    effective length and width of each obstacle.

    References:
            Bagal, N, ER Pardyjak, et MJ Brown. « Improved upwind cavity
       parameterization for a fast response urban wind model ». In 84th Annual
       AMS Meeting. Seattle, WA, 2004.
            Kaplan, H., et N. Dinar. « A Lagrangian Dispersion Model for Calculating
       Concentration Distribution within a Built-up Domain ». Atmospheric
       Environment 30, nᵒ 24 (1 décembre 1996): 4197‑4207.
       https://doi.org/10.1016/1352-2310(96)00144-6.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            obstaclePropertiesTable: String
                Name  of the table containing the obstacle properties Weff, Leff
                and height
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            zonePropertiesTable: String
                Name of the table containing the properties of each obstacle zones"""
    print('Calculates zone properties')

    # Output base name
    outputBaseName = 'ZONE_LENGTH'

    # Name of the output table
    zoneLengthTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the process)
    tempoStackedLengthTab = DataUtil.postfix('TEMPO_STACKED_LENGTH_TAB')
    pointsStackedBlocks = DataUtil.postfix('POINTS_STACKED_BLOCKS')
    stackedBlocksXExt = DataUtil.postfix('STACKED_BLOCKS_X_EXT')
    stackedBlockAzimuths = DataUtil.postfix('STACKED_BLOCKS_AZIMUTHS')

    # Calculates the length (and sometimes height) of each zone:
    #   - for displacement: Lf and Lfv (Bagal et al. - 2004),
    #   - for cavity: Lr (equation 3 in Kaplan et al. - 1996),
    #   - for wake: Lw (3*Lr, Kaplan et al. - 1996)
    #   - rooftop perpendicular: Hcm and Lc (Pol et al. 2006)
    #   - rooftop corner: C1 (Bagal et al. 2004 "Implementation of rooftop...)
    query = """
       DROP TABLE IF EXISTS {0};
       CREATE TABLE {0}
           AS SELECT   {1},
                       {2},
                       {3},
                       1.*1.5*{9}/(1+0.8*{9}/{3}) AS {4},
                       1.*0.6*{9}/(1+0.8*{9}/{3}) AS {10},
                       1.*1.8*{9}/(POWER({8}/{3},0.3)*(1+0.24*{9}/{3})) AS {5},
                       1.*3*1.8*{9}/(POWER({8}/{3},0.3)*(1+0.24*{9}/{3})) AS {6},
                       0.22*(0.67*LEAST({3},{9})+0.33*GREATEST({3},{9})) AS {11},
                       0.9*(0.67*LEAST({3},{9})+0.33*GREATEST({3},{9})) AS {12},
                       1+0.05*{9}/{3} AS {13},
                       {14}
           FROM {7}""".format(
        tempoStackedLengthTab,
        ID_FIELD_STACKED_BLOCK,
        GEOM_FIELD,
        HEIGHT_FIELD,
        DISPLACEMENT_LENGTH_FIELD,
        CAVITY_LENGTH_FIELD,
        WAKE_LENGTH_FIELD,
        obstaclePropertiesTable,
        EFFECTIVE_LENGTH_FIELD,
        EFFECTIVE_WIDTH_FIELD,
        DISPLACEMENT_LENGTH_VORTEX_FIELD,
        ROOFTOP_PERP_HEIGHT,
        ROOFTOP_PERP_LENGTH,
        ROOFTOP_WIND_FACTOR,
        ID_FIELD_BLOCK,
    )
    cursor.execute(query)

    # Calculates the table containing the points corresponding to all polygons,
    # the polygon id and the extremum of the polygon we are looking for
    cursor.execute(
        """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}
               AS SELECT {1}, CAST(ST_X({2}) AS INTEGER) AS X, CAST(ST_Y({2})  AS INTEGER) AS Y, X_MAX, X_MIN, Y_MIN
               FROM ST_EXPLODE('(SELECT CAST(ST_XMAX({2}) AS INTEGER) AS X_MAX,
                                        CAST(ST_XMIN({2}) AS INTEGER) AS X_MIN,
                                        CAST(ST_YMIN({2}) AS INTEGER) AS Y_MIN,
                                        {1},
                                        ST_TOMULTIPOINT({2}) AS {2}
                                FROM {3})');
           """.format(
            pointsStackedBlocks,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            tempoStackedLengthTab,
        )
    )

    # Calculates points table corresponding to xmin, xmax and ymin
    xMinPointTable = DataUtil.getExtremumPoint(
        pointsTable=pointsStackedBlocks,
        axis='X',
        extremum='MIN',
        secondAxisExtremum='MIN',
        cursor=cursor,
        prefix_name=prefix,
    )
    xMaxPointTable = DataUtil.getExtremumPoint(
        pointsTable=pointsStackedBlocks,
        axis='X',
        extremum='MAX',
        secondAxisExtremum='MIN',
        cursor=cursor,
        prefix_name=prefix,
    )
    yMinPointTable = DataUtil.getExtremumPoint(
        pointsTable=pointsStackedBlocks,
        axis='Y',
        extremum='MIN',
        secondAxisExtremum='AVG',
        cursor=cursor,
        prefix_name=prefix,
    )

    # Join x extremum into a single table
    cursor.execute(
        """
           {0}{1}
           DROP TABLE IF EXISTS {2};
           CREATE TABLE {2}
               AS SELECT a.{3}, a.{4} AS {4}_MIN, b.{4} AS {4}_MAX
               FROM {5} AS a LEFT JOIN {6} AS b
               ON a.{3} = b.{3};
           """.format(
            DataUtil.createIndex(
                tableName=xMinPointTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=xMaxPointTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            stackedBlocksXExt,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            xMinPointTable,
            xMaxPointTable,
        )
    )

    # Calculates main azimuth of left-side and right-side downstream facades
    cursor.execute(
        """
           {0}{1}
           DROP TABLE IF EXISTS {2};
           CREATE TABLE {2}
               AS SELECT   a.{3}, ST_X(a.{4}) AS {5}, 
                           ST_AZIMUTH(b.{4}_MIN, a.{4})-PI()/2 AS THETA_LEFT,
                           ST_AZIMUTH(a.{4}, b.{4}_MAX)-PI()/2 AS THETA_RIGHT
               FROM {6} AS a LEFT JOIN {7} AS b
               ON a.{3} = b.{3};
           """.format(
            DataUtil.createIndex(
                tableName=stackedBlocksXExt,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=yMinPointTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            stackedBlockAzimuths,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            STACKED_BLOCK_UPSTREAMEST_X,
            yMinPointTable,
            stackedBlocksXExt,
        )
    )

    # Join calculated indicators to previous ones
    cursor.execute(
        """
           {0}{1}
           DROP TABLE IF EXISTS {2};
           CREATE TABLE {2}
               AS SELECT  a.*, b.{3}, COS(b.THETA_LEFT) AS {4}, SIN(b.THETA_LEFT) AS {5},
                          COS(b.THETA_RIGHT) AS {6}, SIN(b.THETA_RIGHT) AS {7}
               FROM {8} AS a LEFT JOIN {9} AS b
               ON a.{10} = b.{10};
           """.format(
            DataUtil.createIndex(
                tableName=xMinPointTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=xMaxPointTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            zoneLengthTable,
            STACKED_BLOCK_UPSTREAMEST_X,
            COS_BLOCK_LEFT_AZIMUTH,
            SIN_BLOCK_LEFT_AZIMUTH,
            COS_BLOCK_RIGHT_AZIMUTH,
            SIN_BLOCK_RIGHT_AZIMUTH,
            tempoStackedLengthTab,
            stackedBlockAzimuths,
            ID_FIELD_STACKED_BLOCK,
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            """
           DROP TABLE IF EXISTS {0}
           """.format(
                ','.join(
                    [
                        tempoStackedLengthTab,
                        pointsStackedBlocks,
                        stackedBlocksXExt,
                        stackedBlockAzimuths,
                    ]
                )
            )
        )

    return zoneLengthTable


def studyAreaProperties(cursor, upwindTable, stackedBlockTable, vegetationTable):
    """Calculates roughness height (z0) and displacement length (d) of the study area
    for a wind coming from North (thus you first need to rotate your
                                  obstacles to make them facing north if you
                                  want to study a different wind direction).
    The calculation method is based on Equations 17a to 18c from Hanna and
    Britter (2002). For building, each facade facing the wind is considered
    while the calculation is simplified for vegetation : the frontal area is
    simply calculated as the cross wind width of each vegetation patch
    multiplied by its crown vegetation height.

    WARNING: Hanna and Britter (2002) say that: "It is suggested that an upper limit to H,
    should be 20 m and an upper limit to z, is therefore about 3 m. Conse-
    quently these methods should not be used for skyscrapers in a large city
    center or for the Rocky Mountains"

    References:
        Hanna, SR, et RE Britter. « Wind flow and vapor cloud dispersion at
        industrial sites. Am. Inst ». Chem Eng, New York, 2002.


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            upwindTable: String
                Name of the table containing the obstacle upwind facades
                (WARNING : WITH BASE HEIGHT NOT UPDATED)
            stackedBlockTable: String
                Name of the table containing the stacked blocks
            vegetationTable: String
                Name of the table containing the vegetation patches

                Returns
                _ _ _ _ _ _ _ _ _ _

            z0: float
                Value of the study area roughness height
            d: float
                Value of the study area displacement length
            Hr: float
                Value of the study area geometric mean height (weighted by area)
            lambda_f: float
                Value of the study area frontal density"""
    print('Calculates study area properties')

    # Calculate the area of the study area
    cursor.execute(
        """
           SELECT ST_AREA(ST_BUFFER(ST_EXTENT({0}), 15))
           FROM   (SELECT    {0}
                  FROM {1}
                  UNION ALL
                  SELECT    {0}
                  FROM {2}) AS STUDY_AREA_AREA_TAB
           """.format(GEOM_FIELD, stackedBlockTable, vegetationTable)
    )
    area = cursor.fetchall()[0][0]

    # Calculates the obstacle (stacked blocks and vegetation)
    # geometric mean height weighted by the area (H_r)
    cursor.execute(
        """
           {0};
           """.format(
            DataUtil.createIndex(
                tableName=stackedBlockTable, fieldName=ID_FIELD_BLOCK, isSpatial=False
            )
        )
    )
    cursor.execute(
        """
           SELECT   EXP(1.0 / SUM(OBSTACLE_HEIGHT_TAB.AREA) * 
                        SUM(OBSTACLE_HEIGHT_TAB.AREA * LOG(OBSTACLE_HEIGHT_TAB.HEIGHT))) AS H_r,
                    MAX(OBSTACLE_HEIGHT_TAB.HEIGHT) AS H_max
            FROM (SELECT    MAX({0}) AS HEIGHT,
                            ST_AREA(ST_UNION(ST_ACCUM({5}))) AS AREA
                  FROM {1}
                  GROUP BY {4}
                  UNION ALL
                  SELECT    {2} AS HEIGHT,
                            ST_AREA({5}) AS AREA
                  FROM {3}) AS OBSTACLE_HEIGHT_TAB;
            """.format(
            HEIGHT_FIELD,
            stackedBlockTable,
            VEGETATION_CROWN_TOP_HEIGHT,
            vegetationTable,
            ID_FIELD_BLOCK,
            GEOM_FIELD,
        )
    )
    H_r, H_max = cursor.fetchall()[0]

    # Calculates the obstacle (stacked blocks and vegetation)
    # and frontal density (lambda_f)
    cursor.execute(
        """
            SELECT  SUM(FRONTAL_AREA_TAB.CROSS_WIND_LENGTH*
                        FRONTAL_AREA_TAB.CROSS_WIND_HEIGHT)/{7}
                     AS LAMBDA_f
            FROM    (SELECT    ST_XMAX({3})- ST_XMIN({3}) AS CROSS_WIND_LENGTH,
                            {0}-{4} AS CROSS_WIND_HEIGHT
                      FROM {5}
                      UNION ALL
                      SELECT    ST_XMAX({3})- ST_XMIN({3}) AS CROSS_WIND_LENGTH,
                                {1}-{6} AS CROSS_WIND_HEIGHT
                      FROM {2}) AS FRONTAL_AREA_TAB
         """.format(
            HEIGHT_FIELD,
            VEGETATION_CROWN_TOP_HEIGHT,
            vegetationTable,
            GEOM_FIELD,
            BASE_HEIGHT_FIELD,
            upwindTable,
            VEGETATION_CROWN_BASE_HEIGHT,
            area,
        )
    )
    lambda_f = cursor.fetchall()[0][0]

    # Calculates z0 and d according to Hanna and Britter (2002) Equations 16-17
    z0 = 0
    d = 0
    if lambda_f <= 0.15:
        z0 = lambda_f * H_r
        if lambda_f <= 0.05:
            d = 3 * lambda_f * H_r
        else:
            d = (0.15 + 5.5 * (lambda_f - 0.05)) * H_r
    elif lambda_f > 0.15:
        if lambda_f > 1:
            lambda_f = 1
        z0 = 0.15 * H_r
        d = (0.7 + 0.35 * (lambda_f - 0.15)) * H_r

    return z0, d, H_r, H_max, lambda_f


def maxObstacleHeight(cursor, stackedBlockTable, vegetationTable):
    """Calculates the maximum height of the obstacles within the study area.

        Parameters
        _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    stackedBlockTable: String
        Name of the table containing the stacked blocks
    vegetationTable: String
        Name of the table containing the vegetation patches

        Returns
        _ _ _ _ _ _ _ _ _ _

    Hmax: float
        Value of the maximum obstacle height within the study area"""
    print('Calculates maximum obstacle height within the study area')

    # Calculates the obstacle (stacked blocks and vegetation)
    # maximum height (Hmax)
    cursor.execute(
        """
           SELECT   MAX(HEIGHT) AS Hmax,
            FROM (SELECT MAX({0}) AS HEIGHT
                  FROM {1}
                  UNION ALL
                  SELECT MAX({2}) AS HEIGHT
                  FROM {3}) AS OBSTACLE_HEIGHT_TAB;""".format(
            HEIGHT_FIELD,
            stackedBlockTable,
            VEGETATION_CROWN_TOP_HEIGHT,
            vegetationTable,
        )
    )
    H_max = cursor.fetchall()[0][0]

    return H_max
