#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:29:14 2021

@author: Jérémy Bernard, University of Gothenburg
"""

try:
    from GlobalVariables import *
    import DataUtil

except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil


def loadData(
    fromCad,
    prefix,
    idFieldBuild,
    buildingHeightField,
    vegetationBaseHeight,
    vegetationTopHeight,
    idVegetation,
    vegetationAttenuationFactor,
    cursor,
    buildingFilePath,
    vegetationFilePath,
    srid,
):
    """Load the input files into the database (could be converted if from CAD)

        Parameters
        _ _ _ _ _ _ _ _ _ _

    fromCad: boolean
        Whether or not the data has to be converted from a CAD file
    prefix: String
        Name of the case to run. Also the name of the subdirectory containing
        the geometry files.
    idFieldBuild: String
        Name of the ID field from the input building data
    buildingHeightField: String
        Name of the height field from the input building data
    vegetationBaseHeight: String
        Name of the base height field from the input vegetation data
    vegetationTopHeight: String
        Name of the top height field from the input vegetation data
    idVegetation: String
        Name of the ID field from the input vegetation data
    vegetationAttenuationFactor: String
        Name of the attenuatiojn factor field from the input vegetation data
    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    buildingFilePath: String
        The path of the file where are saved buildings data
    vegetationFilePath: String
        The path of the file where are saved vegetation data
    srid: int
        The SRID of the data

        Returns
        _ _ _ _ _ _ _ _ _ _

    None"""
    print('Load input data')

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    buildTablePreSrid = DataUtil.postfix('build_pre_srid')
    vegTablePreSrid = DataUtil.postfix('veg_pre_srid')

    inputDataRel = {}
    inputDataAbs = {}
    # Check if the input comes from CAD file
    if fromCad:
        # IMPORT TRIANGLES AND CONVERT TO BUILDINGS AND VEGETATION GEOMETRIES
        inputDataRel['cadTriangles'] = os.path.join(
            inputDirectory, prefix, inputGeometries['cadTriangles']
        )
        inputDataAbs['cadTriangles'] = os.path.abspath(inputDataRel['cadTriangles'])

        # Load CAD triangles into H2GIS DB
        loadFile(
            cursor=cursor,
            filePath=inputDataAbs['cadTriangles'],
            tableName=CAD_TRIANGLE_NAME,
        )

        if inputGeometries['cadTreesIntersection']:
            inputDataRel['cadTreesIntersection'] = os.path.join(
                inputDirectory, prefix, inputGeometries['cadTreesIntersection']
            )
            inputDataAbs['cadTreesIntersection'] = os.path.abspath(
                inputDataRel['cadTreesIntersection']
            )

            # Load vegetation intersection into H2GIS DB
            loadFile(
                cursor=cursor,
                filePath=inputDataAbs['cadTreesIntersection'],
                tableName=CAD_VEG_INTERSECTION,
            )
            treesZone = CAD_VEG_INTERSECTION
        else:
            cursor.execute(
                """
                DROP TABLE IF EXISTS {0};
                CREATE TABLE {0}(PK INTEGER, {1} GEOMETRY,
                                 {2} DOUBLE, {3} DOUBLE,
                                 {4} INTEGER, {5} DOUBLE)
                """.format(
                    vegTablePreSrid,
                    GEOM_FIELD,
                    VEGETATION_CROWN_BASE_HEIGHT,
                    VEGETATION_CROWN_TOP_HEIGHT,
                    ID_VEGETATION,
                    VEGETATION_ATTENUATION_FACTOR,
                )
            )
            treesZone = None

        # Convert 3D triangles to 2.5 d buildings and tree patches
        fromShp3dTo2_5(
            cursor=cursor,
            triangles3d=CAD_TRIANGLE_NAME,
            TreesZone=treesZone,
            buildTableName=buildTablePreSrid,
            vegTableName=vegTablePreSrid,
            prefix=PREFIX_NAME,
        )

        # Save the building and vegetation layers ready to be used in URock
        DataUtil.saveTable(
            cursor=cursor,
            tableName=buildTablePreSrid,
            filedir=os.path.join(buildingFilePath),
            delete=True,
        )
        DataUtil.saveTable(
            cursor=cursor,
            tableName=vegTablePreSrid,
            filedir=os.path.join(vegetationFilePath),
            delete=True,
        )

    else:
        importQuery = ''
        h2gisBuildSrid = 0
        h2gisVegSrid = 0
        # 1. IMPORT BUILDING GEOMETRIES
        if buildingFilePath:
            # Load buildings into H2GIS DB
            loadFile(
                cursor=cursor,
                filePath=os.path.abspath(buildingFilePath),
                tableName=buildTablePreSrid,
            )

            # Get the building SRID
            cursor.execute(
                """
                           SELECT ST_SRID({0}) AS SRID FROM {1} LIMIT 1
                           """.format(GEOM_FIELD, buildTablePreSrid)
            )
            h2gisBuildSrid = cursor.fetchall()[0][0]

            # Create an ID FIELD if None.
            if idFieldBuild is None or idFieldBuild == '':
                cursor.execute(
                    """ 
                   ALTER TABLE {0} DROP COLUMN IF EXISTS {1};
                   ALTER TABLE {0} ADD COLUMN {1} SERIAL;
                   """.format(buildTablePreSrid, ID_FIELD_BUILD)
                )
                idFieldBuild = ID_FIELD_BUILD

            # Rename building fields to generic names
            if idFieldBuild.upper() != ID_FIELD_BUILD.upper():
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(buildTablePreSrid, idFieldBuild, ID_FIELD_BUILD)
            if buildingHeightField.upper() != HEIGHT_FIELD.upper():
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(buildTablePreSrid, buildingHeightField, HEIGHT_FIELD)

        else:
            importQuery += """ DROP TABLE IF EXISTS {0};
                               CREATE TABLE {0}({1} INTEGER, {2} GEOMETRY,
                                                {3} INTEGER);
                            """.format(
                buildTablePreSrid, ID_FIELD_BUILD, GEOM_FIELD, HEIGHT_FIELD
            )

            # 2. IMPORT VEGETATION GEOMETRIES
        if vegetationFilePath:
            # Load vegetation into H2GIS DB
            loadFile(
                cursor=cursor,
                filePath=os.path.abspath(vegetationFilePath),
                tableName=vegTablePreSrid,
            )

            # Get the vegetation SRID
            cursor.execute(
                """
                           SELECT ST_SRID({0}) AS SRID FROM {1} LIMIT 1
                           """.format(GEOM_FIELD, vegTablePreSrid)
            )
            h2gisVegSrid = cursor.fetchall()[0][0]

            # Create an ID FIELD if None.
            if idVegetation is None or idVegetation == '':
                cursor.execute(
                    """ 
                   ALTER TABLE {0} DROP COLUMN IF EXISTS {1};
                   ALTER TABLE {0} ADD COLUMN {1} SERIAL;
                   """.format(vegTablePreSrid, ID_VEGETATION)
                )
                idVegetation = ID_VEGETATION
            # Create an attenuation attribute with default 'DEFAULT_VEG_ATTEN_FACT'
            # if no column
            if vegetationAttenuationFactor is None or vegetationAttenuationFactor == '':
                cursor.execute(
                    """ 
                   ALTER TABLE {0} DROP COLUMN IF EXISTS {1};
                   ALTER TABLE {0} ADD COLUMN {1} DOUBLE DEFAULT {2};
                   """.format(
                        vegTablePreSrid,
                        VEGETATION_ATTENUATION_FACTOR,
                        DEFAULT_VEG_ATTEN_FACT,
                    )
                )
                vegetationAttenuationFactor = VEGETATION_ATTENUATION_FACTOR
            # Create a base height attribute with default 'DEFAULT_VEG_CROWN_BASE_HEIGHT_FRAC'
            # of the maximum height if no attribute for base height
            if vegetationBaseHeight is None or vegetationBaseHeight == '':
                cursor.execute(
                    """ 
                   ALTER TABLE {0} DROP COLUMN IF EXISTS {1};
                   ALTER TABLE {0} ADD COLUMN {1} DOUBLE;
                   UPDATE {0} SET {1} = {2} * {3};
                   """.format(
                        vegTablePreSrid,
                        VEGETATION_CROWN_BASE_HEIGHT,
                        DEFAULT_VEG_CROWN_BASE_HEIGHT_FRAC,
                        vegetationTopHeight,
                    )
                )
                vegetationBaseHeight = VEGETATION_CROWN_BASE_HEIGHT

            # Load vegetation data and rename fields to generic names
            if vegetationBaseHeight.upper() != VEGETATION_CROWN_BASE_HEIGHT.upper():
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(
                    vegTablePreSrid, vegetationBaseHeight, VEGETATION_CROWN_BASE_HEIGHT
                )
            if vegetationTopHeight.upper() != VEGETATION_CROWN_TOP_HEIGHT.upper():
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(
                    vegTablePreSrid, vegetationTopHeight, VEGETATION_CROWN_TOP_HEIGHT
                )
            if idVegetation.upper() != ID_VEGETATION.upper():
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(vegTablePreSrid, idVegetation, ID_VEGETATION)
            if (
                vegetationAttenuationFactor.upper()
                != VEGETATION_ATTENUATION_FACTOR.upper()
            ):
                importQuery += """
                    ALTER TABLE {0} DROP COLUMN IF EXISTS {2};
                    ALTER TABLE {0} RENAME COLUMN {1} TO {2};
                    """.format(
                    vegTablePreSrid,
                    vegetationAttenuationFactor,
                    VEGETATION_ATTENUATION_FACTOR,
                )
        else:
            importQuery += """ DROP TABLE IF EXISTS {0};
                               CREATE TABLE {0}(PK INTEGER, {1} GEOMETRY,
                                                {2} DOUBLE, {3} DOUBLE,
                                                {4} INTEGER, {5} DOUBLE);
                            """.format(
                vegTablePreSrid,
                GEOM_FIELD,
                VEGETATION_CROWN_BASE_HEIGHT,
                VEGETATION_CROWN_TOP_HEIGHT,
                ID_VEGETATION,
                VEGETATION_ATTENUATION_FACTOR,
            )
        cursor.execute(importQuery)

    # 3. SET VEGETATION AND BUILDING TABLE SRID AND REMOVE SMALL OBSTACLE
    # If H2GIS does not identify any SRID for the tables, set the ones identied by GDAL
    if h2gisBuildSrid == 0 and h2gisVegSrid == 0:
        buildSrid = srid
        vegSrid = srid
    elif h2gisBuildSrid != 0 and h2gisVegSrid == 0:
        buildSrid = h2gisBuildSrid
        vegSrid = h2gisBuildSrid
    elif h2gisBuildSrid == 0 and h2gisVegSrid != 0:
        vegSrid = h2gisVegSrid
        buildSrid = h2gisVegSrid
    else:
        vegSrid = h2gisVegSrid
        buildSrid = h2gisBuildSrid

    cursor.execute(
        """
       DROP TABLE IF EXISTS {0};
       CREATE TABLE {0}
           AS SELECT ST_SETSRID({1}, {2}) AS {1},
                     {3}, CAST(ROUND({4}, 0) AS INT) AS {4}
           FROM     (SELECT  {1}, {3}, CAST({4} AS DOUBLE) AS {4}
                     FROM {5})
           WHERE {4} > 0.5;
       DROP TABLE IF EXISTS {6};
       CREATE TABLE {6}
           AS SELECT ST_SETSRID({1}, {12}) AS {1},
                     {7}, {8}, {9}, {10}
           FROM {11}
           WHERE {9} > 0.5;
       """.format(
            BUILDING_TABLE_NAME,
            GEOM_FIELD,
            buildSrid,
            ID_FIELD_BUILD,
            HEIGHT_FIELD,
            buildTablePreSrid,
            VEGETATION_TABLE_NAME,
            ID_VEGETATION,
            VEGETATION_CROWN_BASE_HEIGHT,
            VEGETATION_CROWN_TOP_HEIGHT,
            VEGETATION_ATTENUATION_FACTOR,
            vegTablePreSrid,
            vegSrid,
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join([vegTablePreSrid, buildTablePreSrid])
            )
        )


def loadFile(cursor, filePath, tableName, srid=None, srid_repro=None):
    """Load a file in the database according to its extension

        Parameters
        _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    filePath: String
        Path of the file to load
    tableName: String
        Name of the table for the loaded file
    srid: int, default None
        SRID of the loaded file (if known)
    srid_repro: int, default None
        SRID if you want to reproject the data

        Returns
        _ _ _ _ _ _ _ _ _ _

    None"""
    print("Load table '{0}'".format(tableName))

    # Get the input building file extension and the appropriate h2gis read function name
    fileExtension = filePath.split('.')[-1]
    readFunction = DataUtil.readFunction(fileExtension)

    if readFunction == 'CSVREAD':
        cursor.execute(
            """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0} 
               AS SELECT * FROM {2}('{1}');
            """.format(tableName, filePath, readFunction)
        )
    else:  # Import and then copy into a new table to remove all constraints (primary keys...)
        cursor.execute(
            """
           DROP TABLE IF EXISTS TEMPO, {0};
            CALL {2}('{1}','TEMPO');
            CREATE TABLE {0}
                AS SELECT *
                FROM TEMPO;
            """.format(tableName, filePath, readFunction)
        )

    if srid_repro:
        reproject_function = 'ST_TRANSFORM('
        reproject_srid = ', {0})'.format(srid_repro)
    else:
        reproject_function = ''
        reproject_srid = ''

    if srid:
        listCols = DataUtil.getColumns(cursor, tableName)
        listCols.remove(GEOM_FIELD)
        listCols_sql = ','.join(listCols)
        if listCols_sql != '':
            listCols_sql += ','

        cursor.execute(
            """
           DROP TABLE IF EXISTS TEMPO_LOAD;
           CREATE TABLE TEMPO_LOAD
               AS SELECT {0} {4}ST_SETSRID({1}, {2}){5} AS {1}
               FROM {3};
           DROP TABLE {3};
           ALTER TABLE TEMPO_LOAD RENAME TO {3}
           """.format(
                listCols_sql,
                GEOM_FIELD,
                srid,
                tableName,
                reproject_function,
                reproject_srid,
            )
        )


def fromShp3dTo2_5(
    cursor,
    triangles3d,
    TreesZone,
    buildTableName,
    vegTableName,
    prefix=PREFIX_NAME,
    save=True,
):
    """Convert 3D shapefile to 2.5 D shapefiles distinguishing
    buildings from trees if the surface intersecting trees is passed

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            triangles3d: String
                Name of the table containing the 3D geometries from the CAD
            TreesZone: String
                Name of the table containing the zones intersecting trees triangles
            buildTableName: String
                Name of the output building table name
            vegTableName: String
                Name of the output vegetation table name
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name
            save: boolean, default True
                Whether or not the resulting 2.5 layers are saved

                Returns
                _ _ _ _ _ _ _ _ _ _

            None"""
    print('From 3D to 2.5D geometries')

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    trianglesWithId = DataUtil.postfix('triangles_with_id')
    trees2d = DataUtil.postfix('trees_2d')
    buildings2d = DataUtil.postfix('buildings_2d')
    treesCovered = DataUtil.postfix('trees_covered')
    buildingsCovered = DataUtil.postfix('buildings_covered')

    # Add ID to the input data and remove vertical polygons...
    cursor.execute(
        """
       DROP TABLE IF EXISTS {0}; 
       CREATE TABLE {0}(ID SERIAL, {1} GEOMETRY) 
            AS (SELECT NULL, {1} 
                FROM ST_EXPLODE('(SELECT * FROM {2} WHERE ST_AREA({1})>0)'))
            """.format(trianglesWithId, GEOM_FIELD, triangles3d)
    )

    if TreesZone:
        # Identify triangles being trees and convert them to 2D polygons
        cursor.execute(
            """
           {6};
           {7};
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0} 
                AS SELECT   a.ID, ST_FORCE2D(a.{1}) AS {1}, 
                            CAST(ST_ZMAX(a.{1}) AS INT) AS {2},
                            CAST(ST_ZMIN(a.{1}) AS INT) AS {3}
                FROM    {4} AS a, {5} AS b
                WHERE   a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1})
                """.format(
                trees2d,
                GEOM_FIELD,
                VEGETATION_CROWN_TOP_HEIGHT,
                VEGETATION_CROWN_BASE_HEIGHT,
                trianglesWithId,
                TreesZone,
                DataUtil.createIndex(
                    tableName=trianglesWithId, fieldName=GEOM_FIELD, isSpatial=True
                ),
                DataUtil.createIndex(
                    tableName=TreesZone, fieldName=GEOM_FIELD, isSpatial=True
                ),
            )
        )

        # Identify triangles being buildings and convert them to 2D polygons
        cursor.execute(
            """
            {5};
            {6};
            DROP TABLE IF EXISTS {0};
            CREATE TABLE {0} 
                AS SELECT   a.ID, ST_FORCE2D(a.{1}) AS {1},
                            CAST(ST_ZMAX(a.{1}) AS INT) AS {2}
                FROM    {3} AS a LEFT JOIN {4} AS b
                ON      a.ID = b.ID
                WHERE   b.ID IS NULL
                """.format(
                buildings2d,
                GEOM_FIELD,
                HEIGHT_FIELD,
                trianglesWithId,
                trees2d,
                DataUtil.createIndex(
                    tableName=trianglesWithId, fieldName='ID', isSpatial=False
                ),
                DataUtil.createIndex(
                    tableName=trees2d, fieldName='ID', isSpatial=False
                ),
            )
        )

        # Identify unique trees triangles keeping only the highest one whenever
        # 2 triangles are superimposed
        cursor.execute(
            """
            {9};
            {10};
            {11};
            DROP TABLE IF EXISTS {0};
            CREATE TABLE {0} 
                AS SELECT   b.ID AS ID
                FROM    {3} AS a, {3} AS b
                WHERE   a.{1} && b.{1} AND 
                        (ST_COVERS(a.{1}, b.{1}) AND a.{2} > b.{2} OR
                        ST_EQUALS(a.{1}, b.{1}) AND a.{2} = b.{2} AND a.ID < b.ID)
                GROUP BY b.ID;
           CREATE INDEX IF NOT EXISTS id_ID_{0} ON {0}(ID);   
           DROP TABLE IF EXISTS {4};
           CREATE TABLE {4}
               AS SELECT    a.ID AS {5}, a.{1}, a.{2}, 0 AS {6}, {7} AS {8}
               FROM         {3} AS a LEFT JOIN {0} AS b
                            ON a.ID = b.ID
           WHERE    b.ID IS NULL
            """.format(
                treesCovered,
                GEOM_FIELD,
                VEGETATION_CROWN_TOP_HEIGHT,
                trees2d,
                vegTableName,
                ID_VEGETATION,
                VEGETATION_CROWN_BASE_HEIGHT,
                DEFAULT_VEG_ATTEN_FACT,
                VEGETATION_ATTENUATION_FACTOR,
                DataUtil.createIndex(
                    tableName=trees2d, fieldName=GEOM_FIELD, isSpatial=True
                ),
                DataUtil.createIndex(
                    tableName=trees2d,
                    fieldName=VEGETATION_CROWN_TOP_HEIGHT,
                    isSpatial=False,
                ),
                DataUtil.createIndex(
                    tableName=trees2d, fieldName='ID', isSpatial=False
                ),
            )
        )

    else:
        # Convert building triangles to to 2.5D polygons
        cursor.execute(
            """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0} 
                AS SELECT   ID, ST_FORCE2D({1}) AS {1}, 
                            CAST(ST_ZMAX({1}) AS INT) AS {2}
                FROM    {3}
                """.format(buildings2d, GEOM_FIELD, HEIGHT_FIELD, trianglesWithId)
        )

    # Identify unique building triangles keeping only the highest one whenever
    # 2 triangles are superimposed
    cursor.execute(
        """
        {6};
        {7};
        {8};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0} 
            AS SELECT   b.ID AS ID
            FROM    {3} AS a, {3} AS b
            WHERE   a.{1} && b.{1} AND 
                    (ST_COVERS(a.{1}, b.{1}) AND a.{2} > b.{2} OR
                    ST_EQUALS(a.{1}, b.{1}) AND a.{2} = b.{2} AND a.ID < b.ID)
            GROUP BY b.ID;
        {9};   
        DROP TABLE IF EXISTS {4};
        CREATE TABLE {4}
            AS SELECT    a.ID AS {5}, a.{1}, a.{2} 
            FROM         {3} AS a LEFT JOIN {0} AS b
                         ON a.ID = b.ID
        WHERE    b.ID IS NULL
            """.format(
            buildingsCovered,
            GEOM_FIELD,
            HEIGHT_FIELD,
            buildings2d,
            buildTableName,
            ID_FIELD_BUILD,
            DataUtil.createIndex(
                tableName=buildings2d, fieldName='ID', isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=buildings2d, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=buildings2d, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=buildingsCovered, fieldName='ID', isSpatial=False
            ),
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join([trianglesWithId, trees2d, buildings2d])
            )
        )
