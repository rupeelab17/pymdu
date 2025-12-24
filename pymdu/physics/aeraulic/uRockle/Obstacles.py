# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:05:28 2021

@author: Jérémy Bernard, University of Gothenburg
"""

try:
    from GlobalVariables import *
    import DataUtil

except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil


def windRotation(
    cursor,
    dicOfInputTables,
    rotateAngle,
    rotationCenterCoordinates=None,
    prefix=PREFIX_NAME,
):
    """Rotates of 'rotateAngle' degrees counter-clockwise the geometries
    of all tables from the 'rotationCenterCoordinates' specified by the user.
    If none is specified, the center of rotation used is the most North-East
    point of the enveloppe of all geometries contained in all tables.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfInputTables: dictionary of String
                Dictionary of String with type of obstacle as key and input
                table name as value (tables containing the geometries to rotate)
            rotateAngle: float
                Counter clock-wise rotation angle (in degree)
            rotationCenterCoordinates: tuple of float
                x and y values of the point used as center of rotation
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfRotateTables: dictionary
                Map of initial table names as keys and rotated table names as values
            rotationCenterCoordinates: tuple of float
                x and y values of the point used as center of rotation"""
    print('Rotates geometries from {0} degrees'.format(rotateAngle))

    # Calculate the rotation angle in radian
    rotateAngleRad = DataUtil.degToRad(rotateAngle)

    # If not specified, get the most North-East point of the envelope of all
    # geometries of all tables as the center of rotation
    if rotationCenterCoordinates is None:
        queryUnionTables = ' UNION ALL '.join(
            [
                """
                                                SELECT {0} FROM ST_EXPLODE('(SELECT {0} FROM {1})')
                                                """.format(GEOM_FIELD, t)
                for t in dicOfInputTables.values()
            ]
        )
        cursor.execute(
            """
           SELECT  ST_XMAX(ST_EXTENT({0})),
                   ST_YMAX(ST_EXTENT({0}))
           FROM    ({1})""".format(GEOM_FIELD, queryUnionTables)
        )
        rotationCenterCoordinates = cursor.fetchall()[0]

    columnNames = {}
    # Store the column names (except geometry field) of each table into a dictionary
    for i, t in enumerate(dicOfInputTables.values()):
        columnNames[t] = DataUtil.getColumns(cursor=cursor, tableName=t)
        columnNames[t].remove(GEOM_FIELD)

    # Rotate table in one query in order to limit the number of connections
    dicOfRotateTables = {
        t: dicOfInputTables[t] + '_ROTATED' for t in dicOfInputTables.keys()
    }
    sqlRotateQueries = [
        """
        DROP TABLE IF EXISTS {0};
        CREATE TABLE    {0}
            AS SELECT   ST_MAKEVALID(ST_ROTATE({1}, {2}, {3}, {4})) AS {1},
                        {5}
            FROM        {6}""".format(
            dicOfRotateTables[t],
            GEOM_FIELD,
            rotateAngleRad,
            rotationCenterCoordinates[0],
            rotationCenterCoordinates[1],
            ','.join(columnNames[dicOfInputTables[t]]),
            dicOfInputTables[t],
        )
        for t in dicOfRotateTables.keys()
    ]
    cursor.execute(';'.join(sqlRotateQueries))

    return dicOfRotateTables, rotationCenterCoordinates


def createsBlocks(
    cursor,
    inputBuildings,
    snappingTolerance=GEOMETRY_MERGE_TOLERANCE,
    prefix=PREFIX_NAME,
):
    """Creates blocks and stacked blocks from buildings touching each other.

        Parameters
        _ _ _ _ _ _ _ _ _ _

    cursor: conn.cursor
        A cursor object, used to perform spatial SQL queries
    inputBuildings: String
        Name of the table containing building geometries and height
    snappingTolerance: float, default GEOMETRY_MERGE_TOLERANCE
        Distance in meter below which two buildings are
        considered as touching each other (m)
    prefix: String, default PREFIX_NAME
        Prefix to add to the output table name

        Returns
        _ _ _ _ _ _ _ _ _ _

    blockTable: String
        Name of the table containing the block geometries
        (only block of touching buildings independantly of their height)
    stackedBlockTable: String
        Name of the table containing blocks considering the vertical dimension
        (only buildings having the same height are merged)"""
    print('Creates blocks and stacked blocks')

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    correlTable = DataUtil.postfix('correl_table')

    # Creates final tables
    blockTable = DataUtil.prefix('block_table', prefix=prefix)
    stackedBlockTable = DataUtil.prefix('stacked_block_table', prefix=prefix)

    # Creates the block (a method based on network - such as H2network
    # would be much more efficient)
    cursor.execute(
        """
       DROP TABLE IF EXISTS {0}; 
       CREATE TABLE {0} 
            AS SELECT EXPLOD_ID AS {1}, ST_MAKEVALID(ST_SIMPLIFY(ST_NORMALIZE({2}), {5})) AS {2} 
            FROM ST_EXPLODE ('(SELECT ST_UNION(ST_ACCUM(ST_BUFFER({2},{3},''join=mitre'')))
                             AS {2} FROM {4})');
            """.format(
            blockTable,
            ID_FIELD_BLOCK,
            GEOM_FIELD,
            snappingTolerance,
            inputBuildings,
            GEOMETRY_SIMPLIFICATION_DISTANCE,
        )
    )

    # Identify building/block relations and convert building height to integer
    cursor.execute(
        """
       {7};
       {8};
       DROP TABLE IF EXISTS {0};
        CREATE TABLE {0} 
                AS SELECT   a.{1}, a.{2}, CAST(a.{3} AS INT) AS {3}, b.{4},
                            b.{2} AS GEOM_BLOCK
                FROM    {5} AS a, {6} AS b
                WHERE   a.{2} && b.{2} AND ST_INTERSECTS(a.{2}, b.{2});
        """.format(
            correlTable,
            ID_FIELD_BUILD,
            GEOM_FIELD,
            HEIGHT_FIELD,
            ID_FIELD_BLOCK,
            inputBuildings,
            blockTable,
            DataUtil.createIndex(
                tableName=inputBuildings, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=blockTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
        )
    )

    # Identify all possible values of height for buildings being in a more than 1 building block
    cursor.execute(
        """
       {0};
       """.format(
            DataUtil.createIndex(
                tableName=correlTable, fieldName=ID_FIELD_BLOCK, isSpatial=False
            )
        )
    )
    cursor.execute(
        """
       SELECT DISTINCT a.{2} 
       FROM {0} AS a RIGHT JOIN (SELECT {1} FROM {0} GROUP BY {1}) AS b
       ON a.{1} = b.{1};
       """.format(correlTable, ID_FIELD_BLOCK, HEIGHT_FIELD)
    )
    listOfHeight = cursor.fetchall()
    if len(listOfHeight) > 0:
        df_listOfHeight = pd.DataFrame(listOfHeight).dropna()[0].astype(int).values

        # Create stacked blocks according to building blocks and height
        listOfSqlQueries = [
            """ SELECT NULL, {2}, ST_MAKEVALID(ST_NORMALIZE({0})) AS {0} , {4}
                FROM ST_EXPLODE('(SELECT ST_MAKEVALID(ST_SNAP(ST_SIMPLIFY(ST_UNION(ST_ACCUM(ST_BUFFER(a.{0},
                                                                                        {6},
                                                                                        ''join=mitre''))),
                                                                        {5}),
                                                             a.GEOM_BLOCK,
                                                             {6})
                                                        ) AS {0},
                                        a.{2} AS {2}
                                FROM {3} AS a RIGHT JOIN (SELECT {2} 
                                                          FROM {3}
                                                          WHERE {1}={4}) AS b
                                ON a.{2}=b.{2} WHERE a.{1}>={4}
                                GROUP BY a.{2})')
                                """.format(
                GEOM_FIELD,
                HEIGHT_FIELD,
                ID_FIELD_BLOCK,
                correlTable,
                height_i,
                GEOMETRY_SIMPLIFICATION_DISTANCE,
                snappingTolerance,
            )
            for height_i in df_listOfHeight
        ]
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0};
            CREATE TABLE {0}({1} SERIAL, {2} INT, {3} GEOMETRY, {4} INT)
                AS {5}
                """.format(
                stackedBlockTable,
                ID_FIELD_STACKED_BLOCK,
                ID_FIELD_BLOCK,
                GEOM_FIELD,
                HEIGHT_FIELD,
                ' UNION ALL '.join(listOfSqlQueries),
            )
        )

    else:
        # Create an empty block table
        cursor.execute(
            """
            DROP TABLE IF EXISTS {0};
            CREATE TABLE {0}({1} SERIAL, {2} INT, {3} GEOMETRY, {4} INT)
            """.format(
                stackedBlockTable,
                ID_FIELD_STACKED_BLOCK,
                ID_FIELD_BLOCK,
                GEOM_FIELD,
                HEIGHT_FIELD,
            )
        )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute('DROP TABLE IF EXISTS {0}'.format(','.join([correlTable])))

    return blockTable, stackedBlockTable


def identifyBlockAndCavityBase(cursor, stackedBlockTable, prefix=PREFIX_NAME):
    """Identify the base of each block and the base of their cavity zone
    (which may go within the cavity zone of the base block where they sit).
    WARNING: THE CAVITY BASE HEIGHT DEPENDS ON WIND DIRECTION

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            stackedBlockTable: String
                Name of the table containing stacked blocks with block id
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            stackedBlockPropTable: String
                Name of the table containing stacked blocks with block base
                height and block cavity base height"""
    print('Identify block base height and block cavity base')

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    tempoAllStacked = DataUtil.postfix('tempo_all_stacked_table')
    tempoAllBlocks = DataUtil.postfix('tempo_all_blocks_table')
    tempoCavityStacked = DataUtil.postfix('tempo_cavity_stacked_table')
    tempoAllCavityStacked = DataUtil.postfix('tempo_all_cavity_stacked_table')

    # Creates final table
    stackedBlockPropTable = DataUtil.prefix('stacked_block_prop_table', prefix=prefix)

    # Identify each block base height and ratio of area between the stacked and its base block
    cursor.execute(
        """
       {7};
       {8};
       {9};
       DROP TABLE IF EXISTS {4};
       CREATE TABLE {4} 
           AS SELECT   a.{1}, a.{2}, a.{5}, MIN(a.{3}) AS {3}, MAX(b.{3}) AS {6},
                       MAX((ST_XMAX(a.{1})-ST_XMIN(a.{1}))/
                           (ST_XMAX(b.{1})-ST_XMIN(b.{1}))) AS AREA_RATIO
           FROM {0} AS a LEFT JOIN {0} AS b ON a.{2} = b.{2}
           WHERE a.{3} > b.{3} AND a.{1} && b.{1} AND (ST_CONTAINS(b.{1}, a.{1})
                                                       OR ST_OVERLAPS(b.{1}, a.{1}))
           GROUP BY a.{5}, a.{2}
       """.format(
            stackedBlockTable,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            HEIGHT_FIELD,
            tempoAllStacked,
            ID_FIELD_STACKED_BLOCK,
            BASE_HEIGHT_FIELD,
            DataUtil.createIndex(
                tableName=stackedBlockTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=stackedBlockTable, fieldName=ID_FIELD_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=stackedBlockTable, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
        )
    )

    # Set the base height to ground base buildings...
    cursor.execute(
        """
       {8};
       {9};
       DROP TABLE IF EXISTS {4};
       CREATE TABLE {4} 
           AS SELECT   b.{1}, b.{2}, b.{5}, b.{3}, COALESCE(a.{6}, 0) AS {6},
                       COALESCE(a.AREA_RATIO, 0) AS AREA_RATIO
           FROM {0} AS a RIGHT JOIN {7} AS b ON a.{5} = b.{5}
       """.format(
            tempoAllStacked,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            HEIGHT_FIELD,
            tempoAllBlocks,
            ID_FIELD_STACKED_BLOCK,
            BASE_HEIGHT_FIELD,
            stackedBlockTable,
            DataUtil.createIndex(
                tableName=tempoAllStacked,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=stackedBlockTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
    )

    # Calculates the depth where the cavity zone
    # of a upper stacked block may go within the base block cavity zone
    cursor.execute(
        """
       {8};
       {9};
       {10};
       DROP TABLE IF EXISTS {4};
       CREATE TABLE {4} 
           AS SELECT   a.{1}, a.{2}, a.{5}, MIN(a.{3}) AS {3}, MAX(a.{6}) AS {6},
                       MAX(a.{6})-a.AREA_RATIO*MIN(a.{6}-b.{6}) AS {7}
           FROM {0} AS a LEFT JOIN {0} AS b ON a.{2} = b.{2}
           WHERE a.{3} > b.{3} AND a.{1} && b.{1} AND (ST_CONTAINS(b.{1}, a.{1})
                                                       OR ST_OVERLAPS(b.{1}, a.{1}))
           GROUP BY a.{5}, a.{2}
       """.format(
            tempoAllBlocks,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            HEIGHT_FIELD,
            tempoCavityStacked,
            ID_FIELD_STACKED_BLOCK,
            BASE_HEIGHT_FIELD,
            CAVITY_BASE_HEIGHT_FIELD,
            DataUtil.createIndex(
                tableName=tempoAllBlocks, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=tempoAllBlocks, fieldName=ID_FIELD_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoAllBlocks, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
        )
    )

    # Same as previous for stacked buildings being above a ground building (not a stacked one...)
    cursor.execute(
        """
       {9};
       {10};
       DROP TABLE IF EXISTS {4};
       CREATE TABLE {4} 
           AS SELECT   a.{1}, a.{2}, a.{5}, a.{3}, a.{6},
                       COALESCE(a.{7}, 
                                b.{6}*(1-b.AREA_RATIO)) AS {7}
           FROM {0} AS a RIGHT JOIN {8} AS b ON a.{5} = b.{5}
       """.format(
            tempoCavityStacked,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            HEIGHT_FIELD,
            tempoAllCavityStacked,
            ID_FIELD_STACKED_BLOCK,
            BASE_HEIGHT_FIELD,
            CAVITY_BASE_HEIGHT_FIELD,
            tempoAllStacked,
            DataUtil.createIndex(
                tableName=tempoCavityStacked,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoAllStacked,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
    )

    # Join blocks being not stacked
    cursor.execute(
        """
       {9};
       {10};
       DROP TABLE IF EXISTS {3};
       CREATE TABLE {3} 
           AS SELECT   a.{1}, a.{4}, a.{5}, a.{8},
                       COALESCE(b.{6}, 0) AS {6},
                       COALESCE(b.{7}, 0) AS {7}
           FROM {0} AS a LEFT JOIN {2} AS b ON a.{1} = b.{1}
       """.format(
            stackedBlockTable,
            ID_FIELD_STACKED_BLOCK,
            tempoAllCavityStacked,
            stackedBlockPropTable,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            BASE_HEIGHT_FIELD,
            CAVITY_BASE_HEIGHT_FIELD,
            HEIGHT_FIELD,
            DataUtil.createIndex(
                tableName=stackedBlockTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=tempoAllCavityStacked,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join(
                    [
                        tempoAllStacked,
                        tempoCavityStacked,
                        tempoAllCavityStacked,
                        tempoAllBlocks,
                    ]
                )
            )
        )

    return stackedBlockPropTable


def initUpwindFacades(cursor, obstaclesTable, prefix=PREFIX_NAME):
    """Identify upwind facades, convert them to lines (they are initially
    included within polygons) and calculates their direction from wind speed
    (90° for a facade perpendicular from the upwind). Also get the base height
    of each facade.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            obstacleTable: String
                Name of the table containing the obstacle geometries
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            upwindTable: String
                Name of the table containing the upwind obstacle facades"""
    print('Initializes upwind facades')

    # Output base name
    outputBaseName = 'UPWIND_INIT'

    # Name of the output table
    upwindTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Identify upwind facade
    cursor.execute(
        """
       DROP TABLE IF EXISTS {0};
       CREATE TABLE {0}({5} SERIAL, {1} INTEGER, {2} GEOMETRY, {3} DOUBLE, 
                        {6} INTEGER, {7} INTEGER, {8} INTEGER)
           AS SELECT   NULL AS {5},
                       {1},
                       {2} AS {2},
                       ST_AZIMUTH(ST_STARTPOINT({2}), 
                                  ST_ENDPOINT({2})) AS {3},
                       {6},
                       {7},
                       {8}
           FROM ST_EXPLODE('(SELECT ST_TOMULTISEGMENTS({2}) AS {2},
                                  {1},
                                  {6},
                                  {7},
                                  {8}
                                  FROM {4})')
           WHERE ST_AZIMUTH(ST_STARTPOINT({2}), 
                            ST_ENDPOINT({2})) < PI()
           """.format(
            upwindTable,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            obstaclesTable,
            UPWIND_FACADE_FIELD,
            HEIGHT_FIELD,
            BASE_HEIGHT_FIELD,
            ID_FIELD_BLOCK,
        )
    )

    return upwindTable


def updateUpwindFacadeBase(cursor, upwindTable, prefix=PREFIX_NAME):
    """Update the base height of each upwind facade (when shared with a facade
    of a stacked block below).

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            upwindTable: String
                Name of the table containing the initialized upwind facades
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            updatedUpwindBaseTable: String
                Name of the table containing the upwind obstacle facades with
                updated base height"""
    print('Update upwind facades base height')

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    tempoUpwind = DataUtil.postfix('tempo_upwind')

    # Output base name
    outputBaseName = 'UPWIND_UPDATED_BASE'

    # Name of the output table
    updatedUpwindBaseTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Update base height for facades being shared with the block below
    cursor.execute(
        """
       {10};
       {11};
       {12};
       {13};
       DROP TABLE IF EXISTS {4};
       CREATE TABLE {4} 
           AS SELECT   a.{1}, a.{5}, a.{8}, MIN(a.{3}) AS {3}, MIN(b.{6}) AS {6},
                       MIN(a.{7})
           FROM     {0} AS a LEFT JOIN {0} AS b ON a.{2} = b.{2}
           WHERE   a.{3} > b.{3} AND a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1})
                   AND ST_DIMENSION(ST_INTERSECTION(ST_SNAP(a.{1}, b.{1}, {9}),
                                                    b.{1}))=1
           GROUP BY a.{5}, a.{2}, a.{8}
       """.format(
            upwindTable,
            GEOM_FIELD,
            ID_FIELD_BLOCK,
            HEIGHT_FIELD,
            tempoUpwind,
            ID_FIELD_STACKED_BLOCK,
            BASE_HEIGHT_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            UPWIND_FACADE_FIELD,
            SNAPPING_TOLERANCE,
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=ID_FIELD_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=HEIGHT_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=UPWIND_FACADE_FIELD, isSpatial=False
            ),
        )
    )

    # Upwind facades being not updated are joined to the updated ones
    cursor.execute(
        """
       {9};
       {10};
       DROP TABLE IF EXISTS {3};
       CREATE TABLE {3} 
           AS SELECT   a.{1}, a.{4}, a.{5}, a.{7}, a.{8}, a.{11},
                       COALESCE(b.{6}, a.{6}) AS {6}
           FROM {0} AS a LEFT JOIN {2} AS b ON a.{5} = b.{5}
       """.format(
            upwindTable,
            ID_FIELD_STACKED_BLOCK,
            tempoUpwind,
            updatedUpwindBaseTable,
            GEOM_FIELD,
            UPWIND_FACADE_FIELD,
            BASE_HEIGHT_FIELD,
            HEIGHT_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=UPWIND_FACADE_FIELD, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=tempoUpwind, fieldName=UPWIND_FACADE_FIELD, isSpatial=False
            ),
            ID_FIELD_BLOCK,
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute('DROP TABLE IF EXISTS {0}'.format(','.join([tempoUpwind])))

    return updatedUpwindBaseTable


def initDownwindFacades(cursor, obstaclesTable, prefix=PREFIX_NAME):
    """Identify downwind facades, convert them to lines (they are initially
    included within polygons) and then union those which intersect each other.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            obstacleTable: String
                Name of the table containing the obstacle geometries and properties
                ('EFFECTIVE_LENGTH_FIELD', 'EFFECTIVE_WIDTH_FIELD' and
                 'HEIGHT_FIELD')
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            downwindTable: String
                Name of the table containing the downwind obstacle facades"""
    print('Initializes downwind facades')

    # Output base name
    outputBaseName = 'DOWNWIND_FACADES'

    # Name of the output table
    downwindTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the process)
    tempoDownwindSegments = DataUtil.postfix('TEMPO_DOWNWIND_SEGMENTS')
    tempoDownwindLines = DataUtil.postfix('TEMPO_DOWNWIND_LINES')

    # Identify upwind facade
    cursor.execute(
        """
       DROP TABLE IF EXISTS {0};
       CREATE TABLE {0}({4} SERIAL, {1} INTEGER, {2} GEOMETRY, 
                        {5} INTEGER)
           AS SELECT   NULL AS {4},
                       {1},
                       {2} AS {2},
                       {5}
           FROM ST_EXPLODE('(SELECT ST_TOMULTISEGMENTS({2}) AS {2},
                                  {1},
                                  {5}
                                  FROM {3})')
           WHERE ST_AZIMUTH(ST_STARTPOINT({2}), 
                            ST_ENDPOINT({2})) > PI()
           """.format(
            tempoDownwindSegments,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            obstaclesTable,
            DOWNWIND_FACADE_FIELD,
            HEIGHT_FIELD,
        )
    )

    # Merge the ones that intersect each other and join stacked block properties
    cursor.execute(
        """
       DROP TABLE IF EXISTS {0}; 
       CREATE TABLE {0}({1} SERIAL, {2} GEOMETRY, {4} INTEGER) 
            AS SELECT NULL AS {1}, ST_NORMALIZE({2}) AS {2}, {4} 
            FROM ST_EXPLODE ('(SELECT ST_LINEMERGE(ST_UNION(ST_ACCUM({2}))) AS {2}, {4} 
                               FROM {3}
                               GROUP BY {4})');
       {5};{6};
       DROP TABLE IF EXISTS {7};
       CREATE TABLE {7}
           AS SELECT    a.{1}, a.{2}, a.{4}, b.{8}, b.{9}, b.{10},
                       (ST_XMAX(b.{2}) + ST_XMIN(b.{2})) / 2 AS {12},
                       (ST_XMAX(b.{2}) - ST_XMIN(b.{2})) AS {13},
                       b.{14}, b.{15}, b.{16}, b.{17}, b.{18}, b.{19}
           FROM {0} AS a LEFT JOIN {11} AS b
           ON a.{4} = b.{4}
            """.format(
            tempoDownwindLines,
            DOWNWIND_FACADE_FIELD,
            GEOM_FIELD,
            tempoDownwindSegments,
            ID_FIELD_STACKED_BLOCK,
            DataUtil.createIndex(
                tableName=tempoDownwindLines,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=obstaclesTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            downwindTable,
            CAVITY_LENGTH_FIELD,
            WAKE_LENGTH_FIELD,
            HEIGHT_FIELD,
            obstaclesTable,
            STACKED_BLOCK_X_MED,
            STACKED_BLOCK_WIDTH,
            STACKED_BLOCK_UPSTREAMEST_X,
            SIN_BLOCK_LEFT_AZIMUTH,
            COS_BLOCK_LEFT_AZIMUTH,
            COS_BLOCK_RIGHT_AZIMUTH,
            SIN_BLOCK_RIGHT_AZIMUTH,
            ID_FIELD_BLOCK,
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join([tempoDownwindSegments, tempoDownwindLines])
            )
        )

    return downwindTable
