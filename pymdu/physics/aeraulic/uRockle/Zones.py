# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:27:25 2021

@author: Jérémy Bernard, University of Gothenburg
"""

try:
    from GlobalVariables import *
    import DataUtil
except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil


def displacementZones(
    cursor, upwindTable, zonePropertiesTable, srid, prefix=PREFIX_NAME
):
    """Creates the displacement zone and the displacement vortex zone
    for each of the building upwind facade based on Kaplan et Dinar (1996)
    for the equations of the ellipsoid
        - Equation 2 when the facade is perpendicular to the wind,
        - Figure 2 and Table 1 when the facade has an angle Theta with the wind.
    Note that the displacement vortex zone is only calculated is the facade is
    nearly perpendicular to wind direction.

    Obstacle length and width in the equations are given in an input table.
    Note that we strongly recommand to use the 'CalculatesIndicators.zoneProperties' function
    to calculate effective length and width instead of maximum length and width...

    References:
       Kaplan, H., et N. Dinar. « A Lagrangian Dispersion Model for Calculating
       Concentration Distribution within a Built-up Domain ». Atmospheric
       Environment 30, nᵒ 24 (1 décembre 1996): 4197‑4207.
       https://doi.org/10.1016/1352-2310(96)00144-6.


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            upwindTable: String
                Name of the table containing upwind segment geometries
                (and also the ID of each stacked obstacle)
            zonePropertiesTable: String
                Name of the table containing obstacle zone properties
                (and also the ID of each stacked obstacle)
            srid: int
                SRID of the building data (useful for zone calculation)
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            displacementZonesTable: String
                Name of the table containing the displacement zones
            displacementVortexZonesTable: String
                Name of the table containing the displacement vortex zones"""
    print('Creates displacement zones')

    # Output base name
    outputBaseDispName = 'DISPLACEMENT_ZONES'
    outputBaseDispVortexName = 'DISPLACEMENT_VORTEX_ZONES'

    # Name of the output table
    displacementZonesTable = DataUtil.prefix(outputBaseDispName, prefix=prefix)
    displacementVortexZonesTable = DataUtil.prefix(
        outputBaseDispVortexName, prefix=prefix
    )

    # Separate the query into two almost similar queries having only
    # different case when conditions and ellipse size
    partOfQueryThatDiffer = pd.DataFrame(
        {
            'where': [
                ' b.{0}*SIN(a.{1})*SIN(a.{1})>{2}'.format(
                    DISPLACEMENT_LENGTH_FIELD,
                    UPWIND_FACADE_ANGLE_FIELD,
                    ELLIPSOID_MIN_LENGTH,
                ),
                ' a.{0}>RADIANS(90-{1}) AND a.{0}<RADIANS(90+{1}) '.format(
                    UPWIND_FACADE_ANGLE_FIELD, PERPENDICULAR_THRESHOLD_ANGLE
                ),
            ],
            'length': [DISPLACEMENT_LENGTH_FIELD, DISPLACEMENT_LENGTH_VORTEX_FIELD],
            'table': [displacementZonesTable, displacementVortexZonesTable],
        },
        index=['displacement', 'vortex'],
    )
    query = [
        """
        {12};
        {13};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT   {1},
                        ST_SETSRID({2}, {14}) AS {2},
                        {3},
                        {4},
                        {8}
            FROM ST_EXPLODE('(SELECT ST_SPLIT(ST_SNAP(ST_ROTATE(ST_SETSRID(ST_MAKEELLIPSE(ST_CENTROID(a.{2}),
                                                                                            ST_LENGTH(a.{2}),
                                                                                            2*b.{5}*SIN(a.{4})*SIN(a.{4})),
                                                                            {14}),
                                                                0.5*PI()-a.{4}),
                                                     a.{2},
                                                     {10}),
                                             a.{2}) AS {2},
                                     a.{1},
                                     b.{3},
                                     a.{4},
                                     a.{8},
                                     ST_LENGTH(a.{2})/2 AS R_x,
                                     b.{5}*SIN(a.{4})*SIN(a.{4}) AS R_y
                             FROM {6} AS a LEFT JOIN {7} AS b ON a.{8} = b.{8}
                             WHERE {9})')
             WHERE      {4}>=0.5*PI()
                            -0.5*PI()+ACOS((1-COS(2*PI()/{11}))*R_x
                                  /SQRT(POWER((1-COS(2*PI()/{11}))*R_x,2)
                                        +POWER(SIN(2*PI()/{11})*R_y,2)))
                   AND EXPLOD_ID = 2 
                   OR   {4}<0.5*PI()
                            -0.5*PI()+ACOS((1-COS(2*PI()/{11}))*R_x
                                  /SQRT(POWER((1-COS(2*PI()/{11}))*R_x,2)
                                        +POWER(SIN(2*PI()/{11})*R_y,2)))
                   AND EXPLOD_ID = 1
           """.format(
            partOfQueryThatDiffer.loc[zone, 'table'],
            UPWIND_FACADE_FIELD,
            GEOM_FIELD,
            HEIGHT_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            partOfQueryThatDiffer.loc[zone, 'length'],
            upwindTable,
            zonePropertiesTable,
            ID_FIELD_STACKED_BLOCK,
            partOfQueryThatDiffer.loc[zone, 'where'],
            SNAPPING_TOLERANCE,
            NPOINTS_ELLIPSE,
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=zonePropertiesTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            srid,
        )
        for zone in partOfQueryThatDiffer.index
    ]
    cursor.execute(';'.join(query))

    return displacementZonesTable, displacementVortexZonesTable


def displacementZones(
    cursor, upwindTable, zonePropertiesTable, srid, prefix=PREFIX_NAME
):
    """Creates the displacement zone and the displacement vortex zone
    for each of the building upwind facade based on Kaplan et Dinar (1996)
    for the equations of the ellipsoid
        - Equation 2 when the facade is perpendicular to the wind,
        - Figure 2 and Table 1 when the facade has an angle Theta with the wind.
    Note that the displacement vortex zone is only calculated is the facade is
    nearly perpendicular to wind direction.

    Obstacle length and width in the equations are given in an input table.
    Note that we strongly recommand to use the 'CalculatesIndicators.zoneProperties' function
    to calculate effective length and width instead of maximum length and width...

    References:
       Kaplan, H., et N. Dinar. « A Lagrangian Dispersion Model for Calculating
       Concentration Distribution within a Built-up Domain ». Atmospheric
       Environment 30, nᵒ 24 (1 décembre 1996): 4197‑4207.
       https://doi.org/10.1016/1352-2310(96)00144-6.


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            upwindTable: String
                Name of the table containing upwind segment geometries
                (and also the ID of each stacked obstacle)
            zonePropertiesTable: String
                Name of the table containing obstacle zone properties
                (and also the ID of each stacked obstacle)
            srid: int
                SRID of the building data (useful for zone calculation)
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            displacementZonesTable: String
                Name of the table containing the displacement zones
            displacementVortexZonesTable: String
                Name of the table containing the displacement vortex zones"""
    print('Creates displacement zones')

    # Output base name
    outputBaseDispName = 'DISPLACEMENT_ZONES'
    outputBaseDispVortexName = 'DISPLACEMENT_VORTEX_ZONES'

    # Name of the output table
    displacementZonesTable = DataUtil.prefix(outputBaseDispName, prefix=prefix)
    displacementVortexZonesTable = DataUtil.prefix(
        outputBaseDispVortexName, prefix=prefix
    )

    # Separate the query into two almost similar queries having only
    # different case when conditions and ellipse size
    partOfQueryThatDiffer = pd.DataFrame(
        {
            'where': [
                ' b.{0}*SIN(a.{1})*SIN(a.{1})>{2}'.format(
                    DISPLACEMENT_LENGTH_FIELD,
                    UPWIND_FACADE_ANGLE_FIELD,
                    ELLIPSOID_MIN_LENGTH,
                ),
                ' a.{0}>RADIANS(90-{1}) AND a.{0}<RADIANS(90+{1}) '.format(
                    UPWIND_FACADE_ANGLE_FIELD, PERPENDICULAR_THRESHOLD_ANGLE
                ),
            ],
            'length': [DISPLACEMENT_LENGTH_FIELD, DISPLACEMENT_LENGTH_VORTEX_FIELD],
            'table': [displacementZonesTable, displacementVortexZonesTable],
        },
        index=['displacement', 'vortex'],
    )
    query = [
        """
        {12};
        {13};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT   {1},
                        ST_SETSRID({2}, {14}) AS {2},
                        {3},
                        {4},
                        {8}
            FROM ST_EXPLODE('(SELECT ST_SPLIT(ST_SNAP(ST_ROTATE(ST_SETSRID(ST_MAKEELLIPSE(ST_CENTROID(a.{2}),
                                                                                            ST_LENGTH(a.{2}),
                                                                                            2*b.{5}*SIN(a.{4})*SIN(a.{4})),
                                                                            {14}),
                                                                0.5*PI()-a.{4}),
                                                     a.{2},
                                                     {10}),
                                             a.{2}) AS {2},
                                     a.{1},
                                     b.{3},
                                     a.{4},
                                     a.{8},
                                     ST_LENGTH(a.{2})/2 AS R_x,
                                     b.{5}*SIN(a.{4})*SIN(a.{4}) AS R_y
                             FROM {6} AS a LEFT JOIN {7} AS b ON a.{8} = b.{8}
                             WHERE {9})')
             WHERE      {4}>=0.5*PI()
                            -0.5*PI()+ACOS((1-COS(2*PI()/{11}))*R_x
                                  /SQRT(POWER((1-COS(2*PI()/{11}))*R_x,2)
                                        +POWER(SIN(2*PI()/{11})*R_y,2)))
                   AND EXPLOD_ID = 2 
                   OR   {4}<0.5*PI()
                            -0.5*PI()+ACOS((1-COS(2*PI()/{11}))*R_x
                                  /SQRT(POWER((1-COS(2*PI()/{11}))*R_x,2)
                                        +POWER(SIN(2*PI()/{11})*R_y,2)))
                   AND EXPLOD_ID = 1
           """.format(
            partOfQueryThatDiffer.loc[zone, 'table'],
            UPWIND_FACADE_FIELD,
            GEOM_FIELD,
            HEIGHT_FIELD,
            UPWIND_FACADE_ANGLE_FIELD,
            partOfQueryThatDiffer.loc[zone, 'length'],
            upwindTable,
            zonePropertiesTable,
            ID_FIELD_STACKED_BLOCK,
            partOfQueryThatDiffer.loc[zone, 'where'],
            SNAPPING_TOLERANCE,
            NPOINTS_ELLIPSE,
            DataUtil.createIndex(
                tableName=upwindTable, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=zonePropertiesTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            srid,
        )
        for zone in partOfQueryThatDiffer.index
    ]
    cursor.execute(';'.join(query))

    return displacementZonesTable, displacementVortexZonesTable


def cavityAndWakeZones(
    cursor, downwindWithPropTable, srid, ellipseResolution, prefix=PREFIX_NAME
):
    """Creates the cavity and wake zones for each of the stacked building
    based on Kaplan et Dinar (1996) for the equations of the ellipsoid
    (Equation 3). When the building has a non rectangular shape or is not
    perpendicular to the wind direction, use the principles of Figure 1b
    in Nelson et al. (2008): the half-ellipse is created from the downwind facade
    coordinates.

    Obstacle length and width in the equations are given in an input table.
    Note that we strongly recommand to use the 'calculatesZoneLength' function
    to calculate effective length and width instead of maximum length and width...

    References:
            Kaplan, H., et N. Dinar. « A Lagrangian Dispersion Model for Calculating
        Concentration Distribution within a Built-up Domain ». Atmospheric
        Environment 30, nᵒ 24 (1 décembre 1996): 4197‑4207.
        https://doi.org/10.1016/1352-2310(96)00144-6.
           Nelson, Matthew, Bhagirath Addepalli, Fawn Hornsby, Akshay Gowardhan,
        Eric Pardyjak, et Michael Brown. « 5.2 Improvements to a Fast-Response
        Urban Wind Model », 2008.


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            downwindWithPropTable: String
                Name of the table containing downwind facade geometries and
                its corresponding stacked block zone properties
            srid: int
                SRID of the building data (useful for zone calculation)
            ellipseResolution: float
                "Kind of" horizontal resolution of the ellipse (in meter)
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            outputZoneTableNames: dictionary
                Contains as key the name of the zone type ('CAVITY_NAME' and 'WAKE_NAME')
                and as value the name of the table containing the zones"""
    print('Creates cavity and wake zones')

    # Output base name
    outputZoneTableNames = {
        CAVITY_NAME: DataUtil.prefix('CAVITY_ZONES', prefix=prefix),
        WAKE_NAME: DataUtil.prefix('WAKE_ZONES', prefix=prefix),
    }

    # Create temporary table names (for tables that will be removed at the end of the process)
    densifiedLinePoints = DataUtil.postfix('DENSIFIED_LINE_POINTS')
    ZonePoints = {
        CAVITY_NAME: CAVITY_NAME + DataUtil.postfix('_ZONE_POINTS'),
        WAKE_NAME: WAKE_NAME + DataUtil.postfix('_ZONE_POINTS'),
    }
    ZonePolygons = {
        CAVITY_NAME: CAVITY_NAME + DataUtil.postfix('_ZONE_POLYGONS'),
        WAKE_NAME: WAKE_NAME + DataUtil.postfix('_ZONE_POLYGONS'),
    }

    # First densify the downwind facades
    cursor.execute(
        f"""
       DROP TABLE IF EXISTS {densifiedLinePoints};
       CREATE TABLE {densifiedLinePoints}
           AS SELECT   EXPLOD_ID, 
                       {GEOM_FIELD}, 
                       X_MED, 
                       HALF_WIDTH, 
                       {CAVITY_LENGTH_FIELD},
                       {WAKE_LENGTH_FIELD}, 
                       {DOWNWIND_FACADE_FIELD}
           FROM ST_EXPLODE('(SELECT ST_ACCUM(ST_TOMULTIPOINT(ST_DENSIFY({GEOM_FIELD}, 
                                                            ST_LENGTH({GEOM_FIELD})/{CAV_N_WAKE_FACADE_NPOINTS}))) AS {GEOM_FIELD},
                                    MAX({CAVITY_LENGTH_FIELD}) AS {CAVITY_LENGTH_FIELD}, 
                                    MAX({WAKE_LENGTH_FIELD}) AS {WAKE_LENGTH_FIELD},
                                    {DOWNWIND_FACADE_FIELD},
                                    MAX(X_MED) AS X_MED, 
                                    MAX(HALF_WIDTH) AS HALF_WIDTH
                           FROM ST_EXPLODE(''(SELECT ST_TOMULTISEGMENTS({GEOM_FIELD}) AS {GEOM_FIELD},
                                                    {CAVITY_LENGTH_FIELD}, 
                                                    {WAKE_LENGTH_FIELD}, 
                                                    {DOWNWIND_FACADE_FIELD},
                                                    {STACKED_BLOCK_X_MED} AS X_MED,
                                                    {STACKED_BLOCK_WIDTH} / 2 AS HALF_WIDTH
                                           FROM {downwindWithPropTable})'')
                           GROUP BY {DOWNWIND_FACADE_FIELD})')
       """
    )

    # Define the names of variables for cavity and wake zones
    variablesNames = pd.DataFrame(
        {'L': [CAVITY_LENGTH_FIELD, WAKE_LENGTH_FIELD]}, index=[CAVITY_NAME, WAKE_NAME]
    )

    # Create the half ellipse for cavity and wake zones from the densified downwind facade points
    cursor.execute(
        ';'.join(
            [
                """
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT {1}, {2}, EXPLOD_ID
            FROM {3}
            UNION ALL
            SELECT  CASE WHEN ST_X({1}) - X_MED < HALF_WIDTH AND ST_X({1}) - X_MED > - HALF_WIDTH
                         THEN ST_TRANSLATE({1}, 0, -{4}*SQRT(1-POWER((ST_X({1}) - X_MED) /
                                                          HALF_WIDTH, 2)))
                         ELSE {1}
                         END  AS {1},
                    {2}, -EXPLOD_ID + 1000 AS EXPLOD_ID
            FROM {3}
            UNION ALL
            SELECT {1}, {2}, 10000 AS EXPLOD_ID
            FROM {3}
            WHERE EXPLOD_ID = 1
            ORDER BY EXPLOD_ID ASC
        """.format(
                    ZonePoints[z],
                    GEOM_FIELD,
                    DOWNWIND_FACADE_FIELD,
                    densifiedLinePoints,
                    variablesNames.loc[z, 'L'],
                )
                for z in variablesNames.index
            ]
        )
    )

    # Create the zone from the half ellipse and the densified line and then join missing columns
    cursor.execute(
        ';'.join(
            [
                f"""
        {DataUtil.createIndex(tableName=ZonePoints[z],
                              fieldName=DOWNWIND_FACADE_FIELD,
                              isSpatial=False)}
        DROP TABLE IF EXISTS {ZonePolygons[z]}, {outputZoneTableNames[z]};
        CREATE TABLE {ZonePolygons[z]}
            AS SELECT   ST_MAKEVALID(ST_MAKEPOLYGON(ST_MAKELINE(ST_ACCUM(ST_PRECISIONREDUCER({GEOM_FIELD},2))))) AS {GEOM_FIELD},
                        {DOWNWIND_FACADE_FIELD}
            FROM {ZonePoints[z]}
            GROUP BY {DOWNWIND_FACADE_FIELD};
        {DataUtil.createIndex(tableName=ZonePolygons[z],
                              fieldName=DOWNWIND_FACADE_FIELD,
                              isSpatial=False)}
        {DataUtil.createIndex(tableName=downwindWithPropTable,
                              fieldName=DOWNWIND_FACADE_FIELD,
                              isSpatial=False)}
        CREATE TABLE {outputZoneTableNames[z]}
            AS SELECT   a.{DOWNWIND_FACADE_FIELD}, 
                        a.{GEOM_FIELD}, b.{ID_FIELD_STACKED_BLOCK},
                        b.{HEIGHT_FIELD},
                        b.{STACKED_BLOCK_X_MED},
                        b.{STACKED_BLOCK_UPSTREAMEST_X},
                        b.{SIN_BLOCK_LEFT_AZIMUTH}, 
                        b.{COS_BLOCK_LEFT_AZIMUTH},
                        b.{COS_BLOCK_RIGHT_AZIMUTH},
                        b.{SIN_BLOCK_RIGHT_AZIMUTH}, 
                        b.{STACKED_BLOCK_WIDTH},
                        b.{ID_FIELD_BLOCK}
            FROM {ZonePolygons[z]} AS a LEFT JOIN {downwindWithPropTable} AS b
            ON a.{DOWNWIND_FACADE_FIELD} = b.{DOWNWIND_FACADE_FIELD}
            WHERE ST_AREA(a.{GEOM_FIELD}) > 0;
        """
                for z in variablesNames.index
            ]
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            """
           DROP TABLE IF EXISTS {0}
           """.format(
                ','.join(
                    [densifiedLinePoints]
                    + list(ZonePoints.values())
                    + list(ZonePolygons.values())
                )
            )
        )

    return outputZoneTableNames


def streetCanyonZones(
    cursor,
    cavityZonesTable,
    zonePropertiesTable,
    upwindTable,
    downwindTable,
    srid,
    prefix=PREFIX_NAME,
):
    """Creates the street canyon zones for each of the stacked building
    based on Nelson et al. (2008) Figure 8b. The method is slightly different
    since we use the cavity zone instead of the Lr buffer.

    References:
           Nelson, Matthew, Bhagirath Addepalli, Fawn Hornsby, Akshay Gowardhan,
        Eric Pardyjak, et Michael Brown. « 5.2 Improvements to a Fast-Response
        Urban Wind Model », 2008.


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            cavityZonesTable: String
                Name of the table containing the cavity zones and the ID of
                each stacked obstacle
            zonePropertiesTable: String
                Name of the table containing the geometry, zone length, height
                and ID of each stacked obstacle
            upwindTable: String
                Name of the table containing upwind segment geometries
                (and also the ID of each stacked obstacle)
            downwindTable: String
                Name of the table containing downwind line geometries and ID
            srid: int
                SRID of the building data (useful for zone calculation)
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            streetCanyonZoneTable: String
                Name of the table containing the street canyon zones"""
    print('Creates street canyon zones')

    # Output base name
    outputBaseName = 'STREETCANYON_ZONE'

    # Name of the output tables
    streetCanyonZoneTable = DataUtil.prefix(outputBaseName, prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    intersectTable = DataUtil.postfix('intersect_table')
    canyonExtendTable = DataUtil.postfix('canyon_extend_table')

    # Identify pieces of upwind facades intersected by cavity zones (only when street canyon angle < 45°)
    cursor.execute(
        f"""
        {DataUtil.createIndex(tableName=upwindTable,
                              fieldName=GEOM_FIELD,
                              isSpatial=True)};
        {DataUtil.createIndex(tableName=cavityZonesTable,
                              fieldName=GEOM_FIELD,
                              isSpatial=True)};
        {DataUtil.createIndex(tableName=upwindTable,
                              fieldName=ID_FIELD_BLOCK,
                              isSpatial=False)};
        {DataUtil.createIndex(tableName=cavityZonesTable,
                              fieldName=ID_FIELD_BLOCK,
                              isSpatial=False)};
        DROP TABLE IF EXISTS {intersectTable};
        CREATE TABLE {intersectTable}
            AS SELECT   b.{ID_FIELD_STACKED_BLOCK} AS {ID_UPSTREAM_STACKED_BLOCK},
                        a.{ID_FIELD_STACKED_BLOCK} AS {ID_DOWNSTREAM_STACKED_BLOCK},
                        a.{BASE_HEIGHT_FIELD},
                        a.{HEIGHT_FIELD},
                        a.{UPWIND_FACADE_ANGLE_FIELD},
                        ST_COLLECTIONEXTRACT(ST_INTERSECTION(a.{GEOM_FIELD}, 
                                                             b.{GEOM_FIELD}),
                                             2) AS {GEOM_FIELD},
                        a.{UPWIND_FACADE_FIELD},
                        b.{DOWNWIND_FACADE_FIELD}
            FROM {upwindTable} AS a, {cavityZonesTable} AS b
            WHERE   a.{GEOM_FIELD} && b.{GEOM_FIELD} AND ST_INTERSECTS(a.{GEOM_FIELD},
                                                                       b.{GEOM_FIELD})
                    AND a.{UPWIND_FACADE_ANGLE_FIELD} >= RADIANS({STREET_CANYON_ANGLE_THRESH}) 
                    AND a.{UPWIND_FACADE_ANGLE_FIELD} <= RADIANS(180-{STREET_CANYON_ANGLE_THRESH})
                    AND a.{ID_FIELD_BLOCK} != b.{ID_FIELD_BLOCK}
            """
    )

    # Identify street canyon extend
    canyonExtendQuery = """
        {14};
        {15};
        DROP TABLE IF EXISTS {3};
        CREATE TABLE {3}
            AS SELECT   a.{1},
                        a.{9},
                        a.{6} AS {7},
                        b.{6} AS {8},
                        a.{11},
                        a.{12},
                        ST_SETSRID(ST_MAKEPOLYGON(ST_MAKELINE(ST_STARTPOINT(a.{4}),
                    								ST_STARTPOINT(ST_TRANSLATE( a.{4}, 
                                                                            0, 
                                                                            ST_YMAX(b.{4})-ST_YMIN(b.{4})+b.{5})),
                    								ST_ENDPOINT(ST_TRANSLATE(   a.{4},
                                                                            0, 
                                                                            ST_YMAX(b.{4})-ST_YMIN(b.{4})+b.{5})),
                    								ST_TOMULTIPOINT(ST_REVERSE(a.{4})))),
                                   {16}) AS THE_GEOM,
                        a.{13},
                        a.{17}
            FROM {0} AS a LEFT JOIN {2} AS b ON a.{1} = b.{10}
            WHERE NOT ST_ISEMPTY(a.{4})
           """.format(
        intersectTable,
        ID_UPSTREAM_STACKED_BLOCK,
        zonePropertiesTable,
        canyonExtendTable,
        GEOM_FIELD,
        CAVITY_LENGTH_FIELD,
        HEIGHT_FIELD,
        DOWNSTREAM_HEIGHT_FIELD,
        UPSTREAM_HEIGHT_FIELD,
        ID_DOWNSTREAM_STACKED_BLOCK,
        ID_FIELD_STACKED_BLOCK,
        UPWIND_FACADE_ANGLE_FIELD,
        BASE_HEIGHT_FIELD,
        UPWIND_FACADE_FIELD,
        DataUtil.createIndex(
            tableName=intersectTable,
            fieldName=ID_UPSTREAM_STACKED_BLOCK,
            isSpatial=False,
        ),
        DataUtil.createIndex(
            tableName=zonePropertiesTable,
            fieldName=ID_FIELD_STACKED_BLOCK,
            isSpatial=False,
        ),
        srid,
        DOWNWIND_FACADE_FIELD,
    )
    cursor.execute(canyonExtendQuery)

    # Creates street canyon zones
    streetCanyonQuery = """
        {15};
        DROP TABLE IF EXISTS {2};
        CREATE TABLE {2}({13} SERIAL,
                         {1} INTEGER,
                         {8} INTEGER,
                         {3} GEOMETRY,
                         {4} INTEGER,
                         {5} INTEGER,
                         {11} DOUBLE,
                         {12} INTEGER,
                         {14} INTEGER,
                         {9} INTEGER)
            AS SELECT   NULL AS {13},
                        {1},
                        {8},
                        ST_SETSRID({3}, {16}) AS {3},
                        {4},
                        {5},
                        {11},
                        {12},
                        {14},
                        {9}
            FROM ST_EXPLODE('(SELECT    a.{1},
                                        a.{8},
                                        ST_SPLIT(a.{3},
                                                 ST_SNAP(b.{3}, a.{3}, 0.01)) AS {3},
                                        a.{4},
                                        a.{5},
                                        a.{11},
                                        a.{12},
                                        a.{14},
                                        a.{9}
                            FROM        {0} AS a LEFT JOIN {7} AS b ON a.{9}=b.{9})')
            WHERE EXPLOD_ID = 1
           """.format(
        canyonExtendTable,
        ID_UPSTREAM_STACKED_BLOCK,
        streetCanyonZoneTable,
        GEOM_FIELD,
        DOWNSTREAM_HEIGHT_FIELD,
        UPSTREAM_HEIGHT_FIELD,
        SNAPPING_TOLERANCE,
        downwindTable,
        ID_DOWNSTREAM_STACKED_BLOCK,
        DOWNWIND_FACADE_FIELD,
        MESH_SIZE,
        UPWIND_FACADE_ANGLE_FIELD,
        BASE_HEIGHT_FIELD,
        ID_FIELD_CANYON,
        UPWIND_FACADE_FIELD,
        DataUtil.createIndex(
            tableName=canyonExtendTable,
            fieldName=ID_UPSTREAM_STACKED_BLOCK,
            isSpatial=False,
        ),
        srid,
    )
    cursor.execute(streetCanyonQuery)

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join([intersectTable, canyonExtendTable])
            )
        )

    return streetCanyonZoneTable


def rooftopZones(cursor, upwindTable, zonePropertiesTable, prefix=PREFIX_NAME):
    """Creates the rooftop zones for each of the upwind facade:
        - recirculation zone if the angle between the wind and the facade is included
        within the range [90-PERPENDICULAR_THRESHOLD_ANGLE, 90+PERPENDICULAR_THRESHOLD_ANGLE].
        See Pol et al. (2006) for more details
        - corner zones if the angle between the wind and the facade is included
        within the range [90-CORNER_THRESHOLD_ANGLE[1], 90-CORNER_THRESHOLD_ANGLE[0]]
        or [90+CORNER_THRESHOLD_ANGLE[0], 90+CORNER_THRESHOLD_ANGLE[1]]. See
        Bagal et al. (2004) for more details

    Obstacle length and width in the equations are given in an input table.
    Note that we strongly recommand to use the 'CalculatesIndicators.zoneProperties' function
    to calculate effective length and width instead of maximum length and width...

    References:
            Pol, SU, NL Bagal, B Singh, MJ Brown, et ER Pardyjak. « IMPLEMENTATION
        OF A ROOFTOP RECIRCULATION PARAMETERIZATION INTO THE QUIC FAST
        RESPONSE URBAN WIND MODEL », 2006.
            Bagal, NL, B Singh, ER Pardyjak, et MJ Brown. « Implementation of
        rooftop recirculation parameterization into the QUIC fast response urban
        wind model ». In Proc. 5th AMS Urban Environ. Symp. Conf, 2004.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            upwindTable: String
                Name of the table containing upwind segment geometries
                (and also the ID of each stacked obstacle)
            zonePropertiesTable: String
                Name of the table stacked obstacle geometries and zone properties
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            rooftopPerpendicularZoneTable: String
                Name of the table containing the rooftop perpendicular zones
            rooftopCornerZoneTable: String
                Name of the table containing the rooftop corner zones"""
    print('Creates rooftop zones (perpendicular and corner)')

    # Output base name
    outputBaseNameroofPerp = 'ROOFTOP_PERP_ZONES'
    outputBaseNameroofCorner = 'ROOFTOP_CORNER_ZONES'

    # Name of the output tables
    roofPerpZonesTable = DataUtil.prefix(outputBaseNameroofPerp, prefix=prefix)
    RoofCornerZonesTable = DataUtil.prefix(outputBaseNameroofCorner, prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    temporaryRooftopPerp = DataUtil.postfix('temporary_rooftop_perp')
    temporaryRooftopCorner = DataUtil.postfix('temporary_rooftop_corner')

    # Creates a dictionary of table names in order to simplify the final query
    dicTableNames = pd.DataFrame(
        {
            'final': [roofPerpZonesTable, RoofCornerZonesTable],
            'temporary': [temporaryRooftopPerp, temporaryRooftopCorner],
        },
        index=['perp', 'corner'],
    )

    # Piece of query to get Lcx and Lcy (based on equations 3, 4 and 5 from
    # Bagal et al. 2004 - note that in 4 and 5, we assumed that X and Y had been
    # reverted)
    pieceOfQueryLcCorner = (
        '2*ST_LENGTH({0})*TAN(2.94*EXP(0.0297*ABS(PI()/2-{1})))'.format(
            GEOM_FIELD, UPWIND_FACADE_ANGLE_FIELD
        )
    )

    # Queries to create temporary rooftop zones (perpendicular and corner)
    queryTempoRooftop = """
        DROP TABLE IF EXISTS {0}, {12};
        CREATE TABLE {0}
            AS SELECT   {1},
                        {2},
                        {4},
                        ABS({6}) AS {14},
                        ST_LENGTH({3}) AS {15},
                        {5},
                        CASE    WHEN {5} < PI()/2
                                THEN ST_MAKEPOLYGON(ST_MAKELINE(ST_ENDPOINT({3}),
                                                                ST_TRANSLATE(ST_STARTPOINT({3}),
                                                                             -{6}*SIN(PI()/2-{5}),
                                                                             {6}*COS(PI()/2-{5})),
                                                                ST_STARTPOINT({3}),
                                                                ST_ENDPOINT({3})))
                                ELSE ST_MAKEPOLYGON(ST_MAKELINE(ST_STARTPOINT({3}),
                                                                ST_ENDPOINT({3}),
                                                                ST_TRANSLATE(ST_ENDPOINT({3}),
                                                                             {6}*SIN({5}-PI()/2),
                                                                             {6}*COS({5}-PI()/2)),
                                                                ST_STARTPOINT({3})))
                                END AS {3}
            FROM {7}
            WHERE   {5} > RADIANS(90-{9}) AND {5} < RADIANS(90-{10})
                    OR {5} > RADIANS(90+{10}) AND {5} < RADIANS(90+{9});
        {16};
        {17};
        CREATE TABLE {12}
            AS SELECT   a.{1},
                        a.{2},
                        a.{4},
                        ST_MAKEPOLYGON(ST_MAKELINE(ST_STARTPOINT(a.{3}),
                                                   ST_TRANSLATE(ST_STARTPOINT(a.{3}),
                                                                0,
                                                                -b.{11}),
                                                   ST_TRANSLATE(ST_ENDPOINT(a.{3}),
                                                                0,
                                                                -b.{11}),
                                                   ST_ENDPOINT(a.{3}),
                                                   ST_STARTPOINT(a.{3}))) AS {3}
            FROM {7} AS a LEFT JOIN {8} AS b ON a.{1} = b.{1} 
            WHERE   a.{5} > RADIANS(90-{13}) AND a.{5} < RADIANS(90+{13})
           """.format(
        temporaryRooftopCorner,
        ID_FIELD_STACKED_BLOCK,
        UPWIND_FACADE_FIELD,
        GEOM_FIELD,
        HEIGHT_FIELD,
        UPWIND_FACADE_ANGLE_FIELD,
        pieceOfQueryLcCorner,
        upwindTable,
        zonePropertiesTable,
        CORNER_THRESHOLD_ANGLE[1],
        CORNER_THRESHOLD_ANGLE[0],
        ROOFTOP_PERP_LENGTH,
        temporaryRooftopPerp,
        PERPENDICULAR_THRESHOLD_ANGLE,
        ROOFTOP_CORNER_LENGTH,
        ROOFTOP_CORNER_FACADE_LENGTH,
        DataUtil.createIndex(
            tableName=upwindTable, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
        ),
        DataUtil.createIndex(
            tableName=zonePropertiesTable,
            fieldName=ID_FIELD_STACKED_BLOCK,
            isSpatial=False,
        ),
    )
    cursor.execute(queryTempoRooftop)

    # Queries to limit the rooftop zones to the rooftop of the stacked block...
    extraFieldToKeep = {
        'perp': 'b.{0}, b.{1},'.format(ROOFTOP_PERP_LENGTH, ROOFTOP_PERP_HEIGHT),
        'corner': 'a.{0}, a.{1}, a.{2}, b.{3},'.format(
            ROOFTOP_CORNER_LENGTH,
            ROOFTOP_CORNER_FACADE_LENGTH,
            UPWIND_FACADE_ANGLE_FIELD,
            ROOFTOP_WIND_FACTOR,
        ),
    }
    queryCutRooftop = [
        """
        {8};
        {9};
        {10};
        {11};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT   a.{1},
                        a.{2},
                        a.{4},
                        {7}
                        ST_INTERSECTION(a.{3}, b.{3}) AS {3}
            FROM {5} AS a LEFT JOIN {6} AS b ON a.{1} = b.{1}
            WHERE a.{3} && b.{3} AND ST_INTERSECTS(a.{3}, b.{3})
           """.format(
            dicTableNames.loc[typeZone, 'final'],
            ID_FIELD_STACKED_BLOCK,
            UPWIND_FACADE_FIELD,
            GEOM_FIELD,
            HEIGHT_FIELD,
            dicTableNames.loc[typeZone, 'temporary'],
            zonePropertiesTable,
            extraFieldToKeep[typeZone],
            DataUtil.createIndex(
                tableName=dicTableNames.loc[typeZone, 'temporary'],
                fieldName=GEOM_FIELD,
                isSpatial=True,
            ),
            DataUtil.createIndex(
                tableName=zonePropertiesTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=dicTableNames.loc[typeZone, 'temporary'],
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=zonePropertiesTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
        for typeZone in dicTableNames.index
    ]
    cursor.execute(';'.join(queryCutRooftop))

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join(dicTableNames['temporary'].values)
            )
        )

    return roofPerpZonesTable, RoofCornerZonesTable


def vegetationZones(cursor, vegetationTable, wakeZonesTable, prefix=PREFIX_NAME):
    """Identify vegetation zones which are in "built up" areas and those
    being in "open areas". Vegetation is considered in a built up area
    when it intersects with build wake zone.

    References:
            Nelson et al., Evaluation of an urban vegetative canopy scheme
        and impact on plume dispersion. 2009


                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            vegetationTable: String
                Name of the table containing vegetation footprints
            wakeZonesTable: String
                Name of the table containing the wake zones
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            vegetationBuiltZoneTable: String
                Name of the table containing the vegetation zone located in
                built-up areas
            vegetationOpenZoneTable: String
                Name of the table containing the vegetation zone located in
                open areas"""
    print('Creates built-up and open vegetation zones')

    # Output base name
    outputBaseNameOpen = 'OPEN_VEGETATION_ZONES'
    outputBaseNameBuilt = 'BUILTUP_VEGETATION_ZONES'

    # Name of the output tables
    vegetationOpenZoneTable = DataUtil.prefix(outputBaseNameOpen, prefix=prefix)
    vegetationBuiltZoneTable = DataUtil.prefix(outputBaseNameBuilt, prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    temporary_built_vegetation = DataUtil.postfix('temporary_built_vegetation')

    # Identify vegetation zones being in building wake zones
    cursor.execute(
        """
        {10};
        {11};
        DROP TABLE IF EXISTS {7};
        CREATE TABLE {7}
            AS SELECT   ST_INTERSECTION(a.{1}, b.{1}) AS {1},
                        a.{3},
                        a.{4},
                        a.{5},
                        a.{6}
            FROM {0} AS a, {2} AS b
            WHERE a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1});
        {12};
        DROP TABLE IF EXISTS {8};
        CREATE TABLE {8}({9} SERIAL     , {1} GEOMETRY   , {3} DOUBLE,
                         {4} DOUBLE     , {5} DOUBLE     , {6} INTEGER)
            AS SELECT NULL, {1}, {3}, {4}, {5}, {6}
            FROM ST_EXPLODE('(SELECT    ST_UNION(ST_ACCUM({1})) AS {1},
                                        MIN({3}) AS {3},
                                        MIN({4}) AS {4},
                                        MIN({5}) AS {5},
                                        {6}
                            FROM {7}
                            GROUP BY {6})')
        """.format(
            vegetationTable,
            GEOM_FIELD,
            wakeZonesTable,
            VEGETATION_CROWN_BASE_HEIGHT,
            VEGETATION_CROWN_TOP_HEIGHT,
            VEGETATION_ATTENUATION_FACTOR,
            ID_VEGETATION,
            temporary_built_vegetation,
            vegetationBuiltZoneTable,
            ID_ZONE_VEGETATION,
            DataUtil.createIndex(
                tableName=vegetationTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=wakeZonesTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=temporary_built_vegetation,
                fieldName=ID_VEGETATION,
                isSpatial=False,
            ),
        )
    )

    # Identify vegetation zones being in open areas
    cursor.execute(
        """
        {9};
        {10};
        DROP TABLE IF EXISTS {7};
        CREATE TABLE {7}({8} SERIAL     , {1} GEOMETRY   , {3} DOUBLE,
                         {4} DOUBLE     , {5} DOUBLE     , {6} INTEGER)
            AS SELECT   NULL, {1}, {3}, {4}, {5}, {6}
            FROM ST_EXPLODE('(SELECT    COALESCE(ST_DIFFERENCE(a.{1}, b.{1}),
                                                a.{1}) AS {1},
                                        a.{3},
                                        a.{4},
                                        a.{5},
                                        a.{6}
                            FROM {0} AS a LEFT JOIN {2} AS b ON a.{6} = b.{6}
                            WHERE NOT ST_ISEMPTY(COALESCE(ST_DIFFERENCE(a.{1}, b.{1}),
                                                          a.{1})))')
        """.format(
            vegetationTable,
            GEOM_FIELD,
            temporary_built_vegetation,
            VEGETATION_CROWN_BASE_HEIGHT,
            VEGETATION_CROWN_TOP_HEIGHT,
            VEGETATION_ATTENUATION_FACTOR,
            ID_VEGETATION,
            vegetationOpenZoneTable,
            ID_ZONE_VEGETATION,
            DataUtil.createIndex(
                tableName=vegetationTable, fieldName=ID_VEGETATION, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=temporary_built_vegetation,
                fieldName=ID_VEGETATION,
                isSpatial=False,
            ),
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(','.join([temporary_built_vegetation]))
        )

    return vegetationBuiltZoneTable, vegetationOpenZoneTable


def identifyImpactingStackedBlocks(
    cursor,
    dicOfBuildRockleZoneTable,
    dicOfVegRockleZoneTable,
    impactedZone,
    stackedBlocksTable,
    vegetationTable,
    crossWindExtend,
    prefix,
):
    """Identify all stacked blocks potentially impacting a given impacted zone.
    This is done in 3 steps
            STEP 1. Whenever any Röckle zone of a stacked block SB1 intersects the impacted zone,
    all the stacked blocks belonging to the same block as SB1 are identified.
            STEP 2. We also identify the blocks which does not intersect the impacted zone
    but which are at the same upstream or downstream position as the blocks
    impacting the impacted zone. In the cross-wind direction, we stop searching
    for blocks when we get further than the most extreme cross-wind
    positions of both the previous identified blocks and the impacted zone
    plus a given extend 'crossWindExtend' in the cross-wind direction.
            STEP 3. For each block identified in 1 and 2, we select the
    stacked blocks and the corresponding Röckle zones.

                Parameters
                _ _ _ _ _ _ _ _ _ _

            cursor: conn.cursor
                A cursor object, used to perform spatial SQL queries
            dicOfBuildRockleZoneTable: Dictionary
                Dictionary containing as key the building Rockle
                zone name and as value the corresponding table name
            dicOfVegRockleZoneTable: Dictionary
                Dictionary containing as key the vegetation Rockle
                zone name and as value the corresponding table name
            impactedZone: String
                Name of the table where is saved the study area table
                (should contain only one polygon)
            stackedBlocksTable: String
                Name of the table containing all stacked blocks
            vegetationTable: String
                Name of the table containing all vegetation patches
            crossWindExtend: float
                Distance (in meter) of the extend of the zone around the
                rotated obstacles and the impactedZone in the cross-wind direction
            prefix: String, default PREFIX_NAME
                Prefix to add to the output table name

                Returns
                _ _ _ _ _ _ _ _ _ _

            dicOfSelectedBuiltZones: Dictionary of Rockle zone tables
                Dictionary containing as key the building or vegetation Rockle
                zone name and as value the corresponding table name
            stackedBlocksTableSelection: String
                Name of the table used to save selected stacked blocks

    """
    print('Identify the buildings concerned by the impacted zone chosen by the user')

    # Name of the output tables
    dicOfSelectedBuildZones = {
        t: dicOfBuildRockleZoneTable[t] + SELECTED_SUFFIX
        for t in dicOfBuildRockleZoneTable.keys()
    }
    dicOfSelectedVegZones = {
        t: dicOfVegRockleZoneTable[t] + SELECTED_SUFFIX
        for t in dicOfVegRockleZoneTable.keys()
    }
    outputStackedBlocks = DataUtil.prefix('IMPACTING_STACKED_BLOCKS', prefix=prefix)
    outputVegetation = DataUtil.prefix('IMPACTING_VEGETATION', prefix=prefix)

    # Create temporary table names (for tables that will be removed at the end of the IProcess)
    tabAllBuildZones = DataUtil.postfix('tab_all_build_zones')
    tabTempStack = DataUtil.postfix('tab_temp_stack')
    tabTempBlock = DataUtil.postfix('tab_temp_blocks')
    tabCrossExtBox = DataUtil.postfix('tab_cross_extend_box')
    tabTempBlock2 = DataUtil.postfix('tab_temp_blocks2')

    # ------------------------------------------------------------------------
    # 1. MAKE THE CALCULATION FOR THE BUILDINGS ------------------------------
    # ------------------------------------------------------------------------
    # Gather all building Röckle zones in one
    gatherBuQuery = [
        """SELECT {0}, {1} FROM {2}""".format(
            ID_FIELD_STACKED_BLOCK, GEOM_FIELD, dicOfBuildRockleZoneTable[t]
        )
        for t in dicOfBuildRockleZoneTable.keys()
    ]
    cursor.execute(
        """
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS {1}
        """.format(tabAllBuildZones, ' UNION ALL '.join(gatherBuQuery))
    )

    # Identify which stacked blocks intersect the Röckle zones
    cursor.execute(
        """
        {0};{1};
        DROP TABLE IF EXISTS {2};
        CREATE TABLE {2}
            AS SELECT DISTINCT(a.{3}) AS {3}
            FROM {4} AS a, {5} AS b
            WHERE a.{6} && b.{6} AND ST_INTERSECTS(a.{6}, b.{6})
        """.format(
            DataUtil.createIndex(
                tableName=tabAllBuildZones, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=impactedZone, fieldName=GEOM_FIELD, isSpatial=True
            ),
            tabTempStack,
            ID_FIELD_STACKED_BLOCK,
            tabAllBuildZones,
            impactedZone,
            GEOM_FIELD,
        )
    )

    # Identify to which blocks belong the identified stacked blocks
    cursor.execute(
        """
        {0};{1};
        DROP TABLE IF EXISTS {2};
        CREATE TABLE {2}
            AS SELECT DISTINCT(b.{3}) AS {3}, b.{7}
            FROM {4} AS a RIGHT JOIN {5} AS b
            ON a.{6} = b.{6}
        """.format(
            DataUtil.createIndex(
                tableName=tabTempStack,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            DataUtil.createIndex(
                tableName=stackedBlocksTable,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            tabTempBlock,
            ID_FIELD_BLOCK,
            tabTempStack,
            stackedBlocksTable,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
        )
    )

    # Identify which blocks are in the cross-wind extend of blocks and impacted zone
    cursor.execute(
        """
        DROP TABLE IF EXISTS {0};
        CREATE TABLE {0}
            AS SELECT ST_EXPAND(ST_EXTENT(a.{1}), {2}, 0) AS {1}
            FROM (SELECT {1} FROM {4} 
                  UNION ALL
                  SELECT {1} FROM {5}) AS a;
        {6};{7};
        DROP TABLE IF EXISTS {8};
        CREATE TABLE {8}
            AS SELECT DISTINCT({9}) AS {9}
            FROM (SELECT a.{9}
                    FROM {10} AS a, {0} AS b
                    WHERE a.{1} && b.{1} AND ST_INTERSECTS(a.{1}, b.{1})
                    UNION ALL
                    SELECT {9} 
                    FROM {11})
        """.format(
            tabCrossExtBox,
            GEOM_FIELD,
            crossWindExtend,
            stackedBlocksTable,
            tabTempBlock,
            impactedZone,
            DataUtil.createIndex(
                tableName=tabCrossExtBox, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=stackedBlocksTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            tabTempBlock2,
            ID_FIELD_BLOCK,
            stackedBlocksTable,
            tabTempBlock,
        )
    )

    # Identify all stacked blocks belonging to the previously identified blocks
    cursor.execute(
        """
        {0};{1};
        DROP TABLE IF EXISTS {2};
        CREATE TABLE {2}
            AS SELECT a.*
            FROM {4} AS a RIGHT JOIN {5} AS b
            ON a.{6} = b.{6}
        """.format(
            DataUtil.createIndex(
                tableName=tabTempBlock2, fieldName=ID_FIELD_BLOCK, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=stackedBlocksTable, fieldName=ID_FIELD_BLOCK, isSpatial=False
            ),
            outputStackedBlocks,
            ID_FIELD_STACKED_BLOCK,
            stackedBlocksTable,
            tabTempBlock2,
            ID_FIELD_BLOCK,
        )
    )

    # Select the Rôckle zones corresponding to the stacked blocks selection
    selectionBuildQueries = [
        """
        {4};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE    {0}
            AS SELECT   a.*
            FROM        {1} AS a RIGHT JOIN {2} AS b
                        ON a.{3} = b.{3}
        """.format(
            dicOfSelectedBuildZones[t],
            dicOfBuildRockleZoneTable[t],
            outputStackedBlocks,
            ID_FIELD_STACKED_BLOCK,
            DataUtil.createIndex(
                tableName=dicOfBuildRockleZoneTable[t],
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
        )
        for t in dicOfBuildRockleZoneTable
    ]
    cursor.execute(
        """{0}; {1}
                   """.format(
            DataUtil.createIndex(
                tableName=outputStackedBlocks,
                fieldName=ID_FIELD_STACKED_BLOCK,
                isSpatial=False,
            ),
            ';'.join(selectionBuildQueries),
        )
    )

    # ------------------------------------------------------------------------
    # 2. MAKE THE CALCULATION FOR THE VEGETATION -----------------------------
    # ------------------------------------------------------------------------
    # Identify which vegetation patches are in the cross-wind extend of blocks and impacted zone
    cursor.execute(
        """
        {0};{1};
        DROP TABLE IF EXISTS {2};
        CREATE TABLE {2}
            AS SELECT a.*
            FROM {3} AS a, {4} AS b
            WHERE a.{5} && b.{5} AND ST_INTERSECTS(a.{5}, b.{5})
        """.format(
            DataUtil.createIndex(
                tableName=tabCrossExtBox, fieldName=GEOM_FIELD, isSpatial=True
            ),
            DataUtil.createIndex(
                tableName=vegetationTable, fieldName=GEOM_FIELD, isSpatial=True
            ),
            outputVegetation,
            vegetationTable,
            tabCrossExtBox,
            GEOM_FIELD,
        )
    )

    # Select the Rôckle zones corresponding to the vegetation patches selection
    selectionVegQueries = [
        """
        {4};
        DROP TABLE IF EXISTS {0};
        CREATE TABLE    {0}
            AS SELECT   a.*
            FROM        {1} AS a RIGHT JOIN {2} AS b
                        ON a.{3} = b.{3}
        """.format(
            dicOfSelectedVegZones[t],
            dicOfVegRockleZoneTable[t],
            outputVegetation,
            ID_VEGETATION,
            DataUtil.createIndex(
                tableName=dicOfVegRockleZoneTable[t],
                fieldName=ID_VEGETATION,
                isSpatial=False,
            ),
        )
        for t in dicOfVegRockleZoneTable
    ]
    cursor.execute(
        """{0}; {1}
                   """.format(
            DataUtil.createIndex(
                tableName=outputVegetation, fieldName=ID_VEGETATION, isSpatial=False
            ),
            ';'.join(selectionVegQueries),
        )
    )

    if not DEBUG:
        # Drop intermediate tables
        cursor.execute(
            'DROP TABLE IF EXISTS {0}'.format(
                ','.join([tabTempStack, tabTempBlock, tabCrossExtBox, tabTempBlock2])
            )
        )

    return (
        dicOfSelectedBuildZones,
        dicOfSelectedVegZones,
        outputStackedBlocks,
        outputVegetation,
    )
