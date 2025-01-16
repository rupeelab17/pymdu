#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:39:39 2021

@author: Jérémy Bernard, University of Gothenburg
"""

import math
import time
from pathlib import Path

import numpy as np
from numba import jit

try:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.GlobalVariables as GlobalVariables
    import pymdu.physics.aeraulic.uRockle.H2gisConnection as H2gisConnection
    import pymdu.physics.aeraulic.uRockle.loadData as loadData
    import pymdu.physics.aeraulic.uRockle.saveData as saveData
    import pymdu.physics.aeraulic.uRockle.Obstacles as Obstacles
    import pymdu.physics.aeraulic.uRockle.Zones as Zones
    import pymdu.physics.aeraulic.uRockle.CalculatesIndicators as CalculatesIndicators
    import pymdu.physics.aeraulic.uRockle.InitWindField as InitWindField
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil
    import pymdu.physics.aeraulic.uRockle.WindSolver as WindSolver
except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    from pymdu.physics.aeraulic.uRockle import H2gisConnection
    from pymdu.physics.aeraulic.uRockle import loadData
    from pymdu.physics.aeraulic.uRockle import saveData
    from pymdu.physics.aeraulic.uRockle import Obstacles
    from pymdu.physics.aeraulic.uRockle import Zones
    from pymdu.physics.aeraulic.uRockle import CalculatesIndicators
    from pymdu.physics.aeraulic.uRockle import InitWindField
    from pymdu.physics.aeraulic.uRockle import DataUtil
    from pymdu.physics.aeraulic.uRockle import WindSolver


def main(
    pluginDirectory,
    outputFilePath,
    buildingFilePath,
    srid,
    outputFilename=OUTPUT_FILENAME,
    vegetationFilePath='',
    z_ref=Z_REF,
    v_ref=V_REF,
    windDirection=WIND_DIRECTION,
    prefix=PREFIX_NAME,
    meshSize=MESH_SIZE,
    dz=DZ,
    alongWindZoneExtend=ALONG_WIND_ZONE_EXTEND,
    crossWindZoneExtend=CROSS_WIND_ZONE_EXTEND,
    verticalExtend=VERTICAL_EXTEND,
    tempoDirectory=TEMPO_DIRECTORY,
    onlyInitialization=ONLY_INITIALIZATION,
    maxIterations=MAX_ITERATIONS,
    thresholdIterations=THRESHOLD_ITERATIONS,
    idFieldBuild=ID_FIELD_BUILD,
    buildingHeightField=HEIGHT_FIELD,
    vegetationBaseHeight=VEGETATION_CROWN_BASE_HEIGHT,
    vegetationTopHeight=VEGETATION_CROWN_TOP_HEIGHT,
    idVegetation=ID_VEGETATION,
    vegetationAttenuationFactor=VEGETATION_ATTENUATION_FACTOR,
    saveRockleZones=SAVE_ROCKLE_ZONES,
    z_out=Z_OUT,
    outputRaster=None,
    feedback=None,
    saveRaster=True,
    saveVector=False,
    saveNetcdf=False,
    debug=DEBUG,
    profileType=PROFILE_TYPE,
    verticalProfileFile=None,
):
    # If the function is called within QGIS, a feedback is sent into the QGIS interface
    if feedback:
        feedback.setProgressText('Initiating algorithm')

    ################################ INIT OUTPUT VARIABLES ############################
    # Define dictionaries of input and output relative directories
    outputDataRel = {}

    # Blocks and stacked blocks
    outputDataRel['blocks'] = os.path.join(tempoDirectory, 'blocks.geojson')
    outputDataRel['stacked_blocks'] = os.path.join(
        tempoDirectory, 'stackedBlocks.geojson'
    )
    outputDataRel['vegetation'] = os.path.join(tempoDirectory, 'vegetation.geojson')

    # Rotated geometries
    outputDataRel['rotated_stacked_blocks'] = os.path.join(
        tempoDirectory, 'rotated_stacked_blocks.geojson'
    )
    outputDataRel['rotated_vegetation'] = os.path.join(
        tempoDirectory, 'vegetationRotated.geojson'
    )
    outputDataRel['upwind_facades'] = os.path.join(
        tempoDirectory, 'upwind_facades.geojson'
    )
    outputDataRel['downwind_facades'] = os.path.join(
        tempoDirectory, 'downwind_facades.geojson'
    )

    # Created zones
    outputDataRel['displacement'] = os.path.join(
        tempoDirectory, 'displacementZones.geojson'
    )
    outputDataRel['displacement_vortex'] = os.path.join(
        tempoDirectory, 'displacementVortexZones.geojson'
    )
    outputDataRel['cavity'] = os.path.join(tempoDirectory, 'cavity.geojson')
    outputDataRel['wake'] = os.path.join(tempoDirectory, 'wake.geojson')
    outputDataRel['street_canyon'] = os.path.join(
        tempoDirectory, 'streetCanyon.geojson'
    )
    outputDataRel['rooftop_perpendicular'] = os.path.join(
        tempoDirectory, 'rooftopPerp.geojson'
    )
    outputDataRel['rooftop_corner'] = os.path.join(
        tempoDirectory, 'rooftopCorner.geojson'
    )
    outputDataRel['vegetation_built'] = os.path.join(
        tempoDirectory, 'vegetationBuilt.geojson'
    )
    outputDataRel['vegetation_open'] = os.path.join(
        tempoDirectory, 'vegetationOpen.geojson'
    )

    # Grid points
    outputDataRel['point_BuildZone'] = os.path.join(tempoDirectory, 'point_BuildZone')
    outputDataRel['point3D_BuildZone'] = os.path.join(
        tempoDirectory, 'point3D_BuildZone'
    )
    outputDataRel['point_VegZone'] = os.path.join(tempoDirectory, 'point_VegZone')
    outputDataRel['point3D_VegZone'] = os.path.join(tempoDirectory, 'point3D_VegZone')
    outputDataRel['point3D_All'] = os.path.join(tempoDirectory, 'point3D_All')

    # Convert relative to absolute paths
    outputDataAbs = {i: os.path.abspath(outputDataRel[i]) for i in outputDataRel}

    ############################################################################
    ################################ SCRIPT ####################################
    ############################################################################
    # ----------------------------------------------------------------------
    # 1. SET H2GIS DATABASE ENVIRONMENT AND LOAD DATA
    # ----------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Creates an H2GIS Instance and load data')
    # Download H2GIS
    resource_path = Path(__file__).parent
    path_db = str(resource_path)

    H2gisConnection.downloadH2gis(dbDirectory=path_db)
    # Initialize a H2GIS database connection
    cursor = H2gisConnection.startH2gisInstance(
        dbDirectory=path_db,
        dbInstanceDir=tempoDirectory,
        instanceName=prefix + INSTANCE_NAME,
    )

    # Load data
    loadData.loadData(
        fromCad=False,
        prefix=prefix,
        idFieldBuild=idFieldBuild,
        buildingHeightField=buildingHeightField,
        vegetationBaseHeight=vegetationBaseHeight,
        vegetationTopHeight=vegetationTopHeight,
        idVegetation=idVegetation,
        vegetationAttenuationFactor=vegetationAttenuationFactor,
        cursor=cursor,
        buildingFilePath=buildingFilePath,
        vegetationFilePath=vegetationFilePath,
        srid=srid,
    )

    timeStartCalculation = time.time()

    # -----------------------------------------------------------------------------------
    # 2. CREATES OBSTACLE GEOMETRIES ----------------------------------------------------
    # -----------------------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Creates the stacked blocks used as obstacles')
    # Create the stacked blocks
    blockTable, stackedBlockTable = Obstacles.createsBlocks(
        cursor=cursor, inputBuildings=BUILDING_TABLE_NAME, prefix=prefix
    )

    # Save the blocks, stacked blocks and vegetation as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=blockTable,
            filedir=outputDataAbs['blocks'],
            delete=True,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=VEGETATION_TABLE_NAME,
            filedir=outputDataAbs['vegetation'],
            delete=True,
        )

    # -----------------------------------------------------------------------------------
    # 3. ROTATES OBSTACLES TO THE RIGHT DIRECTION AND CALCULATES GEOMETRY PROPERTIES ----
    # -----------------------------------------------------------------------------------
    if feedback:
        feedback.setProgressText(
            'Rotates obstacles to the right direction and calculates geometry properties'
        )
    # Define a set of obstacles in a dictionary before the rotation
    dicOfObstacles = {
        BUILDING_TABLE_NAME: stackedBlockTable,
        VEGETATION_TABLE_NAME: VEGETATION_TABLE_NAME,
    }

    # Rotate obstacles
    dicRotatedTables, rotationCenterCoordinates = Obstacles.windRotation(
        cursor=cursor,
        dicOfInputTables=dicOfObstacles,
        rotateAngle=windDirection,
        rotationCenterCoordinates=None,
        prefix=prefix,
    )

    # Get the rotated block and vegetation table names
    rotatedStackedBlocks = dicRotatedTables[BUILDING_TABLE_NAME]
    rotatedVegetation = dicRotatedTables[VEGETATION_TABLE_NAME]

    # Calculates base block height and base of block cavity zone
    rotatedPropStackedBlocks = Obstacles.identifyBlockAndCavityBase(
        cursor, rotatedStackedBlocks, prefix=prefix
    )

    # Init the upwind facades
    upwindInitedTable = Obstacles.initUpwindFacades(
        cursor=cursor, obstaclesTable=rotatedPropStackedBlocks, prefix=prefix
    )
    # Update base height of upwind facades (if shared with the building below)
    upwindTable = Obstacles.updateUpwindFacadeBase(
        cursor=cursor, upwindTable=upwindInitedTable, prefix=prefix
    )

    # Calculates obstacles properties
    obstaclePropertiesTable = CalculatesIndicators.obstacleProperties(
        cursor=cursor, obstaclesTable=rotatedPropStackedBlocks, prefix=prefix
    )

    # Calculates obstacle zone properties
    zonePropertiesTable = CalculatesIndicators.zoneProperties(
        cursor=cursor, obstaclePropertiesTable=obstaclePropertiesTable, prefix=prefix
    )

    # Calculates roughness properties of the study area
    z0, d, Hr, H_ob_max, lambda_f = CalculatesIndicators.studyAreaProperties(
        cursor=cursor,
        upwindTable=upwindInitedTable,
        stackedBlockTable=rotatedStackedBlocks,
        vegetationTable=rotatedVegetation,
    )

    # Calculates downwind facades
    downwindTable = Obstacles.initDownwindFacades(
        cursor=cursor, obstaclesTable=zonePropertiesTable, prefix=prefix
    )

    # Save the rotated obstacles and facades as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=rotatedPropStackedBlocks,
            filedir=outputDataAbs['rotated_stacked_blocks'],
            delete=True,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=rotatedVegetation,
            filedir=outputDataAbs['rotated_vegetation'],
            delete=True,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=upwindTable,
            filedir=outputDataAbs['upwind_facades'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=downwindTable,
            filedir=outputDataAbs['downwind_facades'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=rotatedPropStackedBlocks,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
            filedir=outputDataAbs['stacked_blocks'],
            delete=True,
        )

    # -----------------------------------------------------------------------------------
    # 4. CREATES THE 2D ROCKLE ZONES ----------------------------------------------------
    # -----------------------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Creates the 2D Röckle zones')
    # Creates the displacement zone (upwind)
    displacementZonesTable, displacementVortexZonesTable = Zones.displacementZones(
        cursor=cursor,
        upwindTable=upwindTable,
        zonePropertiesTable=zonePropertiesTable,
        srid=srid,
        prefix=prefix,
    )

    # Save the resulting displacement zones as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=displacementZonesTable,
            filedir=outputDataAbs['displacement'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=displacementVortexZonesTable,
            filedir=outputDataAbs['displacement_vortex'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # Creates the cavity and wake zones
    cavityZonesTable, wakeZonesTable = Zones.cavityAndWakeZones(
        cursor=cursor,
        downwindWithPropTable=downwindTable,
        srid=srid,
        ellipseResolution=meshSize / 3,
        prefix=prefix,
    ).values()

    # Save the resulting displacement zones as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=cavityZonesTable,
            filedir=outputDataAbs['cavity'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=wakeZonesTable,
            filedir=outputDataAbs['wake'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # Creates the street canyon zones
    streetCanyonTable = Zones.streetCanyonZones(
        cursor=cursor,
        cavityZonesTable=cavityZonesTable,
        zonePropertiesTable=zonePropertiesTable,
        upwindTable=upwindTable,
        downwindTable=downwindTable,
        srid=srid,
        prefix=prefix,
    )

    # Save the resulting street canyon zones as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=streetCanyonTable,
            filedir=outputDataAbs['street_canyon'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # Creates the rooftop zones
    rooftopPerpendicularZoneTable, rooftopCornerZoneTable = Zones.rooftopZones(
        cursor=cursor,
        upwindTable=upwindTable,
        zonePropertiesTable=zonePropertiesTable,
        prefix=prefix,
    )
    # Save the resulting rooftop zones as geojson
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=rooftopPerpendicularZoneTable,
            filedir=outputDataAbs['rooftop_perpendicular'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=rooftopCornerZoneTable,
            filedir=outputDataAbs['rooftop_corner'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # Creates the vegetation zones
    vegetationBuiltZoneTable, vegetationOpenZoneTable = Zones.vegetationZones(
        cursor=cursor,
        vegetationTable=rotatedVegetation,
        wakeZonesTable=wakeZonesTable,
        prefix=prefix,
    )
    if debug or saveRockleZones:
        saveData.saveTable(
            cursor=cursor,
            tableName=vegetationBuiltZoneTable,
            filedir=outputDataAbs['vegetation_built'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )
        saveData.saveTable(
            cursor=cursor,
            tableName=vegetationOpenZoneTable,
            filedir=outputDataAbs['vegetation_open'],
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # Define a dictionary of all building Rockle zones and same for veg
    dicOfBuildRockleZoneTable = {
        DISPLACEMENT_NAME: displacementZonesTable,
        DISPLACEMENT_VORTEX_NAME: displacementVortexZonesTable,
        CAVITY_NAME: cavityZonesTable,
        WAKE_NAME: wakeZonesTable,
        STREET_CANYON_NAME: streetCanyonTable,
        ROOFTOP_PERP_NAME: rooftopPerpendicularZoneTable,
        ROOFTOP_CORN_NAME: rooftopCornerZoneTable,
    }
    dicOfVegRockleZoneTable = {
        VEGETATION_BUILT_NAME: vegetationBuiltZoneTable,
        VEGETATION_OPEN_NAME: vegetationOpenZoneTable,
    }

    if outputRaster:
        # Creates a table with a polygon covering the raster zone envelope
        smallStudyZone = 'SMALL_STUDY_ZONE'
        outputRasterExtent = outputRaster.extent()
        cursor.execute(
            """
           DROP TABLE IF EXISTS {0};
           CREATE TABLE {0}({5} GEOMETRY)
               AS SELECT ST_SETSRID(ST_ROTATE(ST_ENVELOPE('MULTIPOINT({1} {2},
                                               {3} {4})'),
                                              {6},
                                              {7},
                                              {8}), {9})
           """.format(
                smallStudyZone,
                outputRasterExtent.xMinimum(),
                outputRasterExtent.yMinimum(),
                outputRasterExtent.xMaximum(),
                outputRasterExtent.yMaximum(),
                GEOM_FIELD,
                DataUtil.degToRad(windDirection),
                rotationCenterCoordinates[0],
                rotationCenterCoordinates[1],
                srid,
            )
        )
        # Identify the stacked blocks, blocks potentially impacting the
        # impacted zone and their corresponding Röckle zones
        (
            dicOfBuildRockleZoneTable,
            dicOfVegRockleZoneTable,
            rotatedPropStackedBlocks,
            rotatedVegetation,
        ) = Zones.identifyImpactingStackedBlocks(
            cursor=cursor,
            dicOfBuildRockleZoneTable=dicOfBuildRockleZoneTable,
            dicOfVegRockleZoneTable=dicOfVegRockleZoneTable,
            impactedZone=smallStudyZone,
            stackedBlocksTable=rotatedPropStackedBlocks,
            vegetationTable=rotatedVegetation,
            crossWindExtend=crossWindZoneExtend,
            prefix=prefix,
        )
    # ----------------------------------------------------------------------
    # 5. SET THE 2D GRID IN THE ROCKLE ZONES -------------------------------
    # ----------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Creates the 2D grid')
    # Creates the grid of points
    gridPoint = InitWindField.createGrid(
        cursor=cursor,
        dicOfInputTables=dict(dicOfBuildRockleZoneTable, **dicOfVegRockleZoneTable),
        srid=srid,
        alongWindZoneExtend=alongWindZoneExtend,
        crossWindZoneExtend=crossWindZoneExtend,
        meshSize=meshSize,
        prefix=prefix,
    )

    # Affects each 2D point to a build Rockle zone and calculates needed variables for 3D wind speed factors
    dicOfInitBuildZoneGridPoint, verticalLineTable = (
        InitWindField.affectsPointToBuildZone(
            cursor=cursor,
            gridTable=gridPoint,
            dicOfBuildRockleZoneTable=dicOfBuildRockleZoneTable,
            prefix=prefix,
        )
    )

    # Same for vegetation Röckle zones
    dicOfVegZoneGridPoint = InitWindField.affectsPointToVegZone(
        cursor=cursor,
        gridTable=gridPoint,
        dicOfVegRockleZoneTable=dicOfVegRockleZoneTable,
        prefix=prefix,
    )

    # Remove some of the Röckle points where building Röckle zones overlap
    dicOfBuildZoneGridPoint = InitWindField.removeBuildZonePoints(
        cursor=cursor,
        dicOfInitBuildZoneGridPoint=dicOfInitBuildZoneGridPoint,
        prefix=prefix,
    )

    # Manage backward cavity and wake zones in the leeward zone of tall buildings
    dicOfBuildZoneGridPoint, facadeWithinCavity = InitWindField.manageBackwardZones(
        cursor=cursor,
        dicOfBuildZoneGridPoint=dicOfBuildZoneGridPoint,
        cavity2dInitPoints=dicOfInitBuildZoneGridPoint[CAVITY_NAME],
        wake2dInitPoints=dicOfInitBuildZoneGridPoint[WAKE_NAME],
        streetCanyonTable=streetCanyonTable,
        gridTable=gridPoint,
        meshSize=meshSize,
        dz=dz,
        prefix=prefix,
    )

    if debug or saveRockleZones:
        for t in dicOfBuildZoneGridPoint:
            cursor.execute(
                """
               DROP TABLE IF EXISTS point_Buildzone_{0};
               {5};
               {6};
               CREATE TABLE point_Buildzone_{0}
                   AS SELECT   a.{2}, b.*
                   FROM {3} AS a RIGHT JOIN {4} AS b
                       ON a.{1} = b.{1}
                   WHERE b.{1} IS NOT NULL
               """.format(
                    t,
                    ID_POINT,
                    GEOM_FIELD,
                    gridPoint,
                    dicOfBuildZoneGridPoint[t],
                    DataUtil.createIndex(
                        tableName=gridPoint, fieldName=ID_POINT, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfBuildZoneGridPoint[t],
                        fieldName=ID_POINT,
                        isSpatial=False,
                    ),
                )
            )
            saveData.saveTable(
                cursor=cursor,
                tableName='point_Buildzone_' + t,
                filedir=outputDataAbs['point_BuildZone'] + t + '.geojson',
                delete=True,
                rotationCenterCoordinates=rotationCenterCoordinates,
                rotateAngle=-windDirection,
            )

    # -----------------------------------------------------------------------------------
    # 6. INITIALIZE THE 3D WIND FACTORS IN THE ROCKLE ZONES -------------------------------
    # -----------------------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Initializes the 3D grid within Röckle zones')
    # Calculates the 3D wind speed factors for each building Röckle zone
    dicOfBuildZone3DWindFactor, maxBuildZoneHeight = (
        InitWindField.calculates3dBuildWindFactor(
            cursor=cursor,
            dicOfBuildZoneGridPoint=dicOfBuildZoneGridPoint,
            dz=dz,
            prefix=prefix,
        )
    )
    if debug or saveRockleZones:
        for t in dicOfBuildZone3DWindFactor:
            cursor.execute(
                """
               DROP TABLE IF EXISTS point3D_Buildzone_{0};
               {5};
               {6};
               CREATE TABLE point3D_Buildzone_{0}
                   AS SELECT   a.{2}, b.*
                   FROM {3} AS a RIGHT JOIN {4} AS b
                       ON a.{1} = b.{1}
                   WHERE b.{1} IS NOT NULL
               """.format(
                    t,
                    ID_POINT,
                    GEOM_FIELD,
                    gridPoint,
                    dicOfBuildZone3DWindFactor[t],
                    DataUtil.createIndex(
                        tableName=gridPoint, fieldName=ID_POINT, isSpatial=False
                    ),
                    DataUtil.createIndex(
                        tableName=dicOfBuildZone3DWindFactor[t],
                        fieldName=ID_POINT,
                        isSpatial=False,
                    ),
                )
            )
            saveData.saveTable(
                cursor=cursor,
                tableName='point3D_Buildzone_' + t,
                filedir=outputDataAbs['point3D_BuildZone'] + t + '.geojson',
                delete=True,
                rotationCenterCoordinates=rotationCenterCoordinates,
                rotateAngle=-windDirection,
            )

    # Calculates the 3D wind speed factors of the vegetation (considering all zone types)
    # after calculation of the top of the "sketch"
    maxHeight = H_ob_max
    if maxBuildZoneHeight:
        if maxBuildZoneHeight > H_ob_max:
            maxHeight = maxBuildZoneHeight
    sketchHeight = maxHeight + verticalExtend
    vegetationWeightFactorTable = InitWindField.calculates3dVegWindFactor(
        cursor=cursor,
        dicOfVegZoneGridPoint=dicOfVegZoneGridPoint,
        sketchHeight=sketchHeight,
        z0=z0,
        d=d,
        dz=dz,
        prefix=prefix,
    )
    if debug or saveRockleZones:
        cursor.execute(
            """
           DROP TABLE IF EXISTS point3D_AllVegZone;
           {4};
           {5};
           CREATE TABLE point3D_AllVegZone
               AS SELECT   a.{1}, b.*
               FROM {2} AS a RIGHT JOIN {3} AS b
                           ON a.{0} = b.{0}
               WHERE b.{0} IS NOT NULL
           """.format(
                ID_POINT,
                GEOM_FIELD,
                gridPoint,
                vegetationWeightFactorTable,
                DataUtil.createIndex(
                    tableName=gridPoint, fieldName=ID_POINT, isSpatial=False
                ),
                DataUtil.createIndex(
                    tableName=vegetationWeightFactorTable,
                    fieldName=ID_POINT,
                    isSpatial=False,
                ),
            )
        )
        saveData.saveTable(
            cursor=cursor,
            tableName='point3D_AllVegZone',
            filedir=outputDataAbs['point3D_VegZone'] + '.geojson',
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

    # ----------------------------------------------------------------
    # 7. DEALS WITH SUPERIMPOSED ZONES -------------------------------
    # ----------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Deals with zones superimposition')
    # Calculates the final weighting factor for each point, dealing with duplicates (superimposition)
    dicAllWeightFactorsTables = dicOfBuildZone3DWindFactor.copy()
    dicAllWeightFactorsTables[ALL_VEGETATION_NAME] = vegetationWeightFactorTable
    allZonesPointFactor = InitWindField.manageSuperimposition(
        cursor=cursor,
        dicAllWeightFactorsTables=dicAllWeightFactorsTables,
        facadeWithinCavity=facadeWithinCavity,
        upstreamPriorityTables=UPSTREAM_PRIORITY_TABLES,
        upstreamWeightingTables=UPSTREAM_WEIGHTING_TABLES,
        upstreamWeightingInterRules=UPSTREAM_WEIGHTING_INTER_RULES,
        upstreamWeightingIntraRules=UPSTREAM_WEIGHTING_INTRA_RULES,
        downstreamWeightingTable=DOWNSTREAM_WEIGTHING_TABLE,
        prefix=prefix,
    )
    if debug or saveRockleZones:
        cursor.execute(
            """
            DROP TABLE IF EXISTS point3D_All;
            {4};
            {5};
            CREATE TABLE point3D_All
                AS SELECT   a.{1}, b.*
                FROM {2} AS a RIGHT JOIN {3} AS b
                            ON a.{0} = b.{0}
                WHERE b.{0} IS NOT NULL
            """.format(
                ID_POINT,
                GEOM_FIELD,
                gridPoint,
                allZonesPointFactor,
                DataUtil.createIndex(
                    tableName=gridPoint, fieldName=ID_POINT, isSpatial=False
                ),
                DataUtil.createIndex(
                    tableName=allZonesPointFactor, fieldName=ID_POINT, isSpatial=False
                ),
            )
        )
        saveData.saveTable(
            cursor=cursor,
            tableName='point3D_All',
            filedir=outputDataAbs['point3D_All'] + '.geojson',
            delete=True,
            rotationCenterCoordinates=rotationCenterCoordinates,
            rotateAngle=-windDirection,
        )

        # -------------------------------------------------------------------
    # 8. 3D WIND SPEED INITIALIZATION -----------------------------------
    # -------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Initialize the 3D wind in the grid')
    # Identify 3D grid points intersected by buildings
    df_gridBuil = InitWindField.identifyBuildPoints(
        cursor=cursor,
        gridPoint=gridPoint,
        stackedBlocksWithBaseHeight=rotatedPropStackedBlocks,
        dz=dz,
        tempoDirectory=tempoDirectory,
    )

    # Set the initial 3D wind speed field
    df_wind0, nPoints, verticalWindProfile = InitWindField.setInitialWindField(
        cursor=cursor,
        initializedWindFactorTable=allZonesPointFactor,
        gridPoint=gridPoint,
        df_gridBuil=df_gridBuil,
        z0=z0,
        sketchHeight=sketchHeight,
        profileType=profileType,
        meshSize=meshSize,
        dz=dz,
        z_ref=z_ref,
        v_ref=v_ref,
        tempoDirectory=tempoDirectory,
        d=d,
        H=Hr,
        lambda_f=lambda_f,
        verticalProfileFile=verticalProfileFile,
    )

    # -------------------------------------------------------------------
    # 9. "RASTERIZE" THE DATA - PREPARE MATRICES FOR WIND CALCULATION ---
    # -------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Rasterize the data')
    # Set the ground as "building" (understand solid wall) - after getting grid size
    nx, ny, nz = nPoints.values()
    df_gridBuil = df_gridBuil.reindex(
        df_gridBuil.index.append(
            pd.MultiIndex.from_product([range(1, nx - 1), range(1, ny - 1), [0]])
        )
    )

    # Set the buildGrid3D object to zero when a cell intersect a building
    buildGrid3D = pd.Series(1, index=df_wind0.index, dtype=np.int32)
    buildGrid3D.loc[df_gridBuil.index] = 0

    # Convert building coordinates and wind speeds to numpy matrix...
    # (note that v axis direction is changed since we first use Röckle schemes
    # considering wind speed coming from North thus axis facing South)
    buildGrid3D = np.array(
        [buildGrid3D.xs(i, level=0).unstack().values for i in range(0, nx)]
    )
    u0 = np.array([df_wind0[U].xs(i, level=0).unstack().values for i in range(0, nx)])
    v0 = -np.array([df_wind0[V].xs(i, level=0).unstack().values for i in range(0, nx)])
    w0 = np.array([df_wind0[W].xs(i, level=0).unstack().values for i in range(0, nx)])

    # Identify all cells needing to be updated by the wind solver and store
    # their coordinates in a 1D array
    # (exclude buildings and sketch boundaries)
    cells4Solver = np.transpose(np.where(buildGrid3D == 1))
    cells4Solver = cells4Solver[cells4Solver[:, 0] > 0]
    cells4Solver = cells4Solver[cells4Solver[:, 1] > 0]
    cells4Solver = cells4Solver[cells4Solver[:, 2] > 0]
    cells4Solver = cells4Solver[cells4Solver[:, 0] < nx - 1]
    cells4Solver = cells4Solver[cells4Solver[:, 1] < ny - 1]
    cells4Solver = cells4Solver[cells4Solver[:, 2] < nz - 1]
    cells4Solver = cells4Solver.astype(np.int32)

    # Identify building 3D coordinates
    buildingCoordinates = np.stack(np.where(buildGrid3D == 0)).astype(np.int32)

    # Interpolation is made in order to have wind speed located on the face of
    # each grid cell
    u0[1:nx, :, :] = (u0[0 : nx - 1, :, :] + u0[1:nx, :, :]) / 2
    v0[:, 1:ny, :] = (v0[:, 0 : ny - 1, :] + v0[:, 1:ny, :]) / 2
    w0[:, :, 1:nz] = (w0[:, :, 0 : nz - 1] + w0[:, :, 1:nz]) / 2

    # Reset input and output wind speed to zero for building cells
    u0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    u0[buildingCoordinates[0] + 1, buildingCoordinates[1], buildingCoordinates[2]] = 0
    v0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    v0[buildingCoordinates[0], buildingCoordinates[1] + 1, buildingCoordinates[2]] = 0
    w0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    w0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2] + 1] = 0

    # Create local grid space coordinates (x, y, z)
    Lz = (nz - 1) * dz
    Lx = (nx - 1) * meshSize
    Ly = (ny - 1) * meshSize
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    print(
        'Time spent for wind speed initialization: {0} s'.format(
            time.time() - timeStartCalculation
        )
    )
    print(
        'Shape: '
        + str(u0.shape)
        + ' - '
        + 'Nb cells: '
        + str(u0.shape[0] * u0.shape[1] * u0.shape[2])
    )
    # -------------------------------------------------------------------
    # 10. WIND SOLVER APPLICATION ----------------------------------------
    # -------------------------------------------------------------------
    if feedback:
        feedback.setProgressText('Apply the wind solver equations')
    if not onlyInitialization:
        # Apply a mass-flow balance to have a more physical 3D wind speed field
        u, v, w = WindSolver.solver(
            x=x,
            y=y,
            z=z,
            dx=meshSize,
            dy=meshSize,
            dz=dz,
            u0=u0,
            v0=v0,
            w0=w0,
            buildingCoordinates=buildingCoordinates,
            cells4Solver=cells4Solver,
            maxIterations=maxIterations,
            thresholdIterations=thresholdIterations,
            feedback=feedback,
        )
    else:
        u = u0
        v = v0
        w = w0

    # Wind speed values are recentered to the middle of the cells
    u[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        u[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + u[1:nx, 0 : ny - 1, 0 : nz - 1]
    ) / 2
    v[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        v[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + v[0 : nx - 1, 1:ny, 0 : nz - 1]
    ) / 2
    w[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        w[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + w[0 : nx - 1, 0 : ny - 1, 1:nz]
    ) / 2
    u0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        u0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + u0[1:nx, 0 : ny - 1, 0 : nz - 1]
    ) / 2
    v0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        v0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + v0[0 : nx - 1, 1:ny, 0 : nz - 1]
    ) / 2
    w0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] = (
        w0[0 : nx - 1, 0 : ny - 1, 0 : nz - 1] + w0[0 : nx - 1, 0 : ny - 1, 1:nz]
    ) / 2

    # Reset input and output wind speed to zero for building cells
    u[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    v[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    w[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    u0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    v0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0
    w0[buildingCoordinates[0], buildingCoordinates[1], buildingCoordinates[2]] = 0

    # -------------------------------------------------------------------
    # 11. ROTATE THE WIND FIELD TO THE INITIAL DISPOSITION --------------
    # -------------------------------------------------------------------
    # Get the relative position of the upper right corner of the grid from
    # the center of rotation used to rotate the grid
    cursor.execute(
        """{0};{1}
        """.format(
            DataUtil.createIndex(
                tableName=gridPoint, fieldName=ID_POINT_X, isSpatial=False
            ),
            DataUtil.createIndex(
                tableName=gridPoint, fieldName=ID_POINT_Y, isSpatial=False
            ),
        )
    )
    cursor.execute(
        """
        SELECT  {3}-ST_X(a.{0}) AS DIST_ROT_X,
                {4}-ST_Y(a.{0}) AS DIST_ROT_Y
        FROM {5} AS a
        WHERE   a.{1} = (SELECT MAX({1}) FROM {5})
                AND a.{2} = (SELECT MAX({2}) FROM {5})
        """.format(
            GEOM_FIELD,
            ID_POINT_X,
            ID_POINT_Y,
            rotationCenterCoordinates[0],
            rotationCenterCoordinates[1],
            gridPoint,
        )
    )
    dist_rot_x, dist_rot_y = cursor.fetchall()[0]
    x += dist_rot_x
    y += dist_rot_y

    x_rot = np.zeros((nx, ny))
    y_rot = np.zeros((nx, ny))
    x_rot, y_rot, u_rot, v_rot = rotateData(
        theta=-windDirection * np.pi / 180,
        nx=nx,
        ny=ny,
        nz=nz,
        x=x,
        y=y,
        x_rot=x_rot,
        y_rot=y_rot,
        u=u,
        v=v,
    )

    x_rot, y_rot, u0_rot, v0_rot = rotateData(
        theta=-windDirection * np.pi / 180,
        nx=nx,
        ny=ny,
        nz=nz,
        x=x,
        y=y,
        x_rot=x_rot,
        y_rot=y_rot,
        u=u0,
        v=v0,
    )
    # Set the real (x,y) grid coordinates
    x_rot += rotationCenterCoordinates[0]
    y_rot += rotationCenterCoordinates[1]

    # -------------------------------------------------------------------
    # 12. SAVE EACH OF THE UROCK OUTPUT ---------------------------------
    # -------------------------------------------------------------------
    # First rotate the coordinates of the grid of points
    rotated_grid = Obstacles.windRotation(
        cursor=cursor,
        dicOfInputTables={gridPoint: gridPoint},
        rotateAngle=-windDirection,
        rotationCenterCoordinates=rotationCenterCoordinates,
    )[0][gridPoint]

    dicVectorTables, netcdf_path = saveData.saveBasicOutputs(
        cursor=cursor,
        z_out=z_out,
        dz=dz,
        u=u_rot,
        v=v_rot,
        w=w,
        gridName=rotated_grid,
        verticalWindProfile=verticalWindProfile,
        outputFilePath=outputFilePath,
        outputFilename=outputFilename,
        meshSize=meshSize,
        outputRaster=outputRaster,
        saveRaster=saveRaster,
        saveVector=saveVector,
        saveNetcdf=saveNetcdf,
        prefix_name=prefix,
    )

    # Save also the initialisation field if needed
    if debug:
        dicVectorTables_ini, netcdf_path_ini = saveData.saveBasicOutputs(
            cursor=cursor,
            z_out=z_out,
            dz=dz,
            u=u0_rot,
            v=v0_rot,
            w=w0,
            gridName=rotated_grid,
            verticalWindProfile=verticalWindProfile,
            outputFilePath=tempoDirectory,
            outputFilename='wind_initiatlisation',
            meshSize=meshSize,
            outputRaster=outputRaster,
            saveRaster=saveRaster,
            saveVector=saveVector,
            saveNetcdf=saveNetcdf,
            prefix_name=prefix,
        )
    else:
        dicVectorTables_ini = None
        netcdf_path_ini = None

    return (
        u_rot,
        v_rot,
        w,
        u0_rot,
        v0_rot,
        w0,
        x_rot,
        y_rot,
        z,
        buildingCoordinates,
        cursor,
        rotated_grid,
        rotationCenterCoordinates,
        verticalWindProfile,
        dicVectorTables,
        netcdf_path,
        netcdf_path_ini,
    )


@jit(nopython=True)
def rotateData(theta, nx, ny, nz, x, y, x_rot, y_rot, u, v):
    u_rot = np.zeros(u.shape)
    v_rot = np.zeros(v.shape)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    xmax = x.max()
    ymax = y.max()
    for i in range(nx):
        for j in range(ny):
            x_rot[i, j] = (xmax - x[i]) * rot[0, 0] + (ymax - y[j]) * rot[0, 1]
            y_rot[i, j] = (xmax - x[i]) * rot[1, 0] + (ymax - y[j]) * rot[1, 1]
            for k in range(nz):
                u_rot[i, j, k] = u[i, j, k] * rot[0, 0] + v[i, j, k] * rot[0, 1]
                v_rot[i, j, k] = u[i, j, k] * rot[1, 0] + v[i, j, k] * rot[1, 1]

    return x_rot, y_rot, u_rot, v_rot


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    prefix = 'test_3'
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    temp_dir = temp_dir.name

    main(
        prefix=prefix,
        pluginDirectory=root,
        z_ref=10,
        v_ref=3.0,
        windDirection=90,
        z_out=[2],
        profileType='power',
        outputFilePath=OUTPUT_DIRECTORY,
        buildingFilePath=os.path.join(
            INPUT_DIRECTORY, 'mon_quartier', BUILDING_FILENAME
        ),
        srid=3857,
        tempoDirectory=temp_dir,
    )
