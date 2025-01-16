# -*- coding: utf-8 -*-
import zipfile
import shutil
import errno
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
import platform
from packaging import version

try:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *


def decompressZip(dirPath, inputFileName, outputFileBaseName=None, deleteZip=False):
    """
    Decompress zip file.

    Parameters
    _ _ _ _ _ _ _ _ _ _
        dirPath: String
            Directory path where is located the zip file
        inputFileName: String
            Name of the file to unzip (with .zip at the end)
        outputFileBaseName: String
            Base name of the file to unzip (without extension)
        deleteZip: boolean, default False
            Whether or not the input zip file should be removed

    Returns
    -------
        None
    """
    print('Start decompressing zip file')

    with open(os.path.join(dirPath, inputFileName), 'rb') as zipsrc:
        zfile = zipfile.ZipFile(zipsrc)
        for member in zfile.infolist():
            print(member.filename + ' is being decompressed')
            if outputFileBaseName is None:
                target_path = os.path.join(dirPath, member.filename)
            else:
                # Initialize output file path
                target_path = os.path.join(dirPath, outputFileBaseName)
                extension = '.' + member.filename.split('.')[-1]
                target_path += extension

            # Create a folder if needed
            if target_path.endswith('/'):  # folder entry, create
                try:
                    os.makedirs(target_path)
                except (OSError, IOError) as err:
                    # Windows may complain if the folders already exist
                    if err.errno != errno.EEXIST:
                        raise
                continue
            with open(target_path, 'wb') as outfile, zfile.open(member) as infile:
                shutil.copyfileobj(infile, outfile)

    return None


def degToRad(angleDeg, origin=0, direction='CLOCKWISE'):
    """Convert angle arrays from degrees to radian.

    Parameters
        _ _ _ _ _ _ _ _ _ _
                angleDeg : float
                        Angle in degrees
                origin : float, default 0
                        Origin of the input degree coordinates (given in a reference North clockwise coordinate system)
                direction : {"CLOCKWISE", "COUNTER-CLOCKWISE"}, default "CLOCKWISE"
                        Direction where go the input coordinate

    Returns
        _ _ _ _ _ _ _ _ _ _
                angle in radian (trigonometric reference).
    """
    if direction == 'CLOCKWISE':
        d = 1
    if direction == 'COUNTER-CLOCKWISE':
        d = -1

    return (angleDeg + d * origin) * np.pi / 180


def postfix(tableName, suffix=None, separator='_'):
    """Add a suffix to an input table name

    Parameters
        _ _ _ _ _ _ _ _ _ _
                tableName : String
                        Name of the input table
        suffix : String, default None (then current datetime is used as string)
            Suffix to add to the table name
        separator : String, default "_"
            Character to separate tableName from suffix


    Returns
        _ _ _ _ _ _ _ _ _ _
                The input table name with the suffix"""
    if suffix is None:
        suffix = datetime.now().strftime('%Y%m%d%H%M%S')

    return tableName + separator + suffix


def prefix(tableName, prefix=PREFIX_NAME, separator='_'):
    """Add a suffix to an input table name

    Parameters
        _ _ _ _ _ _ _ _ _ _
                tableName : String
                        Name of the input table
        prefix : String
            Prefix to add to the table name
        separator : String, default "_"
            Character to separate prefix from tableName

    Returns
        _ _ _ _ _ _ _ _ _ _
                The input table name with the prefix"""
    if prefix == '':
        return tableName
    else:
        return prefix + separator + tableName


def getColumns(cursor, tableName):
    """Get the column name of a table into a list

    Parameters
        _ _ _ _ _ _ _ _ _ _
        cursor: conn.cursor
            A cursor object, used to perform spatial SQL queries
                tableName : String
                        Name of the input table

    Returns
        _ _ _ _ _ _ _ _ _ _
                columnNames: list
            A list of the table column names"""
    cursor.execute("""SELECT * FROM {0}""".format(tableName))
    columnNames = [info[0] for info in cursor.description]

    return columnNames


def readFunction(extension):
    """Return the name of the right H2GIS function to use depending of the file extension

    Parameters
        _ _ _ _ _ _ _ _ _ _
        extension: String
            Extension of the vector file (shp or geojson)

    Returns
        _ _ _ _ _ _ _ _ _ _
                h2gisFunctionName: String
            Return the name of the H2GIS function to use"""
    if extension.lower() == 'shp':
        return 'SHPREAD'
    elif extension.lower() == 'geojson':
        return 'GEOJSONREAD'
    elif extension.lower() == 'csv':
        return 'CSVREAD'


def createIndex(tableName, fieldName, isSpatial):
    """Return the SQL query needed to create an index on a given field of a
    given table. The index should be indicated as spatial if the field is
    a geometry field.

    Parameters
        _ _ _ _ _ _ _ _ _ _
        tableName: String
            Name of the table
        fieldName: String
            Name of the field the index will be created on
        isSpatial: boolean
            Whether or not the index is a spatial index (should be True if
                                                         the field is a geometry field)

    Returns
        _ _ _ _ _ _ _ _ _ _
                query: String
            Return the SQL query needed to create the index"""
    spatialKeyWord = ''
    if isSpatial:
        spatialKeyWord = ' SPATIAL '
    query = 'CREATE {0} INDEX IF NOT EXISTS id_{1}_{2} ON {2}({1});'.format(
        spatialKeyWord, fieldName, tableName
    )
    return query


def radToDeg(data, origin=90, direction='CLOCKWISE'):
    """Convert angle arrays from radian to degree.

    Parameters
        _ _ _ _ _ _ _ _ _ _
                data : pd.Series()
                        Array containing the angle values to convert from radian to degree.
                origin : float
                        Origin of the output coordinate (given in a reference trigonometric coordinate)
                direction : {"CLOCKWISE", "COUNTER-CLOCKWISE"}, default "CLOCKWISE"
                        Direction where go the output coordinate

    Returns
        _ _ _ _ _ _ _ _ _ _
                Array containing the data in degree coordinate.
    """
    if direction == 'CLOCKWISE':
        degree = (360 - data * 180 / np.pi) + origin
    if direction == 'COUNTER-CLOCKWISE':
        degree = (data * 180 / np.pi) - origin

    degree[degree > 360] = degree[degree > 360] - 360
    degree[degree < 0] = degree[degree < 0] + 360

    return degree


def windDirectionFromXY(windSpeedEast, windSpeedNorth):
    """
    Calculates wind direction from wind speeds in carthesian coordinates.

    Parameters
    _ _ _ _ _ _ _ _ _ _
        windSpeedEast: pd.Series
            Wind speed along a West->East axis (m/s)
        windSpeedNorth: pd.Series
            Wind speed along a South->North axis (m/s)

    Returns
    -------
        pd.Series containing the wind direction from East counterclockwise.
    """
    # Calculate the angle in Radian in a [-pi/2, pi/2]
    radAngle = np.zeros(windSpeedEast.shape)
    radAngle[windSpeedEast == 0] = 0
    if type(windSpeedEast) == type(pd.Series(dtype='float64')):
        radAngle[windSpeedEast != 0] = np.arctan(
            windSpeedNorth[windSpeedEast != 0].divide(windSpeedEast[windSpeedEast != 0])
        )
    else:
        radAngle[windSpeedEast != 0] = np.arctan(
            windSpeedNorth[windSpeedEast != 0] / windSpeedEast[windSpeedEast != 0]
        )

    # Add or subtract pi.2 for left side trigonometric circle vectors
    radAngle[(windSpeedEast <= 0) & (windSpeedNorth > 0)] = (
        radAngle[(windSpeedEast <= 0) & (windSpeedNorth > 0)] + np.pi
    )
    radAngle[(windSpeedEast < 0) & (windSpeedNorth <= 0)] = (
        radAngle[(windSpeedEast < 0) & (windSpeedNorth <= 0)] + np.pi
    )
    radAngle[(windSpeedEast >= 0) & (windSpeedNorth < 0)] = (
        radAngle[(windSpeedEast >= 0) & (windSpeedNorth < 0)] + 2 * np.pi
    )

    return radAngle


def getExtremumPoint(
    pointsTable, axis, extremum, secondAxisExtremum, cursor, prefix_name
):
    """Identify the point geometry being an extremum ("MIN" or "MAX"") of a polygon
    along a given axis ("X" or "Y"). If two points are at the same "X" (or "Y"),
    keep only lowest value or highest ("MIN" or "MAX") along the second
    axis "Y" (or "X").

    Parameters
        _ _ _ _ _ _ _ _ _ _
        pointsTable: String
            Name of the table containing the points corresponding to all polygons,
            the polygon id and the extremum of the polygon we are looking for
        axis: string
            Axis along which the extremum is spotted.
                -> "X": along x-axis
                -> "Y": along y-axis
        extremum: string
            Type of extremum to spot:
                -> "MIN": minimum value
                -> "MAX": maximum value
        secondAxisExtremum: string
            If two points fit with the extremum value, keep only one value depending
            on the value along the second axis.
                -> "MIN": keep the point corresponding to the lowest value
                along the other axis
                -> "MAX": keep the point corresponding to the highest value
                along the other axis
                -> "AVG": keep an average value of the points being along the
                other axis
        cursor: conn.cursor
            A cursor object, used to perform spatial SQL queries
        prefix: String, default PREFIX_NAME
            Prefix to add to the output table name

    Returns
        _ _ _ _ _ _ _ _ _ _
                extremumPointTable: String
            Return the table containing the expected extremum point for each polygon"""
    # Output base name
    outputBaseName = '{0}_{1}_{2}_POINTS'.format(pointsTable, axis, extremum)

    # Name of the output table
    extremumPointTable = prefix(outputBaseName, prefix=prefix_name)

    # Extremum field name
    extremumField = axis + '_' + extremum

    # Set the secondary axis and the point creation query depending on 1st axis
    if axis == 'X':
        secondaryAxis = 'Y'
        pointCreationQuery = 'ST_POINT({0}, {1}({2}))'.format(
            axis, secondAxisExtremum, secondaryAxis
        )
    else:
        secondaryAxis = 'X'
        pointCreationQuery = 'ST_POINT({1}({2}), {0})'.format(
            axis, secondAxisExtremum, secondaryAxis
        )

    # Identify the extremum point
    cursor.execute(
        """
           {0}{1}{2}
           DROP TABLE IF EXISTS {3};
           CREATE TABLE {3}({4} INTEGER, {5} GEOMETRY)
               AS SELECT {4}, {6}
               FROM {7}
               WHERE {8} = {9}
               GROUP BY {4};
           """.format(
            createIndex(
                tableName=pointsTable, fieldName=ID_FIELD_STACKED_BLOCK, isSpatial=False
            ),
            createIndex(tableName=pointsTable, fieldName=axis, isSpatial=False),
            createIndex(
                tableName=pointsTable, fieldName=extremumField, isSpatial=False
            ),
            extremumPointTable,
            ID_FIELD_STACKED_BLOCK,
            GEOM_FIELD,
            pointCreationQuery,
            pointsTable,
            axis,
            extremumField,
        )
    )

    return extremumPointTable


####### SHOULD BE DELETED SINCE ALREADY IN UMEP !!!
def locate_py():
    # get Python version
    str_ver_qgis = sys.version.split(' ')[0]

    try:
        # non-Linux
        path_py = os.environ['PYTHONHOME']
    except Exception:
        # Linux
        path_py = sys.executable

    # convert to Path for eaiser processing
    path_py = Path(path_py)

    # pre-defined paths for python executable
    dict_pybin = {
        'Darwin': path_py / 'bin' / 'python3',
        'Windows': path_py
        / (
            '../../bin/pythonw.exe'
            if version.parse(str_ver_qgis) >= version.parse('3.9.1')
            else 'pythonw.exe'
        ),
        'Linux': path_py,
    }

    # python executable
    path_pybin = dict_pybin[platform.system()]

    if path_pybin.exists():
        return path_pybin
    else:
        raise RuntimeError('UMEP cannot locate the Python interpreter used by QGIS!')
