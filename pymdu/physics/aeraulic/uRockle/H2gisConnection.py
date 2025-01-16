#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:49:12 2021

@author: Jérémy Bernard, University of Gothenburg
"""

# !/usr/bin/python
from __future__ import print_function

import re

import urllib3

try:
    # modif simon
    # path_pybin = DataUtil.locate_py()
    # subprocess.check_call([str(path_pybin), "-m", "pip", "install", "jaydebeapi"])
    import jaydebeapi

except ImportError:
    print("'jaydebeapi' Python package is missing, cannot connect to H2 Driver")
    exit(1)

try:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil

except ImportError:
    from pymdu.physics.aeraulic.uRockle.GlobalVariables import *
    import pymdu.physics.aeraulic.uRockle.DataUtil as DataUtil

# Global variables
# H2GIS_VERSION = "1.5.0"
# H2GIS_URL = H2GIS_VERSION.join(["https://github.com/orbisgis/h2gis/releases/download/v",
#                                 "/h2gis-dist-",
#                                 "-bin.zip"])
# H2GIS_UNZIPPED_NAME = "h2gis-standalone" + os.sep + "h2gis-dist-" + H2GIS_VERSION + ".jar"

H2GIS_VERSION = '2.0.0'
H2GIS_URL = 'https://github.com/orbisgis/h2gis/releases/download/v2.0.0/h2gis-dist-2.0.0-bin.zip'
H2GIS_UNZIPPED_NAME = (
    'h2gis-standalone' + os.sep + 'h2gis-dist-' + H2GIS_VERSION + '-SNAPSHOT.jar'
)

JAVA_PATH_POSIX = [os.path.join(os.sep, 'usr', 'lib', 'jvm')]
JAVA_PATH_NT = [
    os.path.join('C:', os.sep, 'Program Files', 'Java'),
    os.path.join('C:', os.sep, 'Program Files', 'OpenJDK'),
    os.path.join('C:', os.sep, 'Program Files (x86)', 'Java'),
    os.path.join('C:', os.sep, 'Program Files (x86)', 'OpenJDK'),
]


def downloadH2gis(dbDirectory):
    """Download the H2GIS spatial database management system (used for Röckle zone calculation)
    For more information about use with Python: https://github.com/orbisgis/h2gis/wiki/4.4-Use-H2GIS-with-Python

            Parameters
            _ _ _ _ _ _ _ _ _ _

                    dbDirectory: String
                            Directory where shoud be stored the H2GIS jar

            Returns
            _ _ _ _ _ _ _ _ _ _

        None"""
    # Get the zip file name and create the local file directory
    zipFileName = H2GIS_URL.split('/')[-1]
    localH2ZipDir = (dbDirectory + os.sep + zipFileName).encode('utf-8')
    localH2JarDir = (dbDirectory + os.sep + H2GIS_UNZIPPED_NAME).encode('utf-8')

    # Test whether the .jar already downloaded
    if os.path.exists(localH2ZipDir) or os.path.exists(localH2JarDir):
        print('H2GIS version %s already downloaded' % (H2GIS_VERSION))
    else:
        print('Downloading H2GIS version %s at %s...' % (H2GIS_URL, H2GIS_VERSION))
        # Download the archive file and save it into the 'dbDirectory'
        http = urllib3.PoolManager()
        r = http.request('GET', url=H2GIS_URL, preload_content=False)
        with open(localH2ZipDir, 'wb') as out:
            while True:
                data = r.read(2**16)
                if not data:
                    break
                out.write(data)
        r.release_conn()
        print('H2GIS version %s downloaded !!' % (H2GIS_VERSION))

    # Test whether the .jar already exists
    if os.path.exists(localH2JarDir):
        print('H2GIS version %s already unzipped' % (H2GIS_VERSION))
    else:
        print('Unzipping H2GIS version %s...' % (H2GIS_VERSION))
        # Unzip the H2GIS archive
        print(dbDirectory)
        print(zipFileName)
        DataUtil.decompressZip(dbDirectory, zipFileName)
        print('H2GIS version %s unzipped !!' % (H2GIS_VERSION))


def startH2gisInstance(
    dbDirectory,
    dbInstanceDir=TEMPO_DIRECTORY,
    instanceName=INSTANCE_NAME,
    instanceId=INSTANCE_ID,
    instancePass=INSTANCE_PASS,
    newDB=NEW_DB,
):
    """Start an H2GIS spatial database instance (used for Röckle zone calculation)
    For more information about use with Python: https://github.com/orbisgis/h2gis/wiki/4.4-Use-H2GIS-with-Python

                Parameters
                _ _ _ _ _ _ _ _ _ _

                        dbDirectory: String
                                Directory where is stored the H2GIS jar
            dbInstanceDir: String
                Directory where should be started the H2GIS instance
            instanceName: String, default INSTANCE_NAME
                File name used for the database
            instanceId: String, default INSTANCE_ID
                ID used to connect to the database
            instancePass: String, default INSTANCE_PASS
                password used to connect to the database
            newDB: Boolean, default NEW_DB
                Whether or not all existing 'public' tables should be deleted
                (if the DB already exists)

                Returns
                _ _ _ _ _ _ _ _ _ _

            cur: conn.cursor
                A cursor object, used to perform queries"""
    # DB extension
    dbExtension = '.mv.db'
    dbTraceExtension = '.trace.db'

    # Define where are the jar of the DB and the H2GIS instance (in absolute paths)
    localH2JarDir = dbDirectory + os.sep + H2GIS_UNZIPPED_NAME
    localH2InstanceDir = dbInstanceDir + os.sep + instanceName

    isDbExist = os.path.exists(localH2InstanceDir + dbExtension)

    # print the connection string we will use to connect
    print('Connecting to database\n	->%s' % (localH2InstanceDir))
    print(localH2JarDir)

    # If the DB already exists and if 'newDB' is set to True, delete all the DB files
    if isDbExist & newDB:
        os.remove(localH2InstanceDir + dbExtension)
        if os.path.exists(localH2InstanceDir + dbTraceExtension):
            os.remove(localH2InstanceDir + dbTraceExtension)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = jaydebeapi.connect(
        'org.h2.Driver',
        'jdbc:h2:'
        + localH2InstanceDir
        + ';AUTO_SERVER=TRUE;DB_CLOSE_ON_EXIT=TRUE;LOCK_TIMEOUT=10000;',
        [instanceId, instancePass],
        localH2JarDir,
    )

    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cur = conn.cursor()
    print('Connected!\n')

    # If the DB already exists and if 'newDB' is set to True, delete all 'public' tables
    # if isDbExist & newDB:
    #     cur.execute("""SELECT TABLE_NAME
    #                     FROM INFORMATION_SCHEMA.TABLES
    #                     WHERE TABLE_SCHEMA = 'PUBLIC'""")
    #     tableNames = [i[0] for i in cur.fetchall()]
    #     cur.execute("""
    #         DROP TABLE IF EXISTS {0}""".format(",".join(tableNames)))

    # Init spatial features
    cur.execute(
        'CREATE ALIAS IF NOT EXISTS H2GIS_SPATIAL FOR "org.h2gis.functions.factory.H2GISFunctions.load";'
    )
    cur.execute('CALL H2GIS_SPATIAL();')
    print('Spatial functions added!\n')

    return cur


def setJavaDir(javaPath):
    """If there is no JAVA variable environment set or neither already one
    saved in the URock repository, ask the user to enter one for
    local use

                Parameters
                _ _ _ _ _ _ _ _ _ _

            javaPath: String
                JAVA variable path

                Returns
                _ _ _ _ _ _ _ _ _ _

            None"""
    os.environ.setdefault('JAVA_HOME', javaPath)


def getJavaDir(pluginDirectory):
    """Try to get the JAVA variable environment if already set.
    Otherwise try to get it from the user plugin repository.
    Otherwise try to set from the OS type

                Parameters
                _ _ _ _ _ _ _ _ _ _

            plugin_directory: String
                Path of the plugin directory where could be saved the java path

                Returns
                _ _ _ _ _ _ _ _ _ _

            javaPath: String
                JAVA variable path"""
    javaPath = os.environ.get('JAVA_HOME')
    javaPathFile = os.path.join(pluginDirectory, JAVA_PATH_FILENAME)

    # For some reason, JAVA_HOME may be set to %JAVA_HOME% while there is no
    # Java home set. This should be associated to None
    if javaPath:
        if javaPath[0] == '%':
            javaPath == None
    if not javaPath:
        if os.path.exists(javaPathFile):
            javaFilePath = open(javaPathFile, 'r')
            javaPath = javaFilePath.read()
            javaFilePath.close()
        else:
            os_type = os.name
            if os_type == 'posix':
                javaPath = identifyJavaDir(JAVA_PATH_POSIX)
            else:
                javaPath = identifyJavaDir(JAVA_PATH_NT)

    return javaPath


def saveJavaDir(javaPath, pluginDirectory):
    """Save the java path into a file in the user plugin repository

        Parameters
        _ _ _ _ _ _ _ _ _ _

    javaPath: String
        JAVA variable path

        Returns
        _ _ _ _ _ _ _ _ _ _

    None"""
    javaPathFile = os.path.join(pluginDirectory, JAVA_PATH_FILENAME)
    if not os.path.exists(pluginDirectory):
        os.makedirs(pluginDirectory)
    if os.path.exists(javaPathFile):
        existingJavaFilePath = open(javaPathFile, 'r', encoding='utf8')
        existingJavaPath = existingJavaFilePath.read()
        existingJavaFilePath.close()
    else:
        existingJavaPath = None
    if (existingJavaPath != javaPath) or existingJavaPath is None:
        if os.path.exists(javaPathFile):
            os.remove(javaPathFile)
        javaFilePath = open(javaPathFile, 'w', encoding='utf8')
        javaFilePath.write(javaPath)
        javaFilePath.close()


def identifyJavaDir(java_path_os_list):
    """Try to get the directory where is located the last Java version in the OS...

        Parameters
        _ _ _ _ _ _ _ _ _ _

    java_path_os_list: list of String
        Possible locations of the Java directory

        Returns
        _ _ _ _ _ _ _ _ _ _

    javaPath: String
        JAVA variable path"""
    JavaExists = False
    i = 0
    while not JavaExists:
        javaBaseDir = java_path_os_list[i]
        JavaExists = os.path.exists(javaBaseDir)
        i += 1
    listJavaVersion = os.listdir(javaBaseDir)
    listSplit = pd.Series(
        {i: re.split('[.\-]', v) for i, v in enumerate(listJavaVersion)}
    )
    df_version = pd.DataFrame(
        {
            'version': [listSplit[i][1] for i in listSplit.index],
            'startWith': [listSplit[i][0] for i in listSplit.index],
        }
    )
    highestVersion = (
        df_version[
            (df_version.startWith == 'java')
            | (df_version.startWith == 'jdk')
            | (df_version.startWith == 'jre1')
        ]
        .version.astype(int)
        .idxmax()
    )
    javaPath = os.path.join(javaBaseDir, listJavaVersion[highestVersion])

    return javaPath
