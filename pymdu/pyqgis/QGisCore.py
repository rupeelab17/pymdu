import json
import os
import platform
import sys

import pandas as pd
from pymdu.GeoCore import GeoCore
from qgis.PyQt import QtCore
from qgis.core import QgsApplication


class QGisCore(GeoCore):
    def __init__(self, output_dir: os.path):
        super().__init__()
        print('__init__ QGisCore')
        self.output_dir = output_dir
        if not os.path.exists(os.path.join(self.output_dir, 'qgis_sys_paths.csv')):
            self.qgis_json_path = self.__create_qgis_path()

        if os.path.exists(os.path.join(self.output_dir, 'qgis_env.json')):
            qepath = os.path.join(self.output_dir, 'qgis_env.json')
            with open(qepath, 'r') as f:
                file_contents = f.read()

            self.qgis_json_path = json.loads(file_contents)

        # QtCore.qInstallMessageHandler(self.__qt_message_handler)
        # path to your qgis installation
        gui_flag = False

        print('__init__ qgsApp')
        self.qgsApp = QgsApplication([], gui_flag)

        # initializing processing module
        if platform.system() in ['Darwin', 'Linux']:
            print('platform.system()', platform.system())
            self.qgsApp.setPrefixPath(self.qgis_json_path['HOME'], True)

        else:
            self.qgsApp.setPrefixPath(self.qgis_json_path['HOMEPATH'], True)
        self.qgsApp.initQgis()  # use qgs.exitQgis() to exit the processing module at the end of the script.qgsApp.initQgis()  # use qgs.exitQgis() to exit the processing module at the end of the script.

    def __create_qgis_path(self):
        paths = sys.path
        if platform.system() not in ['Darwin', 'Linux']:
            paths.append(
                'C:/Users/simon/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins'
            )

        df = pd.DataFrame({'paths': paths})
        df.to_csv(os.path.join(self.output_dir, 'qgis_sys_paths.csv'), index=False)

        env = dict(os.environ)
        rem = ['SECURITYSESSIONID', 'LaunchInstanceID', 'TMPDIR']
        _ = [env.pop(r, None) for r in rem]
        with open(os.path.join(self.output_dir, 'qgis_env.json'), 'w') as f:
            json.dump(env, f, ensure_ascii=False, indent=4)

        # set up system paths
        qspath = os.path.join(self.output_dir, 'qgis_sys_paths.csv')
        # provide the path where you saved this file.
        paths = pd.read_csv(qspath).paths.tolist()
        sys.path += paths
        # set up environment variables
        qepath = os.path.join(self.output_dir, 'qgis_env.json')
        with open(qepath, 'r') as f:
            file_contents = f.read()

        qgis_json_path = json.loads(file_contents)

        for k, v in qgis_json_path.items():
            os.environ[k] = v

        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['QGIS_DISABLE_MESSAGE_HOOKS'] = '1'
        os.environ['QGIS_NO_OVERRIDE_IMPORT'] = '1'

        # if platform.system() == "Darwin":
        #     fileExist1 = os.path.isfile(qspath)
        #     fileExist2 = os.path.isfile(qepath)
        #     if fileExist1 and fileExist2:
        #         os.remove(qspath)
        #         os.remove(qepath)

        return qgis_json_path

    @staticmethod
    def __qt_message_handler(mode, context, message):
        if mode == QtCore.QtInfoMsg:
            mode = 'INFO'
        elif mode == QtCore.QtWarningMsg:
            mode = 'WARNING'
            if 'propagateSizeHints' in message:
                return
        elif mode == QtCore.QtCriticalMsg:
            mode = 'CRITICAL'
        elif mode == QtCore.QtFatalMsg:
            mode = 'FATAL'
        else:
            mode = 'DEBUG'
        print(
            'qt_message_handler: line: %d, func: %s(), file: %s'
            % (context.line, context.function, context.file)
        )
        print('  %s: %s\n' % (mode, message))

    # @property
    # def qgsApp(self):
    #     return self.qgsApp

    def run_processing(
        self, name='umep:Spatial Data: Tree Generator', options: dict = None
    ):
        pass
