import multiprocessing

import joblib
from joblib import Parallel, delayed

import os
import sys

import numpy as np
from osgeo import gdal
import linecache

from pymdu.physics.solar.UtilitiesSolar import saveraster

from pymdu.physics.solar.solweig.SEBESOLWEIGCommonFiles.clearnessindex_2013b import (
    clearnessindex_2013b,
)
from pymdu.physics.solar.solweig.SOLWEIGpython import Solweig_2022a_calc as so
from pymdu.physics.solar.solweig import PET_calculations as p
from pymdu.physics.solar.solweig import UTCI_calculations as utci


def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return 'EXCEPTION IN {}, \nLINE {} "{}" \nERROR MESSAGE: {}'.format(
        filename, lineno, line.strip(), exc_obj
    )


def saveraster_parallel(
    RasterYSize, RasterXSize, GetGeoTransform, GetProjection, filename, raster
):
    rows = RasterYSize
    cols = RasterXSize

    outDs = gdal.GetDriverByName('GTiff').Create(
        filename, cols, rows, int(1), gdal.GDT_Float32
    )
    outBand = outDs.GetRasterBand(1)

    # write the data
    outBand.WriteArray(raster, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-9999)

    # georeference the image and set the projection
    outDs.SetGeoTransform(GetGeoTransform)
    outDs.SetProjection(GetProjection)
    del outDs


def calculate_tmrt_with_parallel(i, data, tmrt_list_shared):
    try:
        # numformat = '%d %d %d %d %.5f ' + '%.2f ' * 28
        numformat = (
            '%d %d %d %d %.5f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f '
            '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
        )

        # Daily water body temperature
        if data['landcover'] == 1:
            if (data['dectime'][i] - np.floor(data['dectime'][i])) == 0 or (i == 0):
                Twater = np.mean(
                    data['Ta'][data['jday'][0] == np.floor(data['dectime'][i])]
                )
        # print(dectime[i])
        # Nocturnal cloudfraction from Offerle et al. 2003
        if (data['dectime'][i] - np.floor(data['dectime'][i])) == 0:
            daylines = np.where(np.floor(data['dectime']) == data['dectime'][i])
            if daylines.__len__() > 1:
                alt = data['altitude'][0][daylines]
                alt2 = np.where(alt > 1)
                rise = alt2[0][0]
                [_, CI, _, _, _] = clearnessindex_2013b(
                    data['zen'][0, i + rise + 1],
                    data['jday'][0, i + rise + 1],
                    data['Ta'][i + rise + 1],
                    data['RH'][i + rise + 1] / 100.0,
                    data['radG'][i + rise + 1],
                    data['location'],
                    data['P'][i + rise + 1],
                )  # i+rise+1 to match matlab code. correct?
                if (CI > 1.0) or (CI == np.inf):
                    CI = 1.0
            else:
                CI = 1.0

        (
            Tmrt,
            Kdown,
            Kup,
            Ldown,
            Lup,
            Tg,
            ea,
            esky,
            I0,
            CI,
            shadow,
            firstdaytime,
            timestepdec,
            timeadd,
            Tgmap1,
            Tgmap1E,
            Tgmap1S,
            Tgmap1W,
            Tgmap1N,
            Keast,
            Ksouth,
            Kwest,
            Knorth,
            Least,
            Lsouth,
            Lwest,
            Lnorth,
            KsideI,
            TgOut1,
            TgOut,
            radIout,
            radDout,
            Lside,
            Lsky_patch_characteristics,
            CI_Tg,
            CI_TgG,
            KsideD,
            dRad,
            Kside,
        ) = so.Solweig_2022a_calc(
            i,
            data['dsm'],
            data['scale'],
            data['rows'],
            data['cols'],
            data['svf'],
            data['svfN'],
            data['svfW'],
            data['svfE'],
            data['svfS'],
            data['svfveg'],
            data['svfNveg'],
            data['svfEveg'],
            data['svfSveg'],
            data['svfWveg'],
            data['svfaveg'],
            data['svfEaveg'],
            data['svfSaveg'],
            data['svfWaveg'],
            data['svfNaveg'],
            data['vegdsm'],
            data['vegdsm2'],
            data['albedo_b'],
            data['absK'],
            data['absL'],
            data['ewall'],
            data['Fside'],
            data['Fup'],
            data['Fcyl'],
            data['altitude'][0][i],
            data['azimuth'][0][i],
            data['zen'][0][i],
            data['jday'][0][i],
            data['usevegdem'],
            data['onlyglobal'],
            data['buildings'],
            data['location'],
            data['psi'][0][i],
            data['landcover'],
            data['lcgrid'],
            data['dectime'][i],
            data['altmax'][0][i],
            data['wallaspect'],
            data['wallheight'],
            data['cyl'],
            data['elvis'],
            data['Ta'][i],
            data['RH'][i],
            data['radG'][i],
            data['radD'][i],
            data['radI'][i],
            data['P'][i],
            data['amaxvalue'],
            data['bush'],
            data['Twater'],
            data['TgK'],
            data['Tstart'],
            data['alb_grid'],
            data['emis_grid'],
            data['TgK_wall'],
            data['Tstart_wall'],
            data['TmaxLST'],
            data['TmaxLST_wall'],
            data['first'],
            data['second'],
            data['svfalfa'],
            data['svfbuveg'],
            data['firstdaytime'],
            data['timeadd'],
            data['timestepdec'],
            data['Tgmap1'],
            data['Tgmap1E'],
            data['Tgmap1S'],
            data['Tgmap1W'],
            data['Tgmap1N'],
            data['CI'],
            data['TgOut1'],
            data['diffsh'],
            data['shmat'],
            data['vegshmat'],
            data['vbshvegshmat'],
            data['anisotropic_sky'],
            data['asvf'],
            data['patch_option'],
        )

        # Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        #    Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
        #    Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radIout, radDout = so.Solweig_2021a_calc(
        #         i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg,
        #         svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg,
        #         vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude[0][i],
        #         azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location,
        #         psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], wallaspect,
        #         wallheight, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i], amaxvalue,
        #         bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
        #         TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec,
        #         Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani)

        # Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        # Tgmap1, timeaddE, Tgmap1E, timeaddS, Tgmap1S, timeaddW, Tgmap1W, timeaddN, Tgmap1N, \
        # Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radIout, radDout \
        #     = so.Solweig_2019a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg,
        #         svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg,
        #         vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude[0][i],
        #         azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location,
        #         psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], wallaspect,
        #         wallheight, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i], amaxvalue,
        #         bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
        #         TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timeaddE, timeaddS,
        #         timeaddW, timeaddN, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani)

        tmrt_list_shared.append(Tmrt)

        if data['altitude'][0][i] > 0:
            w = 'D'
        else:
            w = 'N'

        poisxy = data['poisxy']
        # Write to POIs
        if poisxy is not None:
            for k in range(0, poisxy.shape[0]):
                poi_save = np.zeros((1, 35))
                poi_save[0, 0] = data['YYYY'][0][i]
                poi_save[0, 1] = data['jday'][0][i]
                poi_save[0, 2] = data['hours'][i]
                poi_save[0, 3] = data['minu'][i]
                poi_save[0, 4] = data['dectime'][i]
                poi_save[0, 5] = data['altitude'][0][i]
                poi_save[0, 6] = data['azimuth'][0][i]
                poi_save[0, 7] = data['radIout']
                poi_save[0, 8] = data['radDout']
                poi_save[0, 9] = data['radG'][i]
                poi_save[0, 10] = data['Kdown'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 11] = data['Kup'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 12] = data['Keast'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 13] = data['Ksouth'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 14] = data['Kwest'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 15] = data['Knorth'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 16] = data['Ldown'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 17] = data['Lup'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 18] = data['Least'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 19] = data['Lsouth'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 20] = data['Lwest'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 21] = data['Lnorth'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 22] = data['Ta'][i]
                poi_save[0, 23] = data['TgOut'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 24] = data['RH'][i]
                poi_save[0, 25] = data['esky']
                poi_save[0, 26] = data['Tmrt'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 27] = data['I0']
                poi_save[0, 28] = data['CI']
                poi_save[0, 29] = data['shadow'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 30] = data['svf'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 31] = data['svfbuveg'][int(poisxy[k, 2]), int(poisxy[k, 1])]
                poi_save[0, 32] = data['KsideI']
                int(poisxy[k, 2]), int(poisxy[k, 1])
                # Recalculating wind speed based on powerlaw
                WsPET = (1.1 / data['sensorheight']) ** 0.2 * data['Ws'][i]
                WsUTCI = (10.0 / data['sensorheight']) ** 0.2 * data['Ws'][i]
                resultPET = p._PET(
                    data['Ta'][i],
                    data['RH'][i],
                    data['Tmrt'][int(poisxy[k, 2]), int(poisxy[k, 1])],
                    WsPET,
                    data['mbody'],
                    data['age'],
                    data['ht'],
                    data['activity'],
                    data['clo'],
                    data['sex'],
                )
                poi_save[0, 33] = resultPET
                resultUTCI = utci.utci_calculator(
                    data['Ta'][i],
                    data['RH'][i],
                    Tmrt[int(poisxy[k, 2]), int(poisxy[k, 1])],
                    WsUTCI,
                )
                poi_save[0, 34] = resultUTCI
                data_out = (
                    data['folderPath'][0] + '/POI_' + str(data['poiname'][k]) + '.txt'
                )
                # f_handle = file(data_out, 'a')
                f_handle = open(data_out, 'ab')
                np.savetxt(f_handle, poi_save, fmt=numformat)
                f_handle.close()

        if data['hours'][i] < 10:
            XH = '0'
        else:
            XH = ''
        if data['minu'][i] < 10:
            XM = '0'
        else:
            XM = ''

        saveraster_parallel(
            data['RasterYSize'],
            data['RasterXSize'],
            data['GetGeoTransform'],
            data['GetProjection'],
            os.path.join(
                data['folderPath'],
                'Tmrt_'
                + str(int(data['YYYY'][0, i]))
                + '_'
                + str(int(data['DOY'][i]))
                + '_'
                + XH
                + str(int(data['hours'][i]))
                + XM
                + str(int(data['minu'][i]))
                + w
                + '.tif',
            ),
            Tmrt,
        )

        saveraster_parallel(
            data['RasterYSize'],
            data['RasterXSize'],
            data['GetGeoTransform'],
            data['GetProjection'],
            os.path.join(
                data['folderPath'],
                'Shadow_'
                + str(int(data['YYYY'][0, i]))
                + '_'
                + str(int(data['DOY'][i]))
                + '_'
                + XH
                + str(int(data['hours'][i]))
                + XM
                + str(int(data['minu'][i]))
                + w
                + '.tif',
            ),
            shadow,
        )

        # if self.write_other_files:
        #     saveraster(gdal_dsm,
        #                os.path.join(folderPath, 'Kup_' + str(int(YYYY[0, i])) + '_' + str(int(DOY[i]))
        #                             + '_' + XH + str(int(hours[i])) + XM + str(int(minu[i])) + w + '.tif'),
        #                Kup)
        #     saveraster(gdal_dsm,
        #                os.path.join(folderPath, 'Kdown_' + str(int(YYYY[0, i])) + '_' + str(int(DOY[i]))
        #                             + '_' + XH + str(int(hours[i])) + XM + str(int(minu[i])) + w + '.tif'),
        #                Kdown)
        #     saveraster(gdal_dsm,
        #                os.path.join(folderPath, 'Lup_' + str(int(YYYY[0, i])) + '_' + str(int(DOY[i]))
        #                             + '_' + XH + str(int(hours[i])) + XM + str(int(minu[i])) + w + '.tif'),
        #                Lup)
        #     saveraster(gdal_dsm,
        #                os.path.join(folderPath, 'Ldown_' + str(int(YYYY[0, i])) + '_' + str(int(DOY[i]))
        #                             + '_' + XH + str(int(hours[i])) + XM + str(int(minu[i])) + w + '.tif'),
        #                Ldown)

    except Exception:
        errorstring = print_exception()
        print(errorstring)
    return


class SolweigWorker(object):
    def __init__(
        self,
        dsm,
        scale,
        rows,
        cols,
        svf,
        svfN,
        svfW,
        svfE,
        svfS,
        svfveg,
        svfNveg,
        svfEveg,
        svfSveg,
        svfWveg,
        svfaveg,
        svfEaveg,
        svfSaveg,
        svfWaveg,
        svfNaveg,
        vegdsm,
        vegdsm2,
        albedo_b,
        absK,
        absL,
        ewall,
        Fside,
        Fup,
        Fcyl,
        altitude,
        azimuth,
        zen,
        jday,
        usevegdem,
        onlyglobal,
        buildings,
        location,
        psi,
        landcover,
        lcgrid,
        dectime,
        altmax,
        wallaspect,
        wallheight,
        cyl,
        elvis,
        Ta,
        RH,
        radG,
        radD,
        radI,
        P,
        amaxvalue,
        bush,
        Twater,
        TgK,
        Tstart,
        alb_grid,
        emis_grid,
        TgK_wall,
        Tstart_wall,
        TmaxLST,
        TmaxLST_wall,
        first,
        second,
        svfalfa,
        svfbuveg,
        firstdaytime,
        timeadd,
        timeaddE,
        timeaddS,
        timeaddW,
        timeaddN,
        timestepdec,
        Tgmap1,
        Tgmap1E,
        Tgmap1S,
        Tgmap1W,
        Tgmap1N,
        CI,
        YYYY,
        DOY,
        hours,
        minu,
        gdal_dsm,
        folderPath,
        poisxy,
        poiname,
        Ws,
        mbody,
        age,
        ht,
        activity,
        clo,
        sex,
        sensorheight,
        diffsh,
        shmat,
        vegshmat,
        vbshvegshmat,
        anisotropic_sky,
        asvf,
        patch_option,
        write_other_files,
    ):
        self.tmrtplot = None
        self.killed = False

        self.dsm = dsm
        self.scale = scale
        self.rows = rows
        self.cols = cols
        self.svf = svf
        self.svfN = svfN
        self.svfW = svfW
        self.svfE = svfE
        self.svfS = svfS
        self.svfveg = svfveg
        self.svfNveg = svfNveg
        self.svfWveg = svfWveg
        self.svfEveg = svfEveg
        self.svfSveg = svfSveg
        self.svfaveg = svfaveg
        self.svfNaveg = svfNaveg
        self.svfWaveg = svfWaveg
        self.svfEaveg = svfEaveg
        self.svfSaveg = svfSaveg
        self.vegdsm = vegdsm
        self.vegdsm2 = vegdsm2
        self.albedo_b = albedo_b
        self.absK = absK
        self.absL = absL
        self.ewall = ewall
        self.Fside = Fside
        self.Fup = Fup
        self.Fcyl = Fcyl
        self.altitude = altitude
        self.azimuth = azimuth
        self.zen = zen
        self.jday = jday
        self.usevegdem = usevegdem
        self.onlyglobal = onlyglobal
        self.buildings = buildings
        self.location = location
        self.psi = psi
        self.landcover = landcover
        self.lcgrid = lcgrid
        self.dectime = dectime
        self.altmax = altmax
        self.wallaspect = wallaspect
        self.wallheight = wallheight
        self.cyl = cyl
        self.elvis = elvis
        self.Ta = Ta
        self.RH = RH
        self.radG = radG
        self.radD = radD
        self.radI = radI
        self.P = P
        self.amaxvalue = amaxvalue
        self.bush = bush
        self.Twater = Twater
        self.TgK = TgK
        self.Tstart = Tstart
        self.alb_grid = alb_grid
        self.emis_grid = emis_grid
        self.TgK_wall = TgK_wall
        self.Tstart_wall = Tstart_wall
        self.TmaxLST = TmaxLST
        self.TmaxLST_wall = TmaxLST_wall
        self.first = first
        self.second = second
        self.svfalfa = svfalfa
        self.svfbuveg = svfbuveg
        self.firstdaytime = firstdaytime
        self.timeadd = timeadd
        self.timeaddE = timeaddE
        self.timeaddS = timeaddS
        self.timeaddW = timeaddW
        self.timeaddN = timeaddN
        self.timestepdec = timestepdec
        self.Tgmap1 = Tgmap1
        self.Tgmap1E = Tgmap1E
        self.Tgmap1S = Tgmap1S
        self.Tgmap1W = Tgmap1W
        self.Tgmap1N = Tgmap1N
        self.CI = CI
        self.YYYY = YYYY
        self.DOY = DOY
        self.hours = hours
        self.minu = minu
        self.gdal_dsm = gdal_dsm
        self.folderPath = folderPath
        self.poisxy = poisxy
        self.poiname = poiname
        self.Ws = Ws
        self.mbody = mbody
        self.age = age
        self.ht = ht
        self.activity = activity
        self.clo = clo
        self.sex = sex
        self.sensorheight = sensorheight
        self.anisotropic_sky = anisotropic_sky
        self.diffsh = diffsh
        self.shmat = shmat
        self.vegshmat = vegshmat
        self.vbshvegshmat = vbshvegshmat
        self.asvf = asvf
        self.patch_option = patch_option
        self.write_other_files = write_other_files

    def run(self, parallel=True):
        ret = None
        self.tmrtplot = np.zeros((self.rows, self.cols))
        X = np.zeros((self.rows, self.cols))
        self.TgOut1 = np.zeros((self.rows, self.cols))
        # TgOut = np.zeros((self.rows, self.cols))
        if parallel:
            ###############################################################################
            # Setup the distributed client
            ###############################################################################
            # from dask.distributed import Client

            # If you have a remote cluster running Dask
            # client = Client('tcp://scheduler-address:8786')
            # client = Client('tcp://10.17.36.50:8786')
            # manager = multiprocessing.Manager()
            # tmrtplot = manager.list(np.zeros((self.rows, self.cols)))
            # with joblib.parallel_backend('loky'):
            # with Parallel(n_jobs=multiprocessing.cpu_count() - 1, prefer="threads") as parallel :
            data = {
                'TgOut1': self.TgOut1,
                'landcover': self.landcover,
                'dsm': self.dsm,
                'scale': self.scale,
                'rows': self.rows,
                'cols': self.cols,
                'svf': self.svf,
                'svfN': self.svfN,
                'svfW': self.svfW,
                'svfE': self.svfE,
                'svfS': self.svfS,
                'svfveg': self.svfveg,
                'svfNveg': self.svfNveg,
                'svfWveg': self.svfWveg,
                'svfEveg': self.svfEveg,
                'svfSveg': self.svfSveg,
                'svfaveg': self.svfaveg,
                'svfNaveg': self.svfNaveg,
                'svfWaveg': self.svfWaveg,
                'svfEaveg': self.svfEaveg,
                'svfSaveg': self.svfSaveg,
                'vegdsm': self.vegdsm,
                'vegdsm2': self.vegdsm2,
                'albedo_b': self.albedo_b,
                'absK': self.absK,
                'absL': self.absL,
                'ewall': self.ewall,
                'Fside': self.Fside,
                'Fup': self.Fup,
                'Fcyl': self.Fcyl,
                'altitude': self.altitude,
                'azimuth': self.azimuth,
                'zen': self.zen,
                'jday': self.jday,
                'usevegdem': self.usevegdem,
                'onlyglobal': self.onlyglobal,
                'buildings': self.buildings,
                'location': self.location,
                'psi': self.psi,
                'landcover': self.landcover,
                'lcgrid': self.lcgrid,
                'dectime': self.dectime,
                'altmax': self.altmax,
                'wallaspect': self.wallaspect,
                'wallheight': self.wallheight,
                'cyl': self.cyl,
                'elvis': self.elvis,
                'Ta': self.Ta,
                'RH': self.RH,
                'radG': self.radG,
                'radD': self.radD,
                'radI': self.radI,
                'P': self.P,
                'amaxvalue': self.amaxvalue,
                'bush': self.bush,
                'Twater': self.Twater,
                'TgK': self.TgK,
                'Tstart': self.Tstart,
                'alb_grid': self.alb_grid,
                'emis_grid': self.emis_grid,
                'TgK_wall': self.TgK_wall,
                'Tstart_wall': self.Tstart_wall,
                'TmaxLST': self.TmaxLST,
                'TmaxLST_wall': self.TmaxLST_wall,
                'first': self.first,
                'second': self.second,
                'svfalfa': self.svfalfa,
                'svfbuveg': self.svfbuveg,
                'firstdaytime': self.firstdaytime,
                'timeadd': self.timeadd,
                'timeaddE': self.timeaddE,
                'timeaddS': self.timeaddS,
                'timeaddW': self.timeaddW,
                'timeaddN': self.timeaddN,
                'timestepdec': self.timestepdec,
                'Tgmap1': self.Tgmap1,
                'Tgmap1E': self.Tgmap1E,
                'Tgmap1S': self.Tgmap1S,
                'Tgmap1W': self.Tgmap1W,
                'Tgmap1N': self.Tgmap1N,
                'CI': self.CI,
                'YYYY': self.YYYY,
                'DOY': self.DOY,
                'hours': self.hours,
                'minu': self.minu,
                'RasterYSize': self.gdal_dsm.RasterYSize,
                'RasterXSize': self.gdal_dsm.RasterXSize,
                'GetGeoTransform': self.gdal_dsm.GetGeoTransform(),
                'GetProjection': self.gdal_dsm.GetProjection(),
                'folderPath': self.folderPath,
                'poisxy': self.poisxy,
                'poiname': self.poiname,
                'Ws': self.Ws,
                'mbody': self.mbody,
                'age': self.age,
                'ht': self.ht,
                'activity': self.activity,
                'clo': self.clo,
                'sex': self.sex,
                'sensorheight': self.sensorheight,
                'anisotropic_sky': self.anisotropic_sky,
                'diffsh': self.diffsh,
                'shmat': self.shmat,
                'vegshmat': self.vegshmat,
                'vbshvegshmat': self.vbshvegshmat,
                'asvf': self.asvf,
                'patch_option': self.patch_option,
            }
            # temp_folder = tempfile.mkdtemp()
            # for k, v in test.items():
            #     print(k, v)
            #     filename = os.path.join(temp_folder, f'{k}.mmap')
            #     if os.path.exists(filename): os.unlink(filename)
            #     from joblib import load, dump
            #     _ = dump(v, filename)

            # If you have a remote cluster running Dask
            # client = Client('tcp://scheduler-address:8786')

            # If you want Dask to set itself up on your personal computer
            # client = Client(processes=False)
            # client = Client('tcp://10.17.36.50:8786')
            # client = Client(address='tcp://10.17.36.50:8786')  # True doesn't work (Error)
            # client = Client(threads_per_worker=1, n_workers=5, processes=True)  # True doesn't work (Error)

            manager = multiprocessing.Manager()
            tmrt_list_shared = manager.list()
            with joblib.parallel_backend(backend='multiprocessing'):
                parallel = Parallel(
                    n_jobs=multiprocessing.cpu_count() - 1, prefer='processes'
                )
                parallel(
                    delayed(calculate_tmrt_with_parallel)(i, data, tmrt_list_shared)
                    for i in range(self.Ta.__len__())
                )
            self.tmrtplot = sum(tmrt_list_shared) / self.Ta.__len__()
        else:
            self.calculate_tmrt()

        solweigresult = {'tmrtplot': self.tmrtplot, 'altitude': self.altitude}

        if self.killed is False:
            ret = solweigresult
        return ret

    def calculate_tmrt(self):
        try:
            dsm = self.dsm
            scale = self.scale
            rows = self.rows
            cols = self.cols
            svf = self.svf
            svfN = self.svfN
            svfW = self.svfW
            svfE = self.svfE
            svfS = self.svfS
            svfveg = self.svfveg
            svfNveg = self.svfNveg
            svfWveg = self.svfWveg
            svfEveg = self.svfEveg
            svfSveg = self.svfSveg
            svfaveg = self.svfaveg
            svfNaveg = self.svfNaveg
            svfWaveg = self.svfWaveg
            svfEaveg = self.svfEaveg
            svfSaveg = self.svfSaveg
            vegdsm = self.vegdsm
            vegdsm2 = self.vegdsm2
            albedo_b = self.albedo_b
            absK = self.absK
            absL = self.absL
            ewall = self.ewall
            Fside = self.Fside
            Fup = self.Fup
            Fcyl = self.Fcyl
            altitude = self.altitude
            azimuth = self.azimuth
            zen = self.zen
            jday = self.jday
            usevegdem = self.usevegdem
            onlyglobal = self.onlyglobal
            buildings = self.buildings
            location = self.location
            psi = self.psi
            landcover = self.landcover
            lcgrid = self.lcgrid
            dectime = self.dectime
            altmax = self.altmax
            wallaspect = self.wallaspect
            wallheight = self.wallheight
            cyl = self.cyl
            elvis = self.elvis
            Ta = self.Ta
            RH = self.RH
            radG = self.radG
            radD = self.radD
            radI = self.radI
            P = self.P
            amaxvalue = self.amaxvalue
            bush = self.bush
            Twater = self.Twater
            TgK = self.TgK
            Tstart = self.Tstart
            alb_grid = self.alb_grid
            emis_grid = self.emis_grid
            TgK_wall = self.TgK_wall
            Tstart_wall = self.Tstart_wall
            TmaxLST = self.TmaxLST
            TmaxLST_wall = self.TmaxLST_wall
            first = self.first
            second = self.second
            svfalfa = self.svfalfa
            svfbuveg = self.svfbuveg
            firstdaytime = self.firstdaytime
            timeadd = self.timeadd
            timeaddE = self.timeaddE
            timeaddS = self.timeaddS
            timeaddW = self.timeaddW
            timeaddN = self.timeaddN
            timestepdec = self.timestepdec
            Tgmap1 = self.Tgmap1
            Tgmap1E = self.Tgmap1E
            Tgmap1S = self.Tgmap1S
            Tgmap1W = self.Tgmap1W
            Tgmap1N = self.Tgmap1N
            CI = self.CI
            YYYY = self.YYYY
            DOY = self.DOY
            hours = self.hours
            minu = self.minu
            gdal_dsm = self.gdal_dsm
            folderPath = self.folderPath
            poisxy = self.poisxy
            poiname = self.poiname
            Ws = self.Ws
            mbody = self.mbody
            age = self.age
            ht = self.ht
            activity = self.activity
            clo = self.clo
            sex = self.sex
            sensorheight = self.sensorheight
            anisotropic_sky = self.anisotropic_sky
            diffsh = self.diffsh
            shmat = self.shmat
            vegshmat = self.vegshmat
            vbshvegshmat = self.vbshvegshmat
            asvf = self.asvf
            patch_option = self.patch_option

            # numformat = '%d %d %d %d %.5f ' + '%.2f ' * 28

            numformat = (
                '%d %d %d %d %.5f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f '
                '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
            )

            for i in np.arange(0, Ta.__len__()):
                # Daily water body temperature
                if self.landcover == 1:
                    if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
                        Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])
                # print(dectime[i])
                # Nocturnal cloudfraction from Offerle et al. 2003
                if (dectime[i] - np.floor(dectime[i])) == 0:
                    daylines = np.where(np.floor(dectime) == dectime[i])
                    if daylines.__len__() > 1:
                        alt = altitude[0][daylines]
                        alt2 = np.where(alt > 1)
                        rise = alt2[0][0]
                        [_, CI, _, _, _] = clearnessindex_2013b(
                            zen[0, i + rise + 1],
                            jday[0, i + rise + 1],
                            Ta[i + rise + 1],
                            RH[i + rise + 1] / 100.0,
                            radG[i + rise + 1],
                            location,
                            P[i + rise + 1],
                        )  # i+rise+1 to match matlab code. correct?
                        if (CI > 1.0) or (CI == np.inf):
                            CI = 1.0
                    else:
                        CI = 1.0

                (
                    Tmrt,
                    Kdown,
                    Kup,
                    Ldown,
                    Lup,
                    Tg,
                    ea,
                    esky,
                    I0,
                    CI,
                    shadow,
                    firstdaytime,
                    timestepdec,
                    timeadd,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    Keast,
                    Ksouth,
                    Kwest,
                    Knorth,
                    Least,
                    Lsouth,
                    Lwest,
                    Lnorth,
                    KsideI,
                    TgOut1,
                    TgOut,
                    radIout,
                    radDout,
                    Lside,
                    Lsky_patch_characteristics,
                    CI_Tg,
                    CI_TgG,
                    KsideD,
                    dRad,
                    Kside,
                ) = so.Solweig_2022a_calc(
                    i,
                    dsm,
                    scale,
                    rows,
                    cols,
                    svf,
                    svfN,
                    svfW,
                    svfE,
                    svfS,
                    svfveg,
                    svfNveg,
                    svfEveg,
                    svfSveg,
                    svfWveg,
                    svfaveg,
                    svfEaveg,
                    svfSaveg,
                    svfWaveg,
                    svfNaveg,
                    vegdsm,
                    vegdsm2,
                    albedo_b,
                    absK,
                    absL,
                    ewall,
                    Fside,
                    Fup,
                    Fcyl,
                    altitude[0][i],
                    azimuth[0][i],
                    zen[0][i],
                    jday[0][i],
                    usevegdem,
                    onlyglobal,
                    buildings,
                    location,
                    psi[0][i],
                    landcover,
                    lcgrid,
                    dectime[i],
                    altmax[0][i],
                    wallaspect,
                    wallheight,
                    cyl,
                    elvis,
                    Ta[i],
                    RH[i],
                    radG[i],
                    radD[i],
                    radI[i],
                    P[i],
                    amaxvalue,
                    bush,
                    Twater,
                    TgK,
                    Tstart,
                    alb_grid,
                    emis_grid,
                    TgK_wall,
                    Tstart_wall,
                    TmaxLST,
                    TmaxLST_wall,
                    first,
                    second,
                    svfalfa,
                    svfbuveg,
                    firstdaytime,
                    timeadd,
                    timestepdec,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    CI,
                    self.TgOut1,
                    diffsh,
                    shmat,
                    vegshmat,
                    vbshvegshmat,
                    anisotropic_sky,
                    asvf,
                    patch_option,
                )

                # Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
                #    Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
                #    Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radIout, radDout = so.Solweig_2021a_calc(
                #         i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg,
                #         svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg,
                #         vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude[0][i],
                #         azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location,
                #         psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], wallaspect,
                #         wallheight, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i], amaxvalue,
                #         bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                #         TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec,
                #         Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani)

                # Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
                # Tgmap1, timeaddE, Tgmap1E, timeaddS, Tgmap1S, timeaddW, Tgmap1W, timeaddN, Tgmap1N, \
                # Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radIout, radDout \
                #     = so.Solweig_2019a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg,
                #         svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg,
                #         vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude[0][i],
                #         azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location,
                #         psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], wallaspect,
                #         wallheight, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i], amaxvalue,
                #         bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                #         TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timeaddE, timeaddS,
                #         timeaddW, timeaddN, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani)

                self.tmrtplot = self.tmrtplot + Tmrt

                if altitude[0][i] > 0:
                    w = 'D'
                else:
                    w = 'N'

                # Write to POIs
                if poisxy is not None:
                    for k in range(0, self.poisxy.shape[0]):
                        poi_save = np.zeros((1, 35))
                        poi_save[0, 0] = YYYY[0][i]
                        poi_save[0, 1] = jday[0][i]
                        poi_save[0, 2] = hours[i]
                        poi_save[0, 3] = minu[i]
                        poi_save[0, 4] = dectime[i]
                        poi_save[0, 5] = altitude[0][i]
                        poi_save[0, 6] = azimuth[0][i]
                        poi_save[0, 7] = radIout
                        poi_save[0, 8] = radDout
                        poi_save[0, 9] = radG[i]
                        poi_save[0, 10] = Kdown[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 11] = Kup[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 12] = Keast[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 13] = Ksouth[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 14] = Kwest[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 15] = Knorth[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 16] = Ldown[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 17] = Lup[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 18] = Least[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 19] = Lsouth[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 20] = Lwest[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 21] = Lnorth[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 22] = Ta[i]
                        poi_save[0, 23] = TgOut[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 24] = RH[i]
                        poi_save[0, 25] = esky
                        poi_save[0, 26] = Tmrt[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 27] = I0
                        poi_save[0, 28] = CI
                        poi_save[0, 29] = shadow[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 30] = svf[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 31] = svfbuveg[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        poi_save[0, 32] = KsideI[int(poisxy[k, 2]), int(poisxy[k, 1])]
                        # Recalculating wind speed based on powerlaw
                        WsPET = (1.1 / sensorheight) ** 0.2 * Ws[i]
                        WsUTCI = (10.0 / sensorheight) ** 0.2 * Ws[i]
                        resultPET = p._PET(
                            Ta[i],
                            RH[i],
                            Tmrt[int(poisxy[k, 2]), int(poisxy[k, 1])],
                            WsPET,
                            mbody,
                            age,
                            ht,
                            activity,
                            clo,
                            sex,
                        )
                        poi_save[0, 33] = resultPET
                        resultUTCI = utci.utci_calculator(
                            Ta[i],
                            RH[i],
                            Tmrt[int(poisxy[k, 2]), int(poisxy[k, 1])],
                            WsUTCI,
                        )
                        poi_save[0, 34] = resultUTCI
                        data_out = (
                            self.folderPath[0] + '/POI_' + str(self.poiname[k]) + '.txt'
                        )
                        # f_handle = file(data_out, 'a')
                        f_handle = open(data_out, 'ab')
                        np.savetxt(f_handle, poi_save, fmt=numformat)
                        f_handle.close()

                if hours[i] < 10:
                    XH = '0'
                else:
                    XH = ''
                if minu[i] < 10:
                    XM = '0'
                else:
                    XM = ''

                saveraster(
                    gdal_dsm,
                    os.path.join(
                        folderPath,
                        'Tmrt_'
                        + str(int(YYYY[0, i]))
                        + '_'
                        + str(int(DOY[i]))
                        + '_'
                        + XH
                        + str(int(hours[i]))
                        + XM
                        + str(int(minu[i]))
                        + w
                        + '.tif',
                    ),
                    Tmrt,
                )
                saveraster(
                    gdal_dsm,
                    os.path.join(
                        folderPath,
                        'Shadow_'
                        + str(int(YYYY[0, i]))
                        + '_'
                        + str(int(DOY[i]))
                        + '_'
                        + XH
                        + str(int(hours[i]))
                        + XM
                        + str(int(minu[i]))
                        + w
                        + '.tif',
                    ),
                    shadow,
                )

                if self.write_other_files:
                    saveraster(
                        gdal_dsm,
                        os.path.join(
                            folderPath,
                            'Kup_'
                            + str(int(YYYY[0, i]))
                            + '_'
                            + str(int(DOY[i]))
                            + '_'
                            + XH
                            + str(int(hours[i]))
                            + XM
                            + str(int(minu[i]))
                            + w
                            + '.tif',
                        ),
                        Kup,
                    )
                    saveraster(
                        gdal_dsm,
                        os.path.join(
                            folderPath,
                            'Kdown_'
                            + str(int(YYYY[0, i]))
                            + '_'
                            + str(int(DOY[i]))
                            + '_'
                            + XH
                            + str(int(hours[i]))
                            + XM
                            + str(int(minu[i]))
                            + w
                            + '.tif',
                        ),
                        Kdown,
                    )
                    saveraster(
                        gdal_dsm,
                        os.path.join(
                            folderPath,
                            'Lup_'
                            + str(int(YYYY[0, i]))
                            + '_'
                            + str(int(DOY[i]))
                            + '_'
                            + XH
                            + str(int(hours[i]))
                            + XM
                            + str(int(minu[i]))
                            + w
                            + '.tif',
                        ),
                        Lup,
                    )
                    saveraster(
                        gdal_dsm,
                        os.path.join(
                            folderPath,
                            'Ldown_'
                            + str(int(YYYY[0, i]))
                            + '_'
                            + str(int(DOY[i]))
                            + '_'
                            + XH
                            + str(int(hours[i]))
                            + XM
                            + str(int(minu[i]))
                            + w
                            + '.tif',
                        ),
                        Ldown,
                    )

            self.tmrtplot = (
                self.tmrtplot / Ta.__len__()
            )  # fix average Tmrt instead of sum, 20191022
        except Exception:
            errorstring = self.print_exception()
            print(errorstring)

    def print_exception(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        return 'EXCEPTION IN {}, \nLINE {} "{}" \nERROR MESSAGE: {}'.format(
            filename, lineno, line.strip(), exc_obj
        )

    def kill(self):
        self.killed = True
