# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: wang.q@mail.hnust.edu.cn
@Software: PyCharm
@File: MBG.R_Parallel_Year.py
@Time: 17/08/2022 13:37
@Function: 
"""

# Note -----------------------------------------------------------------------------------------------------------------
# - MBG.R               Retrofit of Mass Balance Gradient Model
# - Parallel            Multiprocess
# - Year                Year Scale


# Pkg mgr ==============================================================================================================
from glob import glob
import multiprocessing
import geopandas as gpd
import shapely
import warnings
from scipy import optimize
import numpy as np
import pandas as pd
import rasterio as rio
# import time
import scipy.stats as stats
import itertools
from tqdm import trange
import os
import xarray as xr

warnings.filterwarnings('ignore')

# def func =============================================================================================================
def Rseq(min_num, max_num, steps):
    seq_ls = np.arange(min_num, max_num, steps)
    if not np.allclose(seq_ls[-1], max_num):
        seq_ls = np.append(seq_ls, max_num)
    return seq_ls.tolist()

def as_vector(vector_ras):
    ras_ls = vector_ras.ReadAsArray().flatten()
    return ras_ls

def Reclassify(ras_arr, span):
    for iter, item in enumerate(span[0: -1]):
        if iter == 0:
            ras_arr = np.where((span[iter] <= ras_arr) & (ras_arr <= span[iter + 1]), iter + 1, ras_arr)
        else:
            ras_arr = np.where((span[iter] < ras_arr) & (ras_arr <= span[iter + 1]), iter + 1, ras_arr)
    return ras_arr

def Rzonal(zone_arr, stat_ras):
    zonal = []
    for i in np.unique(zone_arr):
        if ~np.isnan(i):
            zonalstat = np.nanmean(stat_ras[zone_arr == i])
            zonal.append(zonalstat)
    return zonal

# Setting ==============================================================================================================
# set env ---
mbg_input = r'G:/Project_MBG/MBG_Input'
glc_input = mbg_input + r'/RGI_Glc'
mbg_output = r'G:/Project_MBG/MBG_Output'

# Load and prep general data ===========================================================================================
# read the sampled mean ostrem curve
ostrem = pd.read_csv(mbg_input + r'/Ostrem.csv')
rgi_his_pr = pd.read_csv(mbg_input + r'/Meteor_historical_pr.csv', index_col=0)
rgi_his_tas = pd.read_csv(mbg_input + r'/Meteor_historical_tas_elr.csv', index_col=0)

def mass_balance_gradient(cur_glc):

    # cur_glc = rgi_pt.loc[0, :]

    # set_df = pd.read_csv(mbg_output + r'/%s/settings.csv' % cur_glc['RGIId'])

    # if set_df['gradient'][0] == 0.006:

    try:

        # debug ========================================================================================================

        # cur_glc = rgi_pt.loc[1020, :]
        # cur_glc = rgi_pt.loc[10724, :]
        # cur_glc = rgi_pt.loc[6893, :]
        print('Initialize %s ...' % cur_glc['RGIId'])

        rgi_pr = pd.read_csv(glc_input + r'/%s/pr.csv' % cur_glc['RGIId'], index_col=0)
        rgi_tas = pd.read_csv(glc_input + r'/%s/tas_elr.csv' % cur_glc['RGIId'], index_col=0)

        # Input parameter frame ----------------------------------------------------------------------------------------
        # initialize parameter data frame
        gcm = ['%s_%s' % (rgi_pr['model'][i], rgi_pr['SSP'][i]) for i in range(rgi_pr.shape[0])]
        gcm_ind = range(len(gcm))

        settings = pd.DataFrame({'label': gcm,
                                 'gradient': 0,
                                 'pr.ref': cur_glc['Pr_ref'],
                                 'dd.ref': cur_glc['DD_ref'],  # revise to pr col cur_glc['DD_ref']
                                 'ddf.snow': cur_glc['DDF_snow'],
                                 'ddf.cln': cur_glc['DDF_ice'],
                                 'debthick': 0,
                                 # 'obs.mb.rnd': 0
                                 })

        # Monte Carlo parameters set initialisation --------------------------------------------------------------------
        # initiate the random sample for the monte carlo
        mod_sample = 10
        sample_n = len(gcm) * mod_sample
        ddf_cln_rand, ddf_snow_rand, ddf_deb_rand, dd_ref_rand = np.random.rand(sample_n), np.random.rand(
            sample_n), np.random.rand(sample_n), np.random.rand(sample_n)
        dP_rand, debthick_rand = np.random.rand(sample_n), np.random.rand(sample_n)
        # obs_mb_rand = np.random.rand(sample_n)

        # DDF clean ice
        m = cur_glc['DDF_ice']
        s = 1.15
        a, b = 0, np.inf
        ddf_cln_set = stats.truncnorm.ppf(ddf_cln_rand, a=(a - m) / s, b=(b - m) / s, loc=m, scale=s)  # [0, +∞)

        # DDF snow
        m = cur_glc['DDF_snow']
        s = 1.35
        a, b = 0, np.inf
        ddf_snow_set = stats.truncnorm.ppf(ddf_snow_rand, a=(a - m) / s, b=(b - m) / s, loc=m, scale=s)  # [0, +∞)

        # Degree days
        if cur_glc['DD_ref'] < 10:
            m = cur_glc['DD_ref']
            s = 3.33
            a, b = 0, 10
            dd_ref_set = stats.truncnorm.ppf(dd_ref_rand, a=(a - m) / s, b=(b - m) / s, loc=m, scale=s)  # [0, +∞)
        else:
            m = cur_glc['DD_ref']
            s = cur_glc['err_DD_ref']  # set correspond col
            dd_ref_set = stats.norm.ppf(dd_ref_rand, m, s)

        # delta pr
        m = 1
        s = cur_glc['err_Pr_ref'] / cur_glc['Pr_ref']
        a, b = 0, 2  # sigma with correspond col
        dP_set = stats.truncnorm.ppf(dP_rand, a=(a - m) / s, b=(b - m) / s, loc=m, scale=s)  # [0, 2.0]
        dP_set[dP_set < 1.0] = 1 / (2 - dP_set[dP_set < 1.0])
        dP_set[dP_set > 1.0] = 2 - (1 / dP_set[dP_set > 1.0])

        # mass balance offset
        if cur_glc['Ref_Year'] > 2009:
            obs_mb, obs_mb_err = cur_glc['2010s'], cur_glc['err_2010s']
        else:
            obs_mb, obs_mb_err = cur_glc['2000s'], cur_glc['err_2000s']

        if (obs_mb <= 0) & (cur_glc['DD_ref'] < 10):
            for stds in range(4):
                obs_mb += obs_mb_err
                if obs_mb > 0:
                    break
        #     m = obs_mb
        #     s = obs_mb_err
        #     obs_mb_set = stats.norm.ppf(obs_mb_rand, m, s)
        #     obs_mb_set = np.where(obs_mb_set < -obs_mb, abs(obs_mb_set), obs_mb_set)
        # else:
        #     m = obs_mb
        #     s = obs_mb_err
        #     obs_mb_set = stats.norm.ppf(obs_mb_rand, m, s)


        # debris thickness set (for 95th percentile LST)            # debris error
        m = 0
        s = 0.06
        debthick_set = stats.norm.ppf(debthick_rand, m, s)  # [0.01, 5.0]

        # add monte carlo to settings frame
        setting_size = settings.shape[0]
        for t in range(sample_n):
            settings.loc[settings.shape[0]] = settings.loc[0, :]
        mc_ind = [i + setting_size for i in range(sample_n)]
        gcm_sample = list(itertools.chain.from_iterable(itertools.repeat(x, mod_sample) for x in range(len(gcm))))
        settings.loc[mc_ind, 'label'] = ['MC{0} {1}'.format(str(i), gcm[gcm_sample[i]]) for i in range(sample_n)]
        settings.loc[mc_ind, 'gradient'] = 0
        settings.loc[mc_ind, 'ddf.cln'] = ddf_cln_set
        settings.loc[mc_ind, 'ddf.snow'] = ddf_snow_set
        settings.loc[mc_ind, 'debthick'] = debthick_set
        settings.loc[mc_ind, 'dd.ref'] = dd_ref_set
        settings.loc[mc_ind, 'pr.ref'] = settings['pr.ref'][0] * dP_set
        # settings.loc[mc_ind, 'obs.mb.rnd'] = obs_mb_set

        # derivatives of settings
        # settings['obs.mb'] = obs_mb + settings['obs.mb.rnd']
        settings['obs.mb'] = obs_mb
        settings['acc.max'] = settings['pr.ref'] * 0.001
        settings['abl.max'] = (-settings['dd.ref'] * settings['ddf.cln'] +- cur_glc['Acc_ref'] * 1000 / settings[
            'ddf.snow']) * 0.001

        # if settings['abl.max'][0] > -1.0:

        mbg_output_rgi = mbg_output + r'/%s' % cur_glc['RGIId']

        if not os.path.exists(mbg_output_rgi):
            os.makedirs(mbg_output_rgi)

        settings['abl.max'] = [-0.5 if i > -0.5 else i for i in settings['abl.max']]
        print('Set Monte Carlo model random parameters has done.')

        # Read and process raster inputs ---------------------------------------------------------------------------
        with rio.open(glc_input + r'/%s/dem.tif' % cur_glc['RGIId']) as dem_ras:
            dem_arr = dem_ras.read()
            dem_meta = dem_ras.meta.copy()

            dem_vector = np.where(dem_arr == dem_meta['nodata'], np.nan, dem_arr)
            ddf_cls = np.where(~np.isnan(dem_vector), 1, np.nan)

        with rio.open(glc_input + r'/%s/dem_fill.tif' % cur_glc['RGIId']) as dem_fill_ras:
            dem_fill_arr = dem_fill_ras.read()
            # dem_fill_arr[dem_fill_ras.meta['nodata']] = np.nan
        #     mxs = np.ceil((dem_fill_ras.bounds.right - dem_fill_ras.bounds.left) / 30 + 1)
        #     mys = (dem_fill_ras.bounds.top - dem_fill_ras.bounds.bottom) / 30 + 1

        with rio.open(glc_input + r'/%s/thk.tif' % cur_glc['RGIId']) as thk_ras:
            thk_arr = thk_ras.read()

        with rio.open(glc_input + r'/%s/vel.tif' % cur_glc['RGIId']) as vx_ras:
            with rio.open(glc_input + r'/%s/vel.tif' % cur_glc['RGIId']) as vy_ras:
                vx_arr = vx_ras.read()
                vy_arr = vy_ras.read()

        if os.path.exists(glc_input + r'/%s/debris.tif' % cur_glc['RGIId']):
            with rio.open(glc_input + r'/%s/debris.tif' % cur_glc['RGIId']) as debris_ras:
                debris_arr = debris_ras.read()
                debris_meta = debris_ras.meta.copy()

                debris_arr[(debris_arr == 0) | (debris_arr == debris_meta['nodata'])] = np.nan  # validate nodata value
                ddf_cls[~np.isnan(debris_arr) & ~np.isnan(dem_vector)] = 2
        else:
            debris_arr = np.zeros_like(ddf_cls)

        # get dem stats
        # dem_ras_px = np.sum(~np.isnan(dem_vector))
        elev_min, elev_max, dem_vals = np.nanmin(dem_vector), np.nanmax(dem_vector), dem_vector[~np.isnan(dem_vector)]

        mb_gradient = np.array(np.nan).repeat(settings.shape[0])

        for g in trange(settings.shape[0], desc=r'Determine mbg'):

            red_arr = np.where(debris_arr > 0, (debris_arr + settings['debthick'][g]) * 100, 0)
            red_arr[red_arr < 0] = 0
            red_arr = np.interp(red_arr, ostrem['pred.x'], ostrem['value'])  # + debris thick error
            red_arr[ddf_cls == 1] = 1

            # ddf_ice = settings['ddf.cln'][g]
            # ddf_effective = Rzonal(band_arr, red_arr) * bmdf['debris'] + ddf_ice * bmdf['ice']

            # red_factor = 1 - (ddf_effective / ddf_ice)
            # bmdf['deb.reduct'] = red_factor

            abl_max_cur, acc_max_cur = settings['abl.max'][g], settings['acc.max'][g]
            obs_mb_cur, deb_red_cur = settings['obs.mb'][g], red_arr

            dx = dy = dem_meta['transform'][0]

            def gradopt(mbg):

                smb = (dem_vector - elev_min) * mbg + abl_max_cur
                smb[smb > acc_max_cur] = acc_max_cur

                thk0 = thk_arr.copy() + smb
                thk0[thk0 < 0] = 0

                fx = vx_arr * thk0
                fy = vy_arr * thk0

                dfx_dx = np.gradient(fx, dx, axis=1)
                dfy_dx = np.gradient(fy, dy, axis=0)

                div = dfx_dx + dfy_dx
                mb = thk0 + div
                mb[mb < 0] = 0

                mb_m3 = (mb[mb >= 0].sum() + (mb[mb < 0] * deb_red_cur[mb < 0]).sum())
                squareddiff = ((mb_m3 / mb[~np.isnan(mb)].size - obs_mb_cur) ** 2) + 1e-4 * mbg
                # print(squareddiff)
                return squareddiff

            # Fit gradient for current run only. Increase precip if ELA is below threshold, constrain correction upper limit
            if g == 0:

                mbfit, ela = np.array(-1), -1
                ela_thresh = np.percentile(dem_vals, 25)
                settings['pr.ref.orig'] = settings['pr.ref']

                # iter_time = 0

                while np.nanpercentile(mbfit, 95) < 0 or ela < ela_thresh:
                    # print(f'iter mbg {iter_time}...')
                    # iter_time += 1

                    curgrad = optimize.minimize_scalar(fun=gradopt, bounds=(-1, 1), method='bounded').x
                    mbfit = (dem_vector - elev_min) * curgrad + abl_max_cur
                    obs_mb_cur = obs_mb_cur * 0.9 if np.nanpercentile(mbfit, 95) < 0 else obs_mb_cur

                    def elaFinder(x):
                        return abs(settings['abl.max'][0] + curgrad * x)

                    ela_step = optimize.fminbound(func=elaFinder, x1=-1e5, x2=1e5)
                    ela = elev_min + ela_step
                    acc_max_cur = acc_max_cur * 1.1 if ela < ela_thresh else acc_max_cur  # increase precip iteratively with 10%

                    if acc_max_cur > 3:
                        acc_max_cur = 3
                        break  # limit corrected precip to 3000 mm

                # update settings and monte carlo members with optimized precip
                settings.loc[gcm_ind, 'pr.ref'] = acc_max_cur * 1000
                settings.loc[mc_ind, 'pr.ref'] = [settings['pr.ref'][mc_ind], dP_set * settings['pr.ref'][0]][
                    sample_n > 0]
                settings['acc.max'] = settings['pr.ref'] * 0.001

            # fit mb gradient for all projections and members
            # lower the observed mass balance until there is at least one band in the acc zone (i.e. force positive gradient)
            mbfit = np.array(-1)

            while np.nanpercentile(mbfit, 95) < 0:
                mb_gradient[g] = optimize.minimize_scalar(fun=gradopt, bounds=(-1, 1), method='bounded', options={'maxiter': 100}).x
                mbfit = (dem_vector - elev_min) * mb_gradient[g] + abl_max_cur
                obs_mb_cur = obs_mb_cur * 0.9 if np.nanpercentile(mbfit, 95) < 0 else obs_mb_cur

            settings.loc[g, 'obs.mb'] = obs_mb_cur
            # print('mbg of {0} is: {1}.'.format(settings['label'][g], mb_gradient[g]))

        # mb_gradient[mb_gradient > 0.006] = 0.006
        mb_gradient = pd.Series(mb_gradient, index=settings['label'])
        settings['gradient'] = mb_gradient.values
        ssp_label = [label.split('_')[-1] for label in settings['label']]
        settings['SSP'] = ssp_label
        settings.to_csv(mbg_output_rgi + r'/settings.csv', index=False)

        print('Calculate mass balance gradient has done.')

        # Get annual climate change forcing ----------------------------------------------------------------------------
        # calc the dp and dt for current glacier for all models
        # ref_tas, ref_pr = cur_glc['Tas_ref'], cur_glc['Pr_ref']
        ref_year = cur_glc['Ref_Year']
        bgn_year = 2015
        end_year = 2100

        dT = rgi_tas.loc[:, str(bgn_year): str(end_year)].copy()
        dP = rgi_pr.loc[:, str(bgn_year): str(end_year)].copy()
        dT_his = pd.DataFrame(
            np.tile(rgi_his_tas.loc[cur_glc['RGIId'], str(ref_year): str(bgn_year - 1)].values, (dT.shape[0], 1)),
            columns=[str(yi) for yi in range(ref_year, bgn_year)])
        dP_his = pd.DataFrame(
            np.tile(rgi_his_pr.loc[cur_glc['RGIId'], str(ref_year): str(bgn_year - 1)].values, (dT.shape[0], 1)),
            columns=[str(yi) for yi in range(ref_year, bgn_year)])

        dT = pd.concat([dT_his, dT], axis=1)
        dP = pd.concat([dP_his, dP], axis=1)

        dT_origin = dT.copy()
        dP_origin = dP.copy()

        dT = dT - cur_glc['T2m_ref']
        dP = dP / cur_glc['Pr_ref']

        dT.insert(0, 'model', rgi_tas['model'])
        dP.insert(0, 'model', rgi_tas['model'])
        dT.insert(0, 'SSP', rgi_tas['SSP'])
        dP.insert(0, 'SSP', rgi_tas['SSP'])

        dT_origin.insert(0, 'model', rgi_tas['model'])
        dP_origin.insert(0, 'model', rgi_tas['model'])
        dT_origin.insert(0, 'SSP', rgi_tas['SSP'])
        dP_origin.insert(0, 'SSP', rgi_tas['SSP'])

        # ref_year = 2005
        # end_year = 2100
        #
        # dT = rgi_tas.copy()
        # dP = rgi_pr.copy()
        #
        # dT.loc[:, str(ref_year):] = dT.loc[:, str(ref_year):] - cur_glc['T2m_ref']
        # dP.loc[:, str(ref_year):] = dP.loc[:, str(ref_year):] / cur_glc['Pr_ref']

        df_abl_max = pd.DataFrame(index=settings['label'])
        df_acc_max = pd.DataFrame(index=settings['label'])
        df_ela = pd.DataFrame(index=settings['label'])

        df_abl_max['SSP'] = ssp_label
        df_acc_max['SSP'] = ssp_label
        df_ela['SSP'] = ssp_label

        # get the current ELA from fitted mb gradient, i.e. the zero mass balance crossing
        def elaFinder(x):
            return abs(settings['abl.max'][0] + mb_gradient[0] * x)

        ela_step = optimize.fminbound(func=elaFinder, x1=-1e5, x2=1e5)
        ela = elev_min + ela_step

        # get the ela for current timestep
        deladt_lims = [56.5, 202.5]  # limit to 90pct range of Shea and Immerzeel
        deladt = np.array(16.46 * (settings['pr.ref.orig'] - 379.61) ** 0.32)
        deladt = np.where((deladt < deladt_lims[0]) | np.isnan(deladt), deladt_lims[0], deladt)
        deladt = np.where((deladt > deladt_lims[1]) | np.isnan(deladt), deladt_lims[1], deladt)

        t_ser = range(ref_year, end_year + 1)

        for t in trange(len(t_ser), desc=r'Loop ts'):
            t = str(t_ser[t])

            # Yearly CC and derivs ---------
            # keep climate stable if model is run beyond 2100
            dP_t, dT_t = np.array(dP.loc[:, t]), np.array(dT.loc[:, t])
            dP_t, dT_t = np.append(dP_t, dP_t.repeat(mod_sample)), np.append(dT_t, dT_t.repeat(mod_sample))

            # dela/dt
            ela_t = dT_t * deladt

            # Calculate accumulation and ablation ---------
            accmax_t = np.array(settings['pr.ref'] * 0.001 * dP_t)
            ablmax_t = np.array(settings['abl.max'].values - mb_gradient * ela_t)

            df_ela[t] = np.where((ela_t + ela) > elev_max, elev_max, (ela_t + ela))
            df_ela[t] = np.where((ela_t + ela) < elev_min, elev_min, (ela_t + ela))
            df_abl_max[t] = ablmax_t
            df_acc_max[t] = accmax_t

        # sns.heatmap(df_abl_max)
        # sns.heatmap(df_acc_max)

        # df_abl_max_ssp = df_abl_max.groupby('SSP').mean()
        # df_acc_max_ssp = df_acc_max.groupby('SSP').mean()
        # df_ela_ssp = df_ela.groupby('SSP').mean()
        # mb_gradient_ssp = settings.groupby('SSP').mean()['gradient']
        #
        # df_rgi_tas_ssp = dT_origin.groupby('SSP').mean()
        # debris_ssp = settings.groupby('SSP').mean()['debthick']

        print('Calculate mass balance parameters till EOC has done.')

        # loop mb ----------------------------------------------------------------------------------------------------------
        # export pism init ---
        with xr.open_dataset(glc_input + r'/%s/dem_fill.tif' % cur_glc['RGIId']) as ds:
            proj_x, proj_y = ds['x'].data, ds['y'].data

        # scale_ds = xr.Dataset()

        # scale_ds.coords['time'] = np.array([0])
        # scale_ds.coords['x'] = proj_x
        # scale_ds.coords['y'] = proj_y

        df_mb = pd.DataFrame()
        settings.index = settings['label']

        for scene in settings.index:

            spatial_ds = xr.Dataset()

            mb_ser = np.zeros((len(t_ser), ddf_cls.shape[1], ddf_cls.shape[2]))
            tas_ser = np.zeros((len(t_ser), ddf_cls.shape[1], ddf_cls.shape[2]))

            for t in trange(len(t_ser), desc='Loop mb(%s)' % scene):
                # debug year ---
                t_iter = t
                t = str(t_ser[t])

                elr = cur_glc[t]
                tas = dT_origin[t][scene]
                elr_span = (dem_fill_arr - cur_glc['min_Elev']) / 1000
                elr_delta = elr * elr_span
                tas_arr = np.full_like(dem_fill_arr, tas)
                tas_arr = tas_arr + elr_delta
                tas_ser[t_iter, :, :] = tas_arr

                red_arr = np.where(debris_arr > 0, (debris_arr + settings.loc[scene, 'debthick']) * 100, 0)
                red_arr[red_arr < 0] = 0
                red_arr = np.interp(red_arr, ostrem['pred.x'], ostrem['value'])
                red_arr[ddf_cls == 1] = 1

                mb_arr = ((dem_vector - elev_min) * settings.loc[scene, 'gradient'] + df_abl_max[t][scene]) * red_arr
                mb_arr[mb_arr > df_acc_max[t][scene]] = df_acc_max[t][scene]
                mb_ser[t_iter, :, :] = mb_arr

                df_mb.loc[scene, t] = mb_arr[~np.isnan(mb_arr)].mean()

            mb_ser[np.isnan(mb_ser)] = -9999

            # pism spatial ---
            ice_loss_ser = np.where(mb_ser != -9999, mb_ser * 1e3, -9999)  # kg m-2 yr-1

            spatial_ds['climatic_mass_balance'] = (('time', 'y', 'x'), ice_loss_ser)
            spatial_ds['ice_surface_temp'] = (('time', 'y', 'x'), tas_ser)

            cf_ser = list(range(len(t_ser)))

            spatial_ds.coords['time'] = cf_ser
            spatial_ds.coords['x'] = proj_x
            spatial_ds.coords['y'] = proj_y

            spatial_ds['time'].attrs = {'calender': 'none',
                                        'long_name': 'Time',
                                        'standard_name': 'time',
                                        'units': f'years since {ref_year}-1-1'}

            spatial_ds['climatic_mass_balance'].attrs = {'long_name': 'Surface Mass Balance',
                                                         'standard_name': 'land_ice_surface_specific_mass_balance_flux',
                                                         'units': 'kg m-2 year-1'}

            spatial_ds['x'].attrs = {'long_name': 'Cartesian x-coordinate',
                                     'standard_name': 'projection_x_coordinate',
                                     'units': 'meters'}

            spatial_ds['y'].attrs = {'long_name': 'Cartesian y-coordinate',
                                     'standard_name': 'projection_y_coordinate',
                                     'units': 'meters'}

            spatial_ds['ice_surface_temp'].attrs = {'long_name': 'Annual Mean Air Temperature (2 meter)',
                                                    'standard_name': 'air_temperature',
                                                    'units': 'Celsius'}

            # spatial_ds.attrs = {'proj': rio.crs.CRS.from_wkt(ds.spatial_ref.attrs['crs_wkt']).to_proj4()}
            spatial_ds = spatial_ds.sortby('y')

            spatial_ds.rio.write_crs(rio.crs.CRS.from_wkt(ds.spatial_ref.attrs['crs_wkt']).to_string()). \
                to_netcdf(mbg_output + r'/%s/%s.nc' % (cur_glc['RGIId'], scene))

        # scale nc
        print('Calc coord parameters of %s ...' % cur_glc['RGIId'])

        # scale_ds['topg'] = (('time', 'y', 'x'), dem_fill_arr - thk_arr)
        # scale_ds['thk'] = (('time', 'y', 'x'), thk_arr)

        '''
        lon_lat = np.stack((np.tile(proj_x, (proj_y.size, 1)), np.tile(proj_y, (proj_x.size, 1)).T), axis=0)

        def proj2gcs(z_ser):
            geo_ser = gpd.GeoSeries(shapely.geometry.Point(z_ser),
                                    crs=rio.crs.CRS.from_wkt(ds.spatial_ref.attrs['crs_wkt']).to_string())
            z_lon, z_lat = geo_ser.to_crs('epsg:4326').loc[0].xy

            return np.array([z_lon[0], z_lat[0]])

        lon_lat = np.apply_along_axis(proj2gcs, 0, lon_lat)
        lon_arr, lat_arr = np.zeros_like(dem_fill_arr), np.zeros_like(dem_fill_arr)
        lon_arr[0, :, :], lat_arr[0, :, :] = lon_lat[0], lon_lat[1]
        
        scale_ds['lon'] = (('time', 'y', 'x'), lon_arr)
        scale_ds['lat'] = (('time', 'y', 'x'), lat_arr)

        scale_ds['no_model_mask'] = (('time', 'y', 'x'), np.where(thk_arr == 0, 1, 0))

        scale_ds['time'].attrs = spatial_ds['time'].attrs
        scale_ds['x'].attrs = spatial_ds['x'].attrs
        scale_ds['y'].attrs = spatial_ds['y'].attrs
        # scale_ds.attrs = spatial_ds.attrs

        scale_ds['no_model_mask'].attrs = {'units': '',
                                           'flag_meanings': 'normal special_treatment',
                                           'long_name': 'mask: zeros (modeling domain) and ones (no-model buffer near grid edges)',
                                           'pism_intent': 'model_state',
                                           'flag_values': np.array([0, 1])}

        scale_ds['lat'].attrs = {'long_name': 'Latitude',
                                 'standard_name': 'latitude',
                                 'units': 'degreeN'}

        scale_ds['lon'].attrs = {'long_name': 'Longitude',
                                 'standard_name': 'longitude',
                                 'units': 'degreeE'}

        scale_ds['topg'].attrs = {'long_name': 'Bedrock Topography',
                                  'standard_name': 'bedrock_altitude',
                                  'units': 'meters'}

        scale_ds['thk'].attrs = {'long_name': 'Ice Thickness',
                                 'standard_name': 'land_ice_thickness',
                                 'units': 'meters'}

        scale_ds = scale_ds.sortby('y')
        scale_ds.rio.write_crs(rio.crs.CRS.from_wkt(ds.spatial_ref.attrs['crs_wkt']).to_string()). \
            to_netcdf(mbg_output + r'/%s/init_field.nc' % cur_glc['RGIId'])
        '''

        print('Calculate mass balance till EOC has done.')

        df_ela.to_csv(mbg_output_rgi + r'/ela_ssp.csv')
        df_mb.to_csv(mbg_output_rgi + r'/mb_ssp.csv')

        # nx, ny = int(dem_vector.shape[2] + 1), int(dem_vector.shape[1] + 1)
        # lz = thk_arr.max()
        # lz = 400 if lz / 100 + .5 < 4 else np.ceil(lz / 100 + .5)
        # nz = int(lz / 10 + 1)
        #
        # with open(mbg_output_rgi + r'/ex.sh', 'w') as f:
        #     f.write(f'mpirun -np 16 pismr -regional -i init_field.nc -bootstrap -ys 0 -ye {len(t_ser)}     '
        #             f'-Mx {nx} -My {ny} -Mz {nz} -Mbz 1 -Lz {lz} -skip    '
        #             f'-grid.recompute_longitude_and_latitude false     '
        #             f'-grid.registration corner -surface given -surface_given_file SSP585-with-bounds.nc    '
        #             f'-stress_balance ssa+sia -stress_balance.ice_free_thickness_standard 1 -bed_smoother_range 0     '
        #             f'-no_model_strip 0 -dry    '
        #             f'-ts_file ts_SSP585.nc -ts_times 0:yearly:{len(t_ser)}     '
        #             f'-extra_file ex_SSP585.nc -extra_times 0:yearly:{len(t_ser)}   '
        #             f'-extra_vars ice_surface_temp,mask,thk,usurf,velsurf_mag,velbase_mag,lon,lat     '
        #             f'-o final_SSP585.nc')

        print('End of %s.' % cur_glc['RGIId'])

    except Exception as e:
        print(cur_glc['RGIId'], str(e))

        with open(mbg_output + r'/Err.txt', 'a') as f:
            f.write(cur_glc['RGIId'] + str(e) + '\n')


if __name__ == '__main__':

    rgi_pt = pd.read_csv(mbg_input + r'/RGI_Glc_Info_C.csv')
    rgi_pt_ls = [rgi_pt.loc[i, :] for i in range(rgi_pt.shape[0])]

    exists = mbg_output
    exist_ids = glob(f'{exists}/*/SSP585*')

    exist_ids = [exist_id.split('\\')[-2] for exist_id in exist_ids]
    rgi_pt_ls = [rgi_pt_slice for rgi_pt_slice in rgi_pt_ls if rgi_pt_slice['RGIId'] not in exist_ids]

    # cn_pk = pd.read_csv(f'{mbg_input}\cn_pk.csv')
    # rgi_pt_ls = [rgi_pt_slice for rgi_pt_slice in rgi_pt_ls if rgi_pt_slice['RGIId'] in cn_pk['RGIId'].values]

    # debug sample glacier ---------------------------------------------------------------------------------------------
    cur_glc = rgi_pt.loc[rgi_pt[rgi_pt['RGIId'] == 'RGI60-13.33006'].index[0], :]
    mass_balance_gradient(cur_glc)

    # Parallel process -------------------------------------------------------------------------------------------------
    Pool = multiprocessing.Pool(cpu_count=4)
    Pool.map(mass_balance_gradient, rgi_pt_ls)

    print(r'Process Accomplishment.')
