"""Plots map of SST and timeseries of air temperature in a single figure.
"""

import sys
import os
import re
import xarray as xr
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import cm, gridspec, rcParams, colors
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import AxesGrid
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
from config_semidiurnal import (
    PLOT_DIR,
    DATA_DIR,
    GAS_CONSTANT,
    HEAT_CAPACITY
)
from map_figure import (
    switch_lon_lims,
    order_hour_by_local_time,
    format_lat_lon_label,
    month_names
)


def main():
    # Load data.
    ds_atmo = load_diurnal_comp_monthly().load()
    ds_ocn = load_diurnal_comp_monthly(component='ocn').load()
    
    # Mask out:
    ds_atmo.loc[dict(
        lon=[137.0, 147.0],
        lat=[-8.0, -5.0, -2.0]
    )] = np.nan
    ds_ocn['temp'].loc[dict(
        xt_ocean=[137.0-360.0, 147.0-360.0],
        yt_ocean=[-8.0, -5.0, -2.0]
    )] = np.nan
    
    # Calculate diurnal anomalies and combined quantities.
    ds_anom = diurnal_anomalies_adiabatic(ds_atmo)
    # Calculate residuals from different temperaure models.
    ds_resid = ds_anom[['temp_adiabatic', 'TMP_SFC', 'temp_model']] \
        - ds_anom['TMP_2m']
    # Calculate phase differences from different temperaure models.
    ds_phase_diff_max, ds_phase_diff_min = phase_diffs(ds_anom)
    
    # Setup plot.
    fig, axmap, axts = setup_fig_axs()
    
    # Add example composites.
    example_month = 1
    example_latlon = [0.0, 250.0]
    plot_example_composites(ds_anom, example_month, example_latlon, axts[0])
    
    # Add legend to first figure.
    l = axts[0].legend(loc='upper center',
                       bbox_to_anchor=(0.5, 1.4),
                       ncol=2,
                       columnspacing=-5)
    
    # Add example residuals.
    plot_example_resids(ds_resid, example_month, example_latlon, axts[1])
    
    # Add legend to second figure.
    l2 = axts[1].legend(loc='upper center',
                        bbox_to_anchor=(1.2, 1.4),
                        ncol=1)
    
    # Add pooled residuals.
    plot_all_resids(ds_resid, axts[2])
    
    # Add map of MAEs.
    ds_mae = np.abs(ds_resid).mean(dim='hour').mean(dim='month')
    mae_ratio = ds_mae['temp_model'] / ds_mae['TMP_SFC']
    plot_map(mae_ratio, fig, axmap[0])
    print('----- MAE ratio (< 1 means adiabatic model better than SST) -----')
    print(f'Max: {mae_ratio.max().data},  min: {mae_ratio.min().data},  mean; {mae_ratio.mean().data}')
    
    # Add maps of time error differences.
    tmax_error_diff = ds_phase_diff_max['temp_model'] \
        - ds_phase_diff_max['TMP_SFC']
    plot_time_map(tmax_error_diff, fig, axmap[1],
                  r'$t_{max}$' + ' error diff. (hours)')
    tmin_error_diff = ds_phase_diff_min['temp_model'] \
        - ds_phase_diff_min['TMP_SFC']
    plot_time_map(tmin_error_diff, fig, axmap[2],
                  r'$t_{min}$' + ' error diff. (hours)')
    print('----- t_max error diff (< 0 means adiabatic model better than SST) -----')
    print(f'Max: {tmax_error_diff.max().data},  min: {tmax_error_diff.min().data},  mean; {tmax_error_diff.mean().data}')
    print('----- t_min error diff (< 0 means adiabatic model better than SST) -----')
    print(f'Max: {tmin_error_diff.max().data},  min: {tmin_error_diff.min().data},  mean; {tmin_error_diff.mean().data}')
    
    # Save figure.
    plotfileformat='pdf'
    plt.savefig(PLOT_DIR + 'residuals_MAE_map' + '.' + plotfileformat,
                format=plotfileformat,
                dpi=400,
                bbox_inches='tight')
    
    return


def plot_example_composites(ds, example_month, example_latlon, ax):
    ds_example = ds.sel(month=example_month,
                        lat=example_latlon[0],
                        lon=example_latlon[1])
    ds_example = order_hour_by_local_time(ds_example, 'lon')
    ax.plot(range(24),
            ds_example.TMP_2m.data,
            c='k', label=r'$\Delta MAT$')
    ax.plot(range(24),
            ds_example.temp_model.data,
            c='r', label=r'$\Delta SST + \Delta MAT_{adiabatic}$')
    ax.plot(range(24),
            ds_example.TMP_SFC.data,
            c='b', label=r'$\Delta SST$')
    
    # Add vertical lines for times of min and max.
    ax.axvline(ds_example.TMP_2m.argmin(dim='hour'),
               c='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(ds_example.TMP_SFC.argmin(dim='hour'),
               c='b', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(ds_example.temp_model.argmin(dim='hour'),
               c='r', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(ds_example.TMP_2m.argmax(dim='hour'),
               c='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(ds_example.TMP_SFC.argmax(dim='hour'),
               c='b', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(ds_example.temp_model.argmax(dim='hour'),
               c='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add location text.
    loc_str = month_names(example_month) + ' @ ' + \
        format_lat_lon_label(example_latlon)
    ax.text(0.97, 0.05, loc_str,
            horizontalalignment='right',
            transform=ax.transAxes)
    return


def plot_example_resids(ds, example_month, example_latlon, ax):
    ds_example = ds.sel(month=example_month,
                        lat=example_latlon[0],
                        lon=example_latlon[1])
    ds_example = order_hour_by_local_time(ds_example, 'lon')
    ax.plot(range(24),
            ds_example.TMP_SFC.data,
            c='#377eb8', label=r'$\Delta SST - \Delta MAT$')
    ax.plot(range(24),
            ds_example.temp_model.data,
            c='#ff7f00', label=r'$\Delta SST + \Delta MAT_{adiabatic} - \Delta MAT$')
    ax.scatter(23.5,
               np.abs(ds_example.TMP_SFC).mean().data,
               c='#377eb8')
    ax.scatter(23.5,
               np.abs(ds_example.temp_model).mean().data,
               c='#ff7f00')
    return


def plot_all_resids(ds, ax):
    for lon in ds.lon:
        for lat in ds.lat:
            for mon in ds.month:
                ds_plot = order_hour_by_local_time(
                    ds.sel(month=mon, lat=lat, lon=lon),
                    'lon'
                )
                ax.plot(range(24),
                        ds_plot.TMP_SFC.data,
                        c='#377eb8', alpha=0.5, linewidth=0.5, zorder=2)
                ax.plot(range(24),
                        ds_plot.temp_model.data,
                        c='#ff7f00', alpha=0.2, linewidth=0.5, zorder=3)
    return


def plot_map(ds, fig, axm):

    # Set up map.
    axm.set_extent([130.0, 280.0, -12.0, 12.0],
                   crs=ccrs.PlateCarree(central_longitude=0.0))
    gl = axm.gridlines(draw_labels=True,
                       xlocs=np.arange(-180.0, 361.0, 30),
                       ylocs=np.arange(-10, 11, 5),
                       linewidths=0.1)
    gl.right_labels = True
    gl.top_labels = False
    gl.xformatter = LongitudeFormatter(zero_direction_label=False,
                                       degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    axm.coastlines(zorder=5)
    axm.add_feature(cfeature.LAND, facecolor='lightgray', zorder=4)
    # Define color map.
    cmap_b = plt.get_cmap('inferno')
    norm_b = colors.BoundaryNorm(np.arange(0.0, 1.21, 0.2),
                                 ncolors=cmap_b.N)
    # Plot the data.
    latm, lonm = np.meshgrid(ds.coords['lat'],
                             ds.coords['lon'],
                             indexing='ij')
    for ii in range(len(ds.lat)):
        for jj in range(len(ds.lon)):
            if ~np.isnan(ds.data[ii, jj]):
                sc = axm.scatter(
                    ds.lon.data[jj],
                    ds.lat.data[ii],
                    c=ds.data[ii, jj],
                    cmap=cmap_b, norm=norm_b,
                    s=30, #50*ds.data[ii, jj],
                    marker="X" if ds.data[ii, jj] > 1.0 else "o",
                    edgecolor=None,
                    transform=ccrs.PlateCarree(central_longitude=0.0),
                    zorder=5
                )
    fig.colorbar(sc,
                 label='MAE ratio',
                 ax=axm,
                 orientation='vertical',
                 extend='max',
                 shrink=0.9,
                 fraction=0.08)
    return


def plot_time_map(ds, fig, axm, cb_label):

    # Set up map.
    axm.set_extent([130.0, 280.0, -12.0, 12.0],
                   crs=ccrs.PlateCarree(central_longitude=0.0))
    gl = axm.gridlines(draw_labels=True,
                       xlocs=np.arange(-180.0, 361.0, 30),
                       ylocs=np.arange(-10, 11, 5),
                       linewidths=0.1)
    gl.right_labels = True
    gl.top_labels = False
    gl.xformatter = LongitudeFormatter(zero_direction_label=False,
                                       degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    axm.coastlines(zorder=5)
    axm.add_feature(cfeature.LAND, facecolor='lightgray', zorder=4)
    # Define color map.
    cmap_b = plt.get_cmap('RdYlBu')
    norm_b = colors.BoundaryNorm(np.arange(-6.0, 6.1, 1.0),
                                 ncolors=cmap_b.N)
    # Plot the data.
    latm, lonm = np.meshgrid(ds.coords['lat'],
                             ds.coords['lon'],
                             indexing='ij')
    for ii in range(len(ds.lat)):
        for jj in range(len(ds.lon)):
            if ~np.isnan(ds.data[ii, jj]):
                sc = axm.scatter(
                    ds.lon.data[jj],
                    ds.lat.data[ii],
                    c=ds.data[ii, jj],
                    cmap=cmap_b, norm=norm_b,
                    s=30, #50*ds.data[ii, jj],
                    marker="s" if ds.data[ii, jj] > 0.0 else "o",
                    edgecolor=None,
                    transform=ccrs.PlateCarree(central_longitude=0.0),
                    zorder=5
                )
    fig.colorbar(sc,
                 label=cb_label,
                 ax=axm,
                 orientation='vertical',
                 extend='both',
                 shrink=0.9,
                 fraction=0.08)
    return


def setup_fig_axs():
    """Plot figure with map and timeseries.

    Returns:
        figure, map axes, list of time series axes.
    """
    
    # Set up axes layout.
    fig = plt.figure(figsize=(10,14))
    gs = gridspec.GridSpec(4,3,
                           wspace=0.1, hspace=0.35)
    axm1 = fig.add_subplot(gs[1,0:3],
                           projection=ccrs.PlateCarree(central_longitude=180.0),
                           aspect='auto')
    axm2 = fig.add_subplot(gs[2,0:3],
                           projection=ccrs.PlateCarree(central_longitude=180.0),
                           aspect='auto')
    axm3 = fig.add_subplot(gs[3,0:3],
                           projection=ccrs.PlateCarree(central_longitude=180.0),
                           aspect='auto')
    axt1 = fig.add_subplot(gs[0,0])
    axt2 = fig.add_subplot(gs[0,1], sharey=axt1)
    axt3 = fig.add_subplot(gs[0,2], sharey=axt1)
    
    # Axes details: all panels.
    for ax in [axt1, axt2, axt3]:
        ax.set_xticks(np.arange(0,25,6), minor=False)
        ax.set_xticks(np.arange(0,25,1), minor=True)
        ax.set_yticks(np.arange(-1.0, 1.0,0.05), minor=True)
        ax.tick_params(left=True, right=True, labelleft=False, which='both')
        ax.tick_params(top=True, bottom=True, which='both')
        ax.set_xlim(0,24)
        ax.plot([0.0, 24.0], [0.0, 0.0],
                c='gray', linewidth=0.5, linestyle=':', alpha=0.75)
    # Axes: top row.
    for ax in [axt1, axt2, axt3]:
        ax.tick_params(labeltop=False, labelbottom=True)
    for ax in [axt2]:
        ax.set_xlabel('local time (hours)')
        ax.xaxis.set_label_position('bottom') 
    # Axes: left column.
    for ax in [axt1]:
        ax.tick_params(labelleft=True, labelright=False)
        ax.set_ylabel('temperature (K)')
    
    # Add labels to all axes.
    for (ax,letter) in zip([axt1, axt2, axt3, axm1, axm2, axm3],
                           ['a','b','c','d', 'e', 'f']):
        ax.text(0.01, 0.9, letter,
                horizontalalignment='left',
                transform=ax.transAxes)

    return fig, [axm1, axm2, axm3], [axt1, axt2, axt3]


def load_diurnal_comp_monthly(component='atmo'):
    ds = xr.open_mfdataset(DATA_DIR + component + 
                           "_*_points_all_vars_meanDiurnalCycle.nc")
    
    # Calculate seasonal averages.
    month_length = ds.time.dt.days_in_month
    weights = (
        month_length.groupby("time.month") /
        month_length.groupby("time.month").sum()
    )
    ds_weighted = (ds * weights).groupby("time.month").sum(dim="time")
    
    return ds_weighted


def diurnal_anomalies_adiabatic(ds):
    ds_mean = ds.mean(dim='hour')
    ds_anom = ds - ds_mean
    ds_anom['temp_adiabatic'] = \
        ( GAS_CONSTANT / HEAT_CAPACITY ) \
        * ( ds_mean['TMP_2m'] / ds_mean['PRES'] ) * ds_anom['PRES']
    ds_anom['temp_model'] = ds_anom['temp_adiabatic'] + ds_anom['TMP_SFC']
    return ds_anom


def phase_diffs(ds_anom):
    # Change NaN to an unrealistic value.
    # (Some of the functions below cannot handle all-NaN slices.)
    nan_hr_mask = xr.where(
        np.isnan(ds_anom['TMP_2m']).sum(dim='hour') < 24,
        1, np.nan 
    )
    unrealistic_value = -9999.0
    ds_anom = ds_anom.where(~np.isnan(ds_anom), 
                            other=unrealistic_value)
    
    i_max = ds_anom[['temp_adiabatic', 'TMP_SFC',
                     'temp_model', 'TMP_2m']]\
            .argmax(dim='hour')\
            .compute()
    i_min = ds_anom[['temp_adiabatic', 'TMP_SFC',
                     'temp_model', 'TMP_2m']]\
            .argmin(dim='hour')\
            .compute()
    ds_hourmax = xr.Dataset()
    ds_hourmin = xr.Dataset()
    for var in i_max.data_vars:
        ds_hourmax[var] = ds_anom.hour[i_max[var]]
        ds_hourmin[var] = ds_anom.hour[i_min[var]]
    ds_hourmaxdiff = np.abs(
        ds_hourmax[['temp_adiabatic', 'TMP_SFC', 'temp_model']] \
        - ds_hourmax['TMP_2m']
    )
    ds_hourmaxdiff = ds_hourmaxdiff.where(ds_hourmaxdiff <= 12.0,
                                          24 - ds_hourmaxdiff)\
                                    * nan_hr_mask
    ds_hourmindiff = np.abs(
        ds_hourmin[['temp_adiabatic', 'TMP_SFC', 'temp_model']] \
        - ds_hourmin['TMP_2m']
    )
    ds_hourmindiff = ds_hourmindiff.where(ds_hourmindiff <= 12.0,
                                          24 - ds_hourmindiff)\
                                    * nan_hr_mask
    
    return ds_hourmaxdiff.mean(dim='month'), ds_hourmindiff.mean(dim='month')

if __name__ == "__main__":
    main()
