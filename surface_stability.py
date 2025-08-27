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
from plot_decomposition import(
    load_diurnal_comp_monthly,
    diurnal_anomalies_adiabatic
)


def main():
    # Load data.
    ds_atmo = load_diurnal_comp_monthly().load()
    # Mask out:
    ds_atmo.loc[dict(
        lon=[137.0, 147.0],
        lat=[-8.0, -5.0, -2.0]
    )] = np.nan
    
    # Calculate surface humidity.
    ds_atmo['SPFHS'] = 0.98 * qsat(ds_atmo['TMP_SFC'], ds_atmo['PRES'])
    
    # Calculate wind speed.
    ds_atmo['WSPD_10m'] = np.sqrt(ds_atmo['VGRD_10m']**2 +
                                  ds_atmo['UGRD_10m']**2)
    
    # Calculate diurnal anomalies and combined quantities.
    ds_anom = diurnal_anomalies_adiabatic(ds_atmo)
    
    # Setup plot.
    fig, axs = setup_fig_axs()
    
    # Plots.
    plot_many_months(ds_anom.PRES/100.0,
                     axs[0,0],
                     r'$\Delta p$' + ' (hPa)')
    plot_many_months(ds_anom.temp_adiabatic,
                     axs[0,1],
                     r'$\Delta MAT_{adiabatic}$' + ' (K)')
    plot_many_months(ds_anom.TMP_SFC,
                     axs[1,0],
                     r'$\Delta SST$' + ' (K)')
    plot_many_months(ds_anom.TMP_2m,
                     axs[1,1],
                     r'$\Delta MAT$' + ' (K)')
    ylims_sst = axs[1,0].get_ylim()
    ylims_mat = axs[1,1].get_ylim()
    for ax in [axs[1,0], axs[1,1]]:
        ax.set_ylim(min(ylims_sst[0], ylims_mat[0]),
                    max(ylims_sst[1], ylims_mat[1]))
    plot_many_months(ds_atmo.TMP_SFC - ds_atmo.TMP_2m,
                     axs[2,0],
                     r'$SST - MAT$' + ' (K)')
    plot_many_months(ds_anom.WSPD_10m,
                     axs[2,1],
                     r'$\Delta |U_{10m}|\ (m~s^{-1})$')
    plot_many_months(ds_anom.SPFH_2m * 1000.0,
                     axs[3,0],
                     r'$\Delta q_{2m}\ (g~kg^{-1})$')
    plot_many_months((ds_anom.SPFHS - ds_anom.SPFH_2m) * 1000.0,
                     axs[3,1],
                     r'$\Delta (q_{s} - q_{2m})\ (g~kg^{-1})$')
    plot_many_months(ds_anom.SHTFL,
                     axs[4,0],
                     r'$\Delta Q_{S}\ (W~m^{-2})$')
    plot_many_months(ds_anom.LHTFL,
                     axs[4,1],
                     r'$\Delta Q_{E}\ (W~m^{-2})$')
    
    # Save figure.
    plotfileformat='png'
    plt.savefig(PLOT_DIR + 'surface_stability_main' + '.' + plotfileformat,
                format=plotfileformat,
                dpi=400,
                bbox_inches='tight')
    plotfileformat='pdf'
    plt.savefig(PLOT_DIR + 'surface_stability_main' + '.' + plotfileformat,
                format=plotfileformat,
                dpi=400,
                bbox_inches='tight')
    
    return


def setup_fig_axs():
    """Plot figure with map and timeseries.

    Returns:
        figure, map axes, list of time series axes.
    """
    
    # Set up axes layout.
    fig, axs = plt.subplots(nrows=5, ncols=2,
                            sharex=True, squeeze=False,
                            figsize=(10,14))
    
    # Axes details: all panels.
    for ax in axs.flatten():
        ax.set_xticks(np.arange(0,25,6), minor=False)
        ax.set_xticks(np.arange(0,25,1), minor=True)
        ax.tick_params(left=True, right=True, which='both')
        ax.tick_params(top=True, bottom=True, labelbottom=False, which='both')
        ax.set_xlim(0,24)
        ax.axhline(0.0,
                   c='gray', linewidth=0.5, linestyle=':', alpha=0.75)
        ax.axvline(3.0, c='gray', linewidth=0.5, linestyle='--', alpha=0.75)
        ax.axvline(9.0, c='gray', linewidth=0.5, linestyle='--', alpha=0.75)
        ax.axvline(15.0, c='gray', linewidth=0.5, linestyle='--', alpha=0.75)
        ax.axvline(21.0, c='gray', linewidth=0.5, linestyle='--', alpha=0.75)
    # Axes: bottom row.
    for ax in axs[-1,:].flatten():
        ax.tick_params(labeltop=False, labelbottom=True)
        ax.set_xlabel('local time (hours)')
        ax.xaxis.set_label_position('bottom')
    # Axes: left column.
    for ax in axs[:,0].flatten():
        ax.tick_params(labelleft=True, labelright=False)
    # Axes: right column.
    for ax in axs[:,-1].flatten():
        ax.tick_params(labelleft=False, labelright=True)
        ax.yaxis.set_label_position('right')
    
    # Add labels to all axes.
    for (ax,letter) in zip(axs.flatten(),
                           ['a','b','c','d', 'e', 'f', 'g', 'h', 'i', 'j']):
        ax.text(0.01, 0.9, letter,
                horizontalalignment='left',
                transform=ax.transAxes)

    return fig, axs


def plot_many_months(da, ax, label):
    for lon in da.lon:
        for lat in da.lat:
            for mon in da.month:
                da_plot = order_hour_by_local_time(
                    da.sel(month=mon, lat=lat, lon=lon),
                    'lon'
                )
                if 'da_sum' in locals():
                    da_sum.data = da_sum.data + da_plot.fillna(0.0).data
                    N = N + 1
                else:
                    da_sum = da_plot.fillna(0.0)
                    N = 1
                ax.plot(range(24),
                        da_plot.data,
                        c='b', alpha=0.3, linewidth=0.3)
    ax.plot(range(24),
            da_sum.data / N,
            c='k', alpha=0.5, linewidth=1.5)
    ax.set_ylabel(label)
    return


def qsat(T, p):
    """Uses Tetens' formula for saturation vapor pressure from
    Buck(1981) JAM 20, 1527-1532

    T - temp in K
    p - pressure in Pa
    """
    # Ratio of gas constants (dry air / water vapor).
    loc_epsilon = 287.0 / 461.0
    esat = (1.0007 + 0.00000346 * (p/100.0)) * 6.1121 * \
        np.exp(17.502 * (T - 273.15) / (240.97 + (T - 273.15)))

    # Convert to specific humidity (kg kg-1).
    qsat = loc_epsilon * esat / ((p/100.0) - (1.0 - loc_epsilon)*esat)
    
    return qsat


if __name__ == "__main__":
    main()
