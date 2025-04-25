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
from config_semidiurnal import PLOT_DIR, DATA_DIR, OBS_DIR

def main(plot_month: int = 1) -> None:
    """Creates figure.
    
    Arguments:
        plot_month: the month to plot in the map; one of [1, 2, ..., 12]
    """
    
    ds_map = load_map_data(plot_month)
    ds_timeseries = load_diurnal_comp_monthly()
    
    fig, axm, axts = setup_fig_axs()
    fig.suptitle(month_names(plot_month), y=0.96, x=0.15)
    
    plot_map(fig, axm, ds_map.RANGE, ds_timeseries, plot_month)
    
    # Define timeseries locations.
    ts_locs = np.array([[5.0, 165.0],
                        [8.0, 205.0],
                        [0.0, 265.0],
                        [0.0, 165.0],
                        [-8.0, 205.0],
                        [-2.0, 250.0]])
    
    # Plot timeseries.
    for ii in range(len(axts)):
        plot_ts(fig, axts[ii], axm, ds_timeseries, ts_locs[ii,:], plot_month)
    leg = axts[2].legend(loc='upper right', ncol=4,
                         bbox_to_anchor=(0.97, 1.4))
        
    # Save figure.
    plotfileformat='png'
    plt.savefig(PLOT_DIR + 'map_with_timeseries_JAN_MAT_SST_MODEL' + '.' + plotfileformat,
                format=plotfileformat,
                dpi=400)
    
    return


def plot_ts(fig, ax, axmap, ds, location, plot_month):
    """ Add time series to axes on pre-made figure.
    """
    ds_loc = ds.sel(lat=location[0], lon=location[1],
                    month=plot_month)
    
    # Adjust to local time.
    ds_loc = order_hour_by_local_time(ds_loc, 'lon')
    ds_loc_anom = ds_loc - ds_loc.mean(dim='hour')
    
    # Plot lines.
    '#1f77b4', '#ff7f0e',
    ax.plot(range(24), ds_loc_anom.TMP_2m.data,
            label=r'$\Delta MAT$', color='#1f77b4')
    ax.plot(range(24), ds_loc_anom.TMP_SFC.data,
            label=r'$\Delta SST$', color='#ff7f0e')
    
    # Draw line from location to timeseries.
    # (https://stackoverflow.com/questions/62725479/how-do-i-transform-matplotlib-connectionpatch-i-e-for-cartopy-projection)
    use_proj = ccrs.PlateCarree(central_longitude=180.0)
    xymap = use_proj.transform_point(location[1], location[0],
                                     ccrs.PlateCarree(central_longitude=0.0))
    con = ConnectionPatch(xyB=xymap,
                          xyA=(12, 0.0),
                          coordsB="data",
                          coordsA="data",
                          axesB=axmap,
                          axesA=ax,
                          color="lime")
    axmap.add_artist(con)
    
    # Add location text.
    loc_str = format_lat_lon_label(location)
    ax.text(0.97, 0.05, loc_str,
            horizontalalignment='right',
            transform=ax.transAxes)
    
    return


def setup_fig_axs():
    """Plot figure with map and timeseries.

    Returns:
        figure, map axes, list of time series axes.
    """
    
    # Set up axes layout.
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(3,3,
                           wspace=0.1, hspace=0.15)
    axm = fig.add_subplot(gs[1,0:3],
                          projection=ccrs.PlateCarree(central_longitude=180.0),
                          aspect='auto')
    axt1 = fig.add_subplot(gs[0,0])
    axt2 = fig.add_subplot(gs[0,1], sharey=axt1)
    axt3 = fig.add_subplot(gs[0,2], sharey=axt1)
    axt4 = fig.add_subplot(gs[2,0])
    axt5 = fig.add_subplot(gs[2,1], sharey=axt4)
    axt6 = fig.add_subplot(gs[2,2], sharey=axt4)
    
    # Axes details: all panels.
    for ax in [axt1, axt2, axt3, axt4, axt5, axt6]:
        ax.set_xticks(np.arange(0,25,6), minor=False)
        ax.set_xticks(np.arange(0,25,1), minor=True)
        ax.set_yticks(np.arange(-1.0, 1.0,0.05), minor=True)
        ax.tick_params(left=True, right=True, labelleft=False, which='both')
        ax.tick_params(top=True, bottom=True, which='both')
    # Axes: top row.
    for ax in [axt1, axt2, axt3]:
        ax.tick_params(labeltop=True, labelbottom=False)
    # Axes: bottom row.
    for ax in [axt4, axt5, axt6]:
        ax.tick_params(labeltop=False, labelbottom=True)
    # Axes: left column.
    for ax in [axt1, axt4]:
        ax.tick_params(labelleft=True, labelright=False)
        ax.set_ylabel(r'$\Delta T$' + ' (K)')
    # Axes: label for bottom center.
    for ax in [axt5]:
        ax.set_xlabel('local time (hours)')

    return fig, axm, [axt1, axt2, axt3, axt4, axt5, axt6]


def plot_map(fig, axm,
             da_map,
             ds_timeseries,
             plot_month):
    """Plot map in figure.

    Arguments:
        fig: matplotlib figure
        axm: matplotlib axes
        da_map: data array with the map data.
        ds_timeseries: dataset with the timeseries data.
        plot_month: the number of the month to plot; one of [1, 2, ..., 12]
    """

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
    cmap_b = plt.get_cmap('magma')
    norm_b = colors.BoundaryNorm(np.arange(0.0, 1.01, 0.1),
                                 ncolors=cmap_b.N)
    # Plot the map.
    p = axm.pcolormesh(switch_lon_lims(da_map['geolon_t'].data, 0.0),
                       da_map['geolat_t'].data,
	               da_map.data,
                       transform=ccrs.PlateCarree(central_longitude=0.0),
                       cmap=cmap_b, norm=norm_b, shading='nearest')
    plot_month_name = month_names(plot_month) + ' '
    fig.colorbar(p,
                 label=plot_month_name + r'$\Delta SST$' + ' range (K)',
                 ax=axm,
                 orientation='vertical',
                 extend='max',
                 shrink=0.9,
                 fraction=0.08)

    # Add the TAO data locations.
    for ii in range(len(ds_timeseries.lat)):
        for jj in range(len(ds_timeseries.lon)):
            axm.scatter(ds_timeseries.lon.data[jj],
                        ds_timeseries.lat.data[ii],
                        c='lime', s=15,
                        transform=ccrs.PlateCarree(central_longitude=0.0))
    
    #plt.show()
    
    return
    


def load_map_data(plot_month: int) -> xr.Dataset:
    """Loads 2D SST diurnal cycle data to plot a map with.

    Arguments:
        plot_month: the month to plot in the map; one of [1, 2, ..., 12]

    Returns:
        xarray dataset with varaible RANGE representing the SST diurnal range.
    """
    
    # Construct month strings.
    month_str = month_names(plot_month)
    
    # Construct file name and open it if it exists.
    dicy_fn = DATA_DIR + 'meanDiurnalCycle_metrics_TEMP_' + month_str + '.nc'
    try:
        ds = xr.open_dataset(dicy_fn, decode_timedelta=False)
    except FileNotFoundError:
        print('File not found at:')
        print(dicy_fn)
        print('Try another month or check the data directory.')
        sys.exit()
    
    return ds


def month_names(plot_month: int | float) -> str:
    """Gets month name corresponding to month number.

    Arguments:
        plot_month: the month to get; one of [1, 2, ..., 12]

    Returns:
        The 3-letter abbreviation of the month name.
    """
    month_names = ['JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC']
    return month_names[int(plot_month) - 1]


def load_diurnal_comp_monthly():
    ds = xr.open_mfdataset(DATA_DIR +
                           "atmo_*_points_all_vars_meanDiurnalCycle.nc")
    
    # Calculate seasonal averages.
    month_length = ds.time.dt.days_in_month
    weights = (
        month_length.groupby("time.month") /
        month_length.groupby("time.month").sum()
    )
    ds_weighted = (ds * weights).groupby("time.month").sum(dim="time")
    
    return ds_weighted


def load_diurnal_comp_seasonal():
    ds = xr.open_mfdataset(DATA_DIR +
                           "atmo_*_points_all_vars_meanDiurnalCycle.nc")
    
    # Calculate seasonal averages.
    month_length = ds.time.dt.days_in_month
    weights = (
        month_length.groupby("time.season") /
        month_length.groupby("time.season").sum()
    )
    ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")
    
    return ds_weighted


def load_tao(loc_str):
    filename = f'SST-Tair_Jandc_{loc_str}.cdf'
    ds = xr.open_dataset(OBS_DIR + filename, decode_times=False)
    ds = ds.rename({f'TAIR{loc_str.upper()}_DC':'TAIR',
                    f'SST{loc_str.upper()}_DC':'SST',
                    'T24HR':'hour'})
    if 'LON1' in ds.coords:
        ds = ds.rename({'LON1':'LON'})
    if 'LON2' in ds.coords:
        ds = ds.rename({'LON2':'LON'})
    if 'LON3' in ds.coords:
        ds = ds.rename({'LON3':'LON'})
    if 'LON4' in ds.coords:
        ds = ds.rename({'LON4':'LON'})
    return ds.squeeze()


def switch_lon_lims(lon_list, min_lon=0.0):
    result = (lon_list - min_lon) % 360.0 + min_lon
    return result


def order_hour_by_local_time(ds, lon_name):
    local_minus_utc = switch_lon_lims(ds[lon_name], min_lon=-180.0)/15.0
    local_hour = (ds.hour + local_minus_utc) % 24
    return ds.sortby(local_hour)


def format_lat_lon_label(latlon):
    if latlon[0] == 0.0:
        lat_units = ''
    elif latlon[0] > 0.0:
        lat_units = ' N'
    else:
        lat_units = ' S'
    lat_value = f"{np.abs(latlon[0]):.1f}"

    lon_0_360 = switch_lon_lims(latlon[1], 0.0)
    if lon_0_360 > 180.0:
        lon_units = ' W'
        lon_value = f"{(360.0 - lon_0_360):.1f}"
    else:
        lon_units = ' E'
        lon_value = f"{lon_0_360:.1f}"

    loc_str = lat_value + lat_units + ', ' + lon_value + lon_units
    
    return loc_str

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
