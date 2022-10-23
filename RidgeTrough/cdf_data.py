from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime
import logging
import os
import re

# Data for a single year
class CDFData():
    # tensor dimensions - (day, level, lat, lon)
    def __init__(self, hgt_path, uwind_path=None, vwind_path=None):

        logging.getLogger().setLevel(logging.INFO)
        
        (self.year, self.times, self.lats, self.lons, self.hgts) = self.retrieve_data(hgt_path, 'hgt')
        
        # Convert to +-180
        self.lons[self.lons > 180] = self.lons[self.lons > 180] - 360

        self.uwind_path = uwind_path
        self.vwind_path = vwind_path

        # Collect Wind data and assert shapes and year are compatible
        if uwind_path:
            (u_year, u_winds) = self.retrieve_data(uwind_path, 'uwnd')
            assert u_year == self.year, "Wrong Year"
            assert u_winds.shape == self.hgts.shape, f"wind_shape!=hgts_shape\n{u_winds.shape} != {self.hgts.shape}"
            self.u_winds = u_winds

        if vwind_path:
            (v_year, v_winds) = self.retrieve_data(vwind_path, 'vwnd')
            assert v_year == self.year, "Wrong Year"
            assert v_winds.shape == self.hgts.shape, f"wind_shape!=hgts_shape\n{v_winds.shape} != {self.hgts.shape}"
            self.v_winds = v_winds

        if uwind_path and vwind_path:
            self.winds = np.sqrt(np.square(u_winds) + np.square(v_winds))        
        
        self.localize()

    def retrieve_data(self, path, name):

        fh = Dataset(path, mode='r')

        year = int(path.split('.')[-2])
        lons = fh.variables['lon'][:].data
        lats = fh.variables['lat'][:].data
        times = fh.variables['time'][:].data
        times = times - times[0]

        data = fh.variables[name][:].data
        
        time = datetime.datetime.now().strftime('%H:%M:%S')
        logging.info(f' {time} - Retrieved {year} {name} data')
        
        if name == 'hgt':
            return (year, times, lats, lons, data)
        else:
            return (year, data)

    def localize(self, lat0=20, lat1=70, lon0=-140, lon1=-20): # Default US localization

        lat_mask = ((self.lats >= lat0) & (self.lats <= lat1))
        self.loc_lats = self.lats[lat_mask]

        lon_mask = ((self.lons >= lon0) & (self.lons <= lon1))
        self.loc_lons = self.lons[lon_mask]

        start_lat = np.argwhere(lat_mask).min()
        end_lat = np.argwhere(lat_mask).max()

        start_lon = np.argwhere(lon_mask).min()
        end_lon = np.argwhere(lon_mask).max()

        self.loc_hgts = self.hgts[..., start_lat:end_lat+1, start_lon:end_lon+1]

        if self.uwind_path:
            self.loc_u_winds = self.u_winds[..., start_lat:end_lat+1, start_lon:end_lon+1]
        if self.vwind_path:
            self.loc_v_winds = self.v_winds[..., start_lat:end_lat+1, start_lon:end_lon+1]
        if self.uwind_path and self.vwind_path:
            self.loc_winds = np.sqrt(np.square(self.loc_u_winds) + np.square(self.loc_v_winds))


    def plot(self, data, day, level, loc=True, contours=5, figsize=(15, 4), cmap='YlOrRd'):
        '''
        INPUT:
            data(str): hgts, uwnd, vwnd, wnd, grad, all
            day(int/tuple): 0-365 or (Mon,day)
            level (int): 1-12
            loc (bool): Use localized data
            contours(int): Number of line contours
        '''

        # If day input is (Mon,day) tuple, convert to day
        if isinstance(day, tuple):
            day = self.date_to_day(day)

        # Get localized data, if necessary
        if loc:
            hgts = self.loc_hgts[day-1, level-1, ...]
            if self.uwind_path:
                u_winds = self.loc_u_winds[day-1, level-1, ...]
            if self.vwind_path:
                v_winds = self.loc_v_winds[day-1, level-1, ...]
            if self.uwind_path and self.uwind_path:
                winds = self.loc_winds[day-1, level-1, ...]
            lats = self.loc_lats
            lons = self.loc_lons
        else:
            hgts = self.hgts[day-1, level-1, ...]
            if self.uwind_path:
                u_winds = self.u_winds[day-1,level-1, ...]
            if self.vwind_path:
                v_winds = self.v_winds[day-1, level-1, ...]
            if self.uwind_path and self.uwind_path:
                winds = self.winds[day-1, level-1, ...]
            lats = self.lats
            lons = self.lons

        date = (datetime.datetime(self.year, 1, 1) + datetime.timedelta(day - 1)).strftime(r'%b %d %Y')
        title_str = str(date)+' : Level '+str(level)

        # Instantiate single figure and axis object
        if data in ['hgts', 'uwnd', 'vwnd', 'wnd']:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            fig.suptitle(title_str, fontsize=16)


        def gradient(arr):
    
            x_grad_kernel = np.array([[-1,0,1]])
            y_grad_kernel = x_grad_kernel.reshape(-1,1)

            x_grad = convolve(arr,x_grad_kernel)
            y_grad = convolve(arr,y_grad_kernel)

            grad = np.sqrt(np.square(x_grad) + np.square(y_grad))

            return grad
        

        def plot_data(values, title, ax):
            cf = ax.contourf(lons, lats, values, 200, transform=ccrs.PlateCarree(), cmap=cmap)
            ax.coastlines()
            ax.set_title(title)

            fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03)
            
            if contours and not 'wind' in title.lower():
                cs = ax.contour(lons, lats, values, contours, transform=ccrs.PlateCarree())
                ax.clabel(cs, inline=1, fmt='%1.0f')

        if data == 'hgts':
            plot_data(hgts, 'Geopotential Heights', ax)

        elif data == 'uwnd':
            plot_data(u_winds, 'U Winds', ax)

        elif data == 'vwnd':
            plot_data(v_winds, 'V Winds', ax)

        elif data == 'wnd':
            plot_data(winds, 'Total Wind', ax)
        
        elif data == 'grad':
            grad = gradient(self.loc_hgts[day-1,level-1])
            plot_data(grad,'Pressure Gradient',ax)

        # Plot all Data
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

            plot_data(hgts, 'Geopotential Heights',axes[0])
            axes[0].coastlines()

            plot_data(winds, 'Total Wind Speed', axes[1])
            axes[1].coastlines()
            
            fig.suptitle(title_str, fontsize=16)

            return fig, axes
        
        return fig, ax

    def quiver(self, day, level, loc=True):
        '''
        Quiver plot for wind data
        Suppress fig, ax return with plt.show()
        ''' 
        
        if isinstance(day, tuple):
            day = self.date_to_day(day)

        fig, ax = plt.subplots(1,1, figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        if loc:
            X = self.loc_lons
            Y = self.loc_lats
            U = self.loc_u_winds[day-1,level-1,...]
            V = self.loc_v_winds[day-1,level-1,...]
            C = self.loc_winds[day-1,level-1,...]
        else:
            X = self.lons
            Y = self.lats
            U = self.u_winds[day-1,level-1,...]
            V = self.v_winds[day-1,level-1,...]
            C = self.winds[day-1,level-1,...]

        ax.quiver(X,Y,U,V,C, cmap='YlOrRd', transform = ccrs.PlateCarree())
        ax.set_xlabel('Longitudes')
        ax.set_ylabel('Latitudes')
        
        return fig, ax
        
    def day_to_date(self, day, fmt=str):
        '''
        INPUT
            day(int): 1-365
            fmt(dtype): str, tuple
        OUTPUT date(str/tuple): Mon Day Year / (Mon,Day)
        '''
        dt = datetime.datetime(self.year, 1, 1) + datetime.timedelta(day - 1)
        if fmt == str:
            return dt.strftime(r'%b %d %Y')
        elif fmt == tuple:
            return (dt.month, dt.day)

    def date_to_day(self, date: tuple) -> int:
        '''
        INPUT date(tuple): (Month,day)
        OUTPUT day(int): 1-365
            - Subtract 1 from output for day index
        '''
        return datetime.datetime(self.year, *date).timetuple().tm_yday


#---------------Helper Fcns--------------
from scipy.ndimage.filters import convolve

def gradient(arr):
    
    x_grad_kernel = np.array([[-1,0,1]])
    y_grad_kernel = x_grad_kernel.reshape(-1,1)

    x_grad = convolve(arr,x_grad_kernel)
    y_grad = convolve(arr,y_grad_kernel)

    grad = np.sqrt(np.square(x_grad) + np.square(y_grad))

    return grad


def normalize(arr):
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)
    
    return (arr-arr.min()) / (arr.max()-arr.min())


def date_to_day(date: tuple) -> int:
    '''
    INPUT date(tuple): (Year,Month,day)
    OUTPUT day(int): 1-365
        - Subtract 1 from output for day index
    '''
    return datetime.datetime(*date).timetuple().tm_yday


def day_to_date(year, day, fmt=str):
    '''
    INPUT
        year(int)
        day(int): 1-365
        fmt(dtype): str, tuple
    OUTPUT date(str/tuple): Mon Day Year / (Mon,Day)
    '''
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    if fmt == str:
        return dt.strftime(r'%b %d %Y')
    elif fmt == tuple:
        return (dt.month, dt.day)