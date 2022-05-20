#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: pha-subsets.py
#-----------------------------------------------------------------------
# Version 0.2
# 30 January, 2021
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#-----------------------------------------------------------------------

"""
PHA Subsets: CRUTEM5 filtered by source code / search radius / homogenisation level
"""

import os, glob
import imageio
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmocean

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from scipy.interpolate import griddata
from scipy import spatial
from math import radians, cos, sin, asin, sqrt

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16

#target_country = 'iceland'
#target_station = 'STYKKISHOLMUR'
#radius = 5000 # km

#target_country = 'australia'
#target_station = 'PERTH'
#radius = 5000 # km

#target_country = 'russia'
#target_station = 'MOSKVA'
#radius = 6000 # km

#target_country = 'ireland'
#target_station = 'DUBLIN'
#radius = 5000 # km

#target_country = 'switzerland'
#target_station = 'BERN'
#radius = 5000 # km

#target_country = 'usa'
#target_station = 'BLUE HILL'

target_country = 'new zealand'
target_station = 'AUKLAND'

radius = 200  # Manley CET: (Edinburgh-Plymouth)/2 = 312 km

#projection = 'robinson'        
projection = 'platecarree'        

plot_annotation = True

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

# Haversine Function: 

def haversine(lat1, lon1, lat2, lon2):

    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # Radius of earth is 6371 km
    km = 6371* c
    return km

# Find nearest cell in array

def find_nearest(lat, lon, df):
    distances = df.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
    return df.loc[distances.idxmin(),:]

def find_nearest_lasso(lat, lon, df, radius):
    distances = df.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
    lasso = distances < radius
    return df.loc[lasso,:]

#-----------------------------------------------------------------------------
# LOAD HOMOGENISATION CODES --> str ARRAY
#-----------------------------------------------------------------------------

filename = 'DATA/crutem2digit_sourcecodes.csv'
nheader = 24

f = open(filename)
lines = f.readlines()    
source_codes = []
source_names = []
source_regions = []
source_descs = []
source_refs = []
source_links =[]
hom_cats	= []
hom_starts = []
hom_ends	= []
hom_comments = []
for i in range(nheader,len(lines)):    
    words = lines[i].split(',')
    if len(words) > 1:                    
        source_code = int(words[0])
        source_name = words[1]
        source_region = words[2]
        source_desc = words[3]
        source_ref_array = words[4:len(words)-5]
        source_ref = ''.join(source_ref_array)
        source_link = words[len(words)-5]
        hom_cat	= words[len(words)-4]
        hom_start = words[len(words)-3]
        hom_end	= words[len(words)-2]
        hom_comment = words[len(words)-1]

        source_codes.append(source_code)
        source_names.append(source_name)
        source_regions.append(source_region)
        source_descs.append(source_desc)
        source_refs.append(source_ref)
        source_links.append(source_link)
        hom_cats.append(hom_cat)
        hom_starts.append(hom_start)
        hom_ends.append(hom_end)
        hom_comments.append(hom_comment)

f.close()    

# Put in dataframe

ds = pd.DataFrame()
ds['source_codes'] = source_codes
ds['source_names'] = source_names
ds['source_regions'] = source_regions
ds['source_descs'] = source_descs
ds['source_refs'] = source_refs
ds['source_links'] = source_links
ds['hom_cats'] = hom_cats
ds['hom_starts'] = hom_starts
ds['hom_ends'] = hom_ends
ds['hom_comments'] = hom_comments
dt = ds.copy()
dt.replace({'hom_cats':{'###############':'HOM01'}}, inplace = True) # sc=10
dt.replace({'hom_cats':{'HOM03/HOM01':'HOM02'}}, inplace = True)     # sc=95
dt.replace({'hom_cats':{'':'HOM04'}}, inplace = True)                # sc=35
dt.replace({'hom_cats':{'HOM00':0}}, inplace = True)                  
dt.replace({'hom_cats':{'HOM01':1}}, inplace = True)                  
dt.replace({'hom_cats':{'HOM02':2}}, inplace = True)                  
dt.replace({'hom_cats':{'HOM03':3}}, inplace = True)                  
dt.replace({'hom_cats':{'HOM04':4}}, inplace = True)                  
ds = dt.copy()

#-----------------------------------------------------------------------------
# LOAD CRUTEM5
#-----------------------------------------------------------------------------

#df_anom = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')
df_anom = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')

#Index(['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#       'stationcode', 'stationlat', 'stationlon', 'stationelevation',
#       'stationname', 'stationcountry', 'stationfirstyear', 'stationlastyear',
#       'stationsource', 'stationfirstreliable'],

#-----------------------------------------------------------------------------
# APPLY COUNTRY FILTER + SET TARGET CENTRE
#-----------------------------------------------------------------------------

da = df_anom[df_anom['stationcountry'].str.contains(target_country, case = False)]
stationlat = np.array(da.groupby('stationcode')['stationlat'].mean())
stationlon = np.array(da.groupby('stationcode')['stationlon'].mean())
stationsource = np.array(da.groupby('stationcode')['stationsource'].mean())
stationcode = np.array(da['stationcode'].unique())
stationhom = []
for i in range(len(stationsource)):
    hom = int(ds[ds['source_codes']==stationsource[i]]['hom_cats'])
    stationhom.append(hom)

target = df_anom[df_anom['stationname'].str.contains(target_station, case = True)]
target_lat = target['stationlat'].unique()[0]
target_lon = target['stationlon'].unique()[0]

#-----------------------------------------------------------------------------
# Put the station id,lat,lon,source,hom code into DataFrame and exclude centre
#-----------------------------------------------------------------------------

dg = pd.DataFrame({'id':stationcode, 'lon':stationlon, 'lat':stationlat, 'source':stationsource, 'hom':stationhom})
df = dg.drop( dg[(dg['lon']==target_lon) & (dg['lat']==target_lat)].index[0] )

#-----------------------------------------------------------------------------
# FIND CLOSEST STATION (using Haversine distance)
#-----------------------------------------------------------------------------

pt = [target_lat,target_lon]  
query = find_nearest(pt[0],pt[1],df)
nearest_lon = query.lon
nearest_lat = query.lat
nearest_cell = query.name
nearest_lon_idx = np.where(stationlon==nearest_lon)[0][0]
nearest_lat_idx = np.where(stationlat==nearest_lat)[0][0]

#-----------------------------------------------------------------------------
# FIND STATIONS WITHIN SEARCH RADIUS
#-----------------------------------------------------------------------------

lasso = find_nearest_lasso(pt[0],pt[1],df,radius)

stationnames = []
for i in range(len(lasso)):
    stationnames.append(df_anom[df_anom['stationcode']==lasso.id.iloc[i]]['stationname'].values[0])
lasso_extended = lasso.copy()    
lasso_extended['name'] = stationnames
lasso_extended.reset_index(drop=True)
lasso = lasso_extended.copy()
lasso.to_csv('lasso.csv')

#-----------------------------------------------------------------------------
# MAP STATIONS BY SOURCE CODE & SEARCH RADIUS
#-----------------------------------------------------------------------------

figstr = target_country.lower() + '_' + target_station.lower() + '_' + str(radius) + '.png'
titlestr = 'CRUTEM5: ' + target_country.title() + ' stations by source code and search radius = ' + str(radius) + ' km'     

x, y = np.meshgrid(lasso['lon'], lasso['lat'])    

# Colormap for all source codes [0-99]:

n = 100
colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
            
fig  = plt.figure(figsize=(15,10))
if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(central_longitude=-30); threshold = 0
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
ax = plt.axes(projection=p)        
#ax.set_global()
ax.coastlines()
ax.add_feature(cartopy.feature.OCEAN, zorder=0, edgecolor=None, alpha=1.0)	
ax.add_feature(cf.LAND, facecolor='white', zorder=1)
ax.add_feature(cf.COASTLINE, edgecolor="black", linewidth=0.7, zorder=1)
ax.add_feature(cf.BORDERS, edgecolor="black", linewidth=0.5, zorder=1)
    
ax.gridlines()            
if projection == 'platecarree':
#   ax.set_extent([-180, 180, -90, 90], crs=p)    
    gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-', zorder=2)
    gl.xlines = True
    gl.ylines = True
    gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
    gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}   
            
for i in range(len(lasso['source'].unique())):
    code = sorted(lasso['source'].unique())[i]
    dh = lasso[lasso['source']==code]
    plt.scatter(x=dh['lon'], y=dh['lat'], marker='o', color=hexcolors[code], s=50, alpha=0.8, zorder=2, transform=ccrs.PlateCarree(), label='sourcecode='+str(code))
    if plot_annotation == True:
        for j in range(len(dh)):
            plt.text(x=dh['lon'].iloc[j], y=dh['lat'].iloc[j], s=dh['name'].iloc[j], transform=ccrs.PlateCarree()) 
plt.scatter(x=target_lon, y=target_lat, marker='s', facecolor='red', edgecolor='black', s=50, alpha=1, zorder=2, transform=ccrs.PlateCarree(), label='center='+target_station) 
if plot_annotation == True:
    plt.text(x=target_lon, y=target_lat, s=target_station, color='red', transform=ccrs.PlateCarree()) 
#plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
#fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)     
plt.legend(loc='lower right', ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
plt.tick_params(labelsize=fontsize)
plt.xticks(ax.get_xticks()[abs(ax.get_xticks())<=180])
plt.yticks(ax.get_yticks()[abs(ax.get_yticks())<=90])
plt.title(titlestr, fontsize=fontsize, pad=20)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')
       
#-----------------------------------------------------------------------------
# PLOT PER HOMOGENISATION CODE
#-----------------------------------------------------------------------------

for j in range(len(lasso['hom'].unique())):

    hom = lasso['hom'].unique()[j]
    homstr = 'HOM0'+str(hom)
    lasso_j = lasso[lasso['hom']==hom]

    figstr = target_country.lower() + '_' + target_station.lower() + '_' + str(radius) +'_' + homstr + '.png'
    titlestr = 'CRUTEM5: ' + target_country.title() + ' stations' + ' (' + homstr + ')' + ' by source code and search radius = ' + str(radius) + ' km'     

    x, y = np.meshgrid(lasso_j['lon'], lasso_j['lat'])    
            
    fig  = plt.figure(figsize=(15,10))
    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(central_longitude=-30); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
    if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)        
    #ax.set_global()
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, edgecolor=None, alpha=1.0)	
    ax.add_feature(cf.LAND, facecolor='white', zorder=1)
    ax.add_feature(cf.COASTLINE, edgecolor="black", linewidth=0.7, zorder=1)
    ax.add_feature(cf.BORDERS, edgecolor="black", linewidth=0.5, zorder=1)
        
    ax.gridlines()            
    if projection == 'platecarree':
    #   ax.set_extent([-180, 180, -90, 90], crs=p)    
        gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-', zorder=2)
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
        gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': fontsize}
        gl.ylabel_style = {'size': fontsize}   
        
    for i in range(len(lasso_j['source'].unique())):
        code = sorted(lasso_j['source'].unique())[i]
#       dh = lasso_j[lasso_j['source']==lasso_j['source'].unique()[i]]
        dh = lasso_j[lasso_j['source']==code]
        plt.scatter(x=dh['lon'], y=dh['lat'], marker='o', color=hexcolors[code], s=50, alpha=0.8, zorder=2, transform=ccrs.PlateCarree(), label='sourcecode='+str(code))
        if plot_annotation == True:
            for j in range(len(dh)):
                plt.text(x=dh['lon'].iloc[j], y=dh['lat'].iloc[j], s=dh['name'].iloc[j], transform=ccrs.PlateCarree()) 
    plt.scatter(x=target_lon, y=target_lat, marker='s', facecolor='red', edgecolor='black', s=50, alpha=1, zorder=2, transform=ccrs.PlateCarree(), label='center='+target_station) 
    if plot_annotation == True:
        plt.text(x=target_lon, y=target_lat, s=target_station, color='red', transform=ccrs.PlateCarree()) 
#   plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
#   fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)     
    plt.legend(loc='lower right', ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    plt.tick_params(labelsize=fontsize)
    plt.xticks(ax.get_xticks()[abs(ax.get_xticks())<=180])
    plt.yticks(ax.get_yticks()[abs(ax.get_yticks())<=90])
    plt.title(titlestr, fontsize=fontsize, pad=20)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

# -----------------------------------------------------------------------------
print('** END')

