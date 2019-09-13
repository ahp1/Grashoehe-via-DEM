# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:55:04 2019

@author: bfhNT
"""
import rasterio
import rasterio.plot
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from fiona.crs import from_epsg
from rasterio.mask import mask
from skimage import filters 
import fiona
from scipy import stats
import matplotlib.mlab as mlab
from shapely.geometry import shape      

plt.close("all")

#%% Hier ändern
# Mai
os.chdir('D:\\aa_Weidemanagement\\Orthophotos\\SA_Flavien')

###########################################################################################################
# Referenz DEM
###########################################################################################################

readDEM1 = '2019-03-06_Gampelen\\GampelenStreifen\\Agisoft\Agi_EXPORT\\GampelenStreifen_DEM.tif'

###########################################################################################################
# Weitere DEMs
###########################################################################################################

# -> 03.05.2019 (Mai)
#readDEM2 =  '2019-05-03_Gampelen\Agisoft\Agi_EXPORT\\20190503_GampelenStreifen_DEM_transformed.tif'
#shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Flavien_ParzellenPlan.shp')
#shift_z =  0.475 - 0.004787165 #Streifen März zu Streifen Mai -> 03.05.2019


# -> 19.06.2019 (Juni)
#readDEM2 =  '2019-06-19_Gampelen\Agisoft\Agi_EXPORT\\20190619_GampelenStreifen_DEM_transformed.tif'
#shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Flavien_ParzellenPlan.shp')
#shift_z = 1.798 -0.004787165 #-2.88 #Streifen März zu Streifen Juni -> 19.06.2019

# -> 25.07.2019 (Juli)
readDEM2 =  '2019-07-25_Gampelen\Agisoft\Agi_EXPORT\\20190725_GampelenStreifen_DEM_transformed.tif'
shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Schnittflaechen_Juli.shp')
shift_z = 0.825 -0.004787165 #-2.88 #Streifen März zu Streifen Juni -> 19.06.2019

data_dir = '.\\zz_PythonCodes\\'


'''
Zur Info (don not (yet) killMe)

#shift_z =  -0.4837068 # Full1
#shift_z = -0.4837068 - 0.0021 # Full 2
#shift_z = -0.4837068 - 0.0093 # Full 3
#shift_z = -0.4837068 +0.0674 # Full4
'''
#%%
#################################################################################
#%% Read ShapeFile
#################################################################################

# print(shape.schema)
#first feature of the shapefile
all_shapes = []
all_areas = []

num = 9

with shapefile as input:
    for feat in input:
        id = feat['id']
        shp_geom = shape(feat['geometry'])
        print(shp_geom)
        all_shapes.append(shp_geom)
        all_areas.append(shp_geom.area)
        
#shp_geom = all_shapes[num - 1]


#################################################################################             
#%% Define Shape with shapely to do calculations on the same area with the same dimensions
#################################################################################

#input_image = 'GampelenStreifen_DEM.tif'
#output_mask = 'Gampelen_Masked.tif'

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def clip(shapefile, input_image, output_mask):
    data = rasterio.open(input_image)
    # Output raster
    out_tif = os.path.join(data_dir, output_mask)
    
    #bbox = box(minx, miny, maxx, maxy)       
    geo = gpd.GeoDataFrame({'geometry': shapefile}, index=[0], crs=from_epsg(21781)) # Muss angepasst werden, da minx etc. in CH1903 oder epsg:21781
    # print(geo)
    
    # Project the Polygon into same CRS as the grid
    geo = geo.to_crs(crs=data.crs.data)
    
    coords = getFeatures(geo)
    # print(coords)
    # Clip the raster with Polygon
    
    out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
    
    # Copy the metadata
    out_meta = data.meta.copy()
    # print(out_meta)

    # Parse EPSG code
    epsg_code = int(data.crs.data['init'][5:])
    print(epsg_code)

    out_meta.update({"driver": "GTiff",
                  "height": out_img.shape[1],
                  "width": out_img.shape[2],
                  "transform": out_transform
                  #"crs": pycrs.parse.from_epsg_code(21781).to_proj4()
                  })
    
    with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)
            
    # Open the clipped raster file (out_tif from function)
    clipped = rasterio.open(out_tif)
    # Visualize

    return out_img, clipped

height_all_miniSquares = np.array([])
sigma_all_miniSquares = np.array([])

for l in all_shapes:
    # print(l)
    shp_geom = l

    clipped_array_1, clipped_img_1 = clip(shp_geom, readDEM1, 'Gampelen_Masked1.tif') 
    clipped_array_2, clipped_img_2 = clip(shp_geom, readDEM2, 'Gampelen_Masked2.tif') 
    
    
    clipped_array_1[clipped_array_1 == -32767.0] = 'nan'
    clipped_array_2[clipped_array_2 == -32767.0] = 'nan'
    
    clipped_array_1_gauss = filters.gaussian(clipped_array_1, sigma = 0.5)# 30
    clipped_array_2_gauss = filters.gaussian(clipped_array_2, sigma = 0.5)
    
    clipped_array_1_offset = filters.gaussian(clipped_array_1, sigma=2)# 30
    clipped_array_2_offset = filters.gaussian(clipped_array_2, sigma=2) 


#################################################################################         
#%% Berechnungen / Histogramm
#################################################################################

    # ACHTUNG anzahl rows, cols nicht identisch da cm/px variieren kann
    
    rows1 = len(clipped_array_1_gauss[0])
    rows2 = len(clipped_array_2_gauss[0])
    
    cols1 = len((clipped_array_1_gauss[0])[0])
    cols2 = len((clipped_array_2_gauss[0])[0])
    
    
    upper_clim = np.min([cols1, cols2])
    upper_rlim = np.min([rows1, rows2])
    
    
    diff_1 = (clipped_array_2_gauss[0] + shift_z)[0:upper_rlim,0:upper_clim] - (clipped_array_1_gauss[0])[0:upper_rlim,0:upper_clim]
    diff_1 = diff_1[~np.isnan(diff_1)]
    
    diff_1 = diff_1[diff_1 > -0.4]
    diff_1 = diff_1[diff_1 < 0.4]
    
    height= np.nanmean(diff_1.flatten())
    sigma_height = np.nanstd(diff_1.flatten()) #ohne nan's
    
    print(height) 
    height_all_miniSquares = np.append(height_all_miniSquares, height)
    sigma_all_miniSquares = np.append(sigma_all_miniSquares, sigma_height)

    
print (height_all_miniSquares)
print (sigma_all_miniSquares)

modelFS =  height_all_miniSquares*np.asarray(all_areas)

realFS = np.array([260,290,150,100,100,120,50,60,60])
factor = realFS/modelFS

mean_parcel_modelFS = np.mean(modelFS.reshape(-1, 3), axis=1)
mean_parcel_realFS = np.mean(realFS.reshape(-1, 3), axis=1)

factor_mean = mean_parcel_realFS/mean_parcel_modelFS

np.mean(factor_mean)


model_prediction = np.mean(factor_mean) * modelFS


#%% save all fiugres to one pdf
'''
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
multipage("./zz_Resultate/Juli/%i-Gampelen_Juli_v2.pdf" %num, [fig1, fig2, fig3, fig4], dpi=250)
'''