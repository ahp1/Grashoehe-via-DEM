# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:55:04 2019

@author: bfhNT
"""
import rasterio
import rasterio.plot


import matplotlib.pyplot as plt
import os
import geopandas as gpd
from fiona.crs import from_epsg
from rasterio.mask import mask 
import fiona
import numpy as np

plt.close("all")

#################################################################################
#%% Set Parameters
#################################################################################

# set wd
# os.chdir('\\Users\\MinorNT\\OneDrive - Berner Fachhochschule\\aa_Weidenmanagement\\Orthophotos')
os.chdir('D:\\aa_Weidemanagement\\Orthophotos\\SA_Flavien')

#choose image folder

img_1 = '2019-07-25_Gampelen\Agisoft\Agi_EXPORT\\20190725_GampelenStreifen_DEM.tif'
#img_2 = '2019-03-06_Gampelen\GampelenFull-1\Agisoft\Agi_EXPORT\\20190306_GampelenFull2_Orthophoto.tif'
# Choose ShapeFile
shape_dir = 'aa_Shapefiles\\Gampelen\\Gampelen_ganzerStreifen.shp'

#################################################################################
#%% Read ShapeFile
#################################################################################
     
shape = fiona.open(shape_dir)
#shape = fiona.open("Shapefile_Bretzwil/Overlap_Chillsabel_1u2.shp")

# print(shape.schema)
#first feature of the shapefile
first = shape.next() # 1st shapefile (id=1)

# conversion shapely
# now use the shape function of Shapely
from shapely.geometry import shape
#shp_geom = shape(first['geometry']) 
shp_geom = shape(first['geometry']) 
print(shp_geom)

#################################################################################             
#%% Define Shape with shapely to do calculations on the same area with the same dimensions
#################################################################################

data_dir = 'D:/aa_Weidemanagement/Python-Codes'

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
    print(coords)
    # Clip the raster with Polygon
    
    out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
    
    # Copy the metadata
    out_meta = data.meta.copy()
    print(out_meta)

    # Parse EPSG code
    epsg_code = int(data.crs.data['init'][5:])
    print(epsg_code)

    out_meta.update({"driver": "GTiff",
                  "height": out_img.shape[1],
                  "width": out_img.shape[2],
                  "transform": out_transform
                  #"crs": pycrs.parse.from_epsg_code(21781).to_proj4()
                  })
    
    with rasterio.open(out_tif, "w", **out_meta) as dest: # "w" : open in writing mode
            dest.write(out_img)
            
    # Open the clipped raster file (out_tif from function)
    clipped = rasterio.open(out_tif)
    # Visualize

    return out_img, clipped


clipped_array_1, clipped_img_1 = clip(shp_geom, img_1, '1.tif')
#clipped_array_2, clipped_img_2 = clip(shp_geom, img_2, '2.tif') #GampelenFull-1_DEM

#################################################################################
#%% Adapt xy-coordinates
#################################################################################

clipped_img_1.bounds
#clipped_img_2.bounds

gt1 = clipped_img_1.transform
#gt2 = clipped_img_2.transform


x_res1 = gt1[0]
y_res1 = gt1[4]
x_topleft1 = gt1[2]
y_topleft1 = gt1[5]
'''
x_res2 = gt2[0]
y_res2 = gt2[4]
x_topleft2 = gt2[2]
y_topleft2 = gt2[5]

shift_x = x_topleft1 - x_topleft2
shift_y = y_topleft1 - y_topleft2 

'''

## -> 10.05.2019    
# shift_x = -0.88
# shift_y = -0.98


## -> 19.06.2019   
shift_x = -0.425
shift_y = 0.559

## -> 25.07.2019   
shift_x = -1.193
shift_y = -1.099

from rasterio.transform import from_origin
transformed_coords = from_origin(x_topleft1 + shift_x, y_topleft1 + shift_y, x_res1, -y_res1) #achtung signs
transformed_coords

# Copy the metadata
data = rasterio.open(img_1)
test = np.reshape(data.read(1), (1, data.shape[0], data.shape[1])) # damit (1, x, y)
out_meta = data.meta.copy()
print(out_meta)



out_meta.update({"driver": "GTiff",
                "height": clipped_img_1.shape[0], # clipped_img_1.shape[0]
                "width": clipped_img_1.shape[1],# clipped_img_1.shape[1]
                "transform": transformed_coords
                #"crs": pycrs.parse.from_epsg_code(21781).to_proj4()
                })
#new_dataset = rasterio.open("2019-03-23_Bretzwil\\Chillsabel_1\\Agisoft\\Agi_EXPORT\\2019-03-23-Chillsabel-1_DEM_transformed.tif", "w", **out_meta)


with rasterio.open(img_1[:-4] +'_transformed.tif', "w", **out_meta) as dest: # "w" : open in writing mode
    print(dest.bounds)
    #dest.write(test) 
    dest.write(clipped_array_1) # 

