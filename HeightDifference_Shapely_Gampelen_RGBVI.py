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
readORTHO = '2019-07-25_Gampelen\Agisoft\Agi_EXPORT\\20190725_GampelenStreifen_Orthophoto_47mm_transformed.tif'
###########################################################################################################
# Weitere DEMs
###########################################################################################################

# -> 03.05.2019 (Mai)
#readDEM2 =  '2019-05-03_Gampelen\Agisoft\Agi_EXPORT\\20190503_GampelenStreifen_DEM_transformed.tif'
#shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Flavien_ParzellenPlan_Mai_v2.shp')
#shift_z =  0.475 - 0.004787165 #Streifen März zu Streifen Mai -> 03.05.2019


# -> 19.06.2019 (Juni)
#readDEM2 =  '2019-06-19_Gampelen\Agisoft\Agi_EXPORT\\20190619_GampelenStreifen_DEM_transformed.tif'
#shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Flavien_ParzellenPlan_v2.shp')
#shift_z = 1.798 -0.004787165 #-2.88 #Streifen März zu Streifen Juni -> 19.06.2019

# -> 25.07.2019 (Juli)
readDEM2 =  '2019-07-25_Gampelen\Agisoft\Agi_EXPORT\\20190725_GampelenStreifen_DEM_transformed.tif'
shapefile = fiona.open('aa_Shapefiles\\Gampelen\\Gampelen_Flavien_ParzellenPlan_v2.shp')
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

num = 2

with shapefile as input:
    for feat in input:
        id = feat['id']
        shp_geom = shape(feat['geometry'])
        print(shp_geom)
        all_shapes.append(shp_geom)

shp_geom = all_shapes[num-1]
'''
num =1
first = shapefile.next() # 1st shapefile (id=1) (Gulli)

# now use the shape function of Shapely
from shapely.geometry import shape      
#shp_geom = shape(first['geometry']) 
shp_geom = shape(first['geometry'])
print(shp_geom) 
'''
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
    
    with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)
            
    # Open the clipped raster file (out_tif from function)
    clipped = rasterio.open(out_tif)
    # Visualize

    return out_img, clipped


clipped_array_1, clipped_img_1 = clip(shp_geom, readDEM1, 'Gampelen_Masked1.tif') 
clipped_array_2, clipped_img_2 = clip(shp_geom, readDEM2, 'Gampelen_Masked2.tif') 
clipped_ortho_array, clipped_ortho_img = clip(shp_geom, readORTHO, 'Gampelen_Masked3.tif') 
clipped_ortho_array_trans=clipped_ortho_array.transpose((1, 2, 0))


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

mu_1 = np.nanmean(diff_1.flatten())
sigma_1 = np.nanstd(diff_1.flatten()) #ohne nan's


k2, p = stats.normaltest(diff_1.flatten())
alpha = 1e-3
print("p = {:g}".format(p))

if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
                
# best fit of data

# the histogram of the data
fig1=plt.figure()
#n, bins, patches = plt.hist(diff_1.flatten(), 80, normed=1, facecolor='blue', alpha=0.75, label =r'$\mu = %.4f \ m,\  \sigma = %.4f$' %(mu_1, sigma_1))
n, bins1, patches1 = plt.hist(diff_1.flatten(), 80, normed=1, facecolor='blue', alpha=0.75, label =r'$\mu = %.4f \ m,\  \sigma = %.4f$' %(mu_1, sigma_1))

# add a 'best fit' line
y = mlab.normpdf(bins1, mu_1, sigma_1)

l = plt.plot(bins1, y, 'r--', linewidth=2)


#plot
plt.xlabel('Differenz [m]')
plt.ylabel('rel. Haeufigkeit [%]')
plt.title("Diff. Marz - Juli, Gampelen-%i" % num)
plt.xlim(-0.3, 0.3)
plt.ylim(0., 35.)

#plt.grid(True)
plt.legend()


#################################################################################
#%% RGBVI calculations
#################################################################################

plt.figure()
plt.imshow(clipped_ortho_array_trans)


R_R = (clipped_ortho_array_trans[:,:,0]/255)
R_G = (clipped_ortho_array_trans[:,:,1]/255)
R_B = (clipped_ortho_array_trans[:,:,2]/255)

RGBVI = (R_G**2 - (R_B*R_R)) / (R_G**2 + (R_B*R_R))
RGBVI = RGBVI[0:upper_rlim,0:upper_clim] 

plt.figure("RGBVI")
plt.imshow(RGBVI)

height_DEM = (clipped_array_2_gauss[0] + shift_z)[0:upper_rlim,0:upper_clim] - (clipped_array_1_gauss[0])[0:upper_rlim,0:upper_clim]


grassI = height_DEM  + RGBVI

model_height = 0.0091*np.exp(3.1447*np.nanmean(grassI)) # MOdel from BAreth et al.
model_height_HAFL = 0.0091*np.exp(3.1447*np.nanmean(grassI)) + 0.0185

print("DEM_mean_height:", mu_1)
print("grassIndex:", np.nanmean(grassI))
print("RPM model Bareth:", model_height)
print("RPM model HAFL:" , model_height_HAFL)

#%%&

#Own model but compared to DEM measurements out of curiosity
from numpy.polynomial.polynomial import polyfit

grassI= grassI[~np.isnan(grassI)]
height_DEM = height_DEM[~np.isnan(height_DEM)]

# Fit with polyfit
b, m = polyfit(grassI.flatten(), height_DEM.flatten(), 1)

plt.figure("Own model DEM Height as a fct. GrassI)")
plt.scatter(grassI.flatten(),height_DEM.flatten(), s = 0.05)
plt.plot(grassI, b + m * grassI, '-')
plt.xlabel("grassI value")
plt.ylabel("DEM_height")

import numpy

# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['r-squared'] = ssreg / sstot

    return results

polyfit(grassI,height_DEM,1)

#################################################################################
#%% Plot the Flattening of one Line
#################################################################################

fig2 = plt.figure()
#plt.plot((clipped_array_1[0])[int(upper_rlim/2)], 'k')
#plt.plot((clipped_array_1_perc[0])[int(upper_rlim/2)], label="Percentile (95%, size 8)")
#plt.plot((clipped_array_1_gauss[0, int(upper_rlim/1.1), : ]), label="1.")
#plt.plot((clipped_array_2_gauss[0, int(upper_rlim/1.1), :] - shift_z), label="2.")

#plt.plot((clipped_array_4_gauss[0, int(upper_rlim/2.5), :] - shift_z_3), label="4.")

for i in range(int(upper_rlim/2), int(upper_rlim/2)+1):
    print (i)
    plt.plot((clipped_array_1_offset[0, i, :]), label= "Marz horiz.")
    plt.plot((clipped_array_2_offset[0, i, :] + shift_z), label="Juli horiz.")
    plt.xlabel("Pixel #")
    plt.ylabel("m. ü. M")


plt.title("Gampelen-%i" % num)


#plt.plot((clipped_array_1_gauss[0,:, int(upper_clim/2)]), label="Marz vert.")
#plt.plot((clipped_array_2_gauss[0,:,int(upper_clim/2)] - shift_z), label="Juli vert.")

plt.show()   

plt.legend()


#%%

fig3=plt.figure()

plt.plot(diff_1, 'o', markersize = 0.5, rasterized=True)
plt.title("Gampelen-%i, Diff. all pixels" % num)

#%%

fig4=plt.figure()

plt.title("Gampelen-%i" % num)
plt.imshow(clipped_array_2_gauss[0])
plt.colorbar()
plt.clim(429, 431) # limits (Gampelen btw. 429 & 433)
#plt.axvline(int(upper_clim/2))
plt.axhline(int(upper_rlim/2), color = "k")

plt.xlabel('Column #')
plt.ylabel('Row #')

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