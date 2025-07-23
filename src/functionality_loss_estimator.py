from rasterstats import zonal_stats
import rioxarray as rio
from scipy.interpolate import interp1d
import numpy as np

"""
Function to estimate the flood inundation depth at the exposure location

"""



# Inputs
# vector = geodataframe of substations, industrial sites etc.
# raster =  floodmaps
# org_crs = the initial crs of the vector

def depth_calculator(vector,raster):
    
    # CRS consistency between the vector and the raster
    vector = vector.to_crs(raster.rio.crs)
    
    # Converting no data (i.e., no inundationvalues) to zero
    raster = raster.where(raster != raster.rio.nodata, 0)
    
    # Calucate the maximum inundation depth within each polygon
    stats = zonal_stats(vector, raster[0].values, affine=raster.rio.transform(), stats="mean")
    
    # Create a new column depth and if the stat is None (i.e., the polygon is outside the raster extent) we make the depth zero
    vector['depth'] = [stat['mean'] if stat['mean'] is not None else 0 for stat in stats]
    
    return vector



def depth_estimates(exposure, substations, flood_map):

    # depth at exposure locations
    exposure = depth_calculator(exposure,flood_map)

    # depth at substation locations
    substations = depth_calculator(substations,flood_map)
    return exposure, substations





"""
Functions to estimate the functionality loss at the expsoure locations

"""


def damage_functionalityloss_exposure(exposure, dam_model): 
    for i in range(len(exposure)):
    
        depth = exposure.loc[i,'depth']
        dam_class = exposure.loc[i,'dam_model']
        
        interp_func = interp1d(dam_model['Depth'], dam_model[dam_class], kind='linear', fill_value='extrapolate')
        ratio = interp_func(depth)
        exposure.loc[i,'dr/pf'] = ratio

        if dam_class == 'Agricultural':
            exposure.loc[i, 'fun_state'] = ratio
            
        else:

            # Random number 1
            #if ratio <= np.random.uniform(0,1):
            if ratio <= 0.5:
                exposure.loc[i,'fun_state'] = 0
            else:
                exposure.loc[i,'fun_state'] = 1
        
    return exposure

def functionalityloss_substations(exposure, dam_model):
    for i in range(len(exposure)):
    
        depth = exposure.loc[i,'depth']
        dam_class = 'Substations'
        
        interp_func = interp1d(dam_model['Depth'], dam_model[dam_class], kind='linear', fill_value='extrapolate')
        ratio = interp_func(depth)
        exposure.loc[i,'dr/pf'] = ratio

        # Random number 2
        #if ratio <= np.random.uniform(0,1):
        if ratio <= 0.5:
            exposure.loc[i,'fun_state'] = 0
        else:
            exposure.loc[i,'fun_state'] = 1
        
    return exposure

def fun_loss_estimates(exposure, substations, dam_model):

    # Functioanlity loss of exposure
    exposure = damage_functionalityloss_exposure(exposure, dam_model)
    
    # Functionality loss of substations
    substations = functionalityloss_substations(substations, dam_model)

    return exposure, substations