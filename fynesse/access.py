"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

# drone mini_Project_access.py
# access.py

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show

def load_rasters(orthophoto_path, dsm_path, dem_path):
    """
    Load orthophoto, DSM, and DEM raster files.
    """
    with rasterio.open(orthophoto_path) as src:
        orthophoto = src.read()  # shape (bands, rows, cols)
        ortho_profile = src.profile
    
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        dsm_profile = src.profile
    
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem_profile = src.profile
    
    return orthophoto, dsm, dem, dsm_profile


def generate_chm(dsm, dem):
    """
    Generate Canopy Height Model (CHM) = DSM - DEM.
    """
    chm = np.where((dsm != 0) & (dem != 0), dsm - dem, np.nan)
    return chm


def mask_nodata(raster, nodata_value=-9999):
    """
    Replace nodata values with NaN for proper visualization.
    """
    return np.where(raster == nodata_value, np.nan, raster)


def visualize_dem(dem, profile, title="Digital Elevation Model (DEM)"):
    dem_masked = mask_nodata(dem, profile.get('nodata', -9999))
    plt.figure(figsize=(10, 8))
    # Clip extremes to 2nd-98th percentile for better contrast
    vmin, vmax = np.nanpercentile(dem_masked, [2, 98])
    img = plt.imshow(dem_masked, cmap="terrain",
                     extent=rasterio.plot.plotting_extent(dem_masked, profile['transform']),
                     vmin=vmin, vmax=vmax)
    plt.colorbar(img, label="Elevation (m)")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def visualize_dsm(dsm, profile, title="Digital Surface Model (DSM)"):
    dsm_masked = mask_nodata(dsm, profile.get('nodata', -9999))
    plt.figure(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(dsm_masked, [2, 98])
    img = plt.imshow(dsm_masked, cmap="terrain",
                     extent=rasterio.plot.plotting_extent(dsm_masked, profile['transform']),
                     vmin=vmin, vmax=vmax)
    plt.colorbar(img, label="Surface Elevation (m)")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def visualize_chm(chm, profile, title="Canopy Height Model (CHM)"):
    chm_masked = mask_nodata(chm, profile.get('nodata', -9999))
    plt.figure(figsize=(10, 8))
    img = plt.imshow(chm_masked, cmap="YlGn",
                     extent=rasterio.plot.plotting_extent(chm_masked, profile['transform']))
    plt.colorbar(img, label="Height (m)")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()



def visualize_orthophoto(orthophoto, title="Orthophoto (RGB)"):
    """
    Visualize RGB orthophoto with true colors.
    """
    plt.figure(figsize=(10, 8))
    # Convert (bands, rows, cols) -> (rows, cols, bands)
    plt.imshow(np.transpose(orthophoto, (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.show()



