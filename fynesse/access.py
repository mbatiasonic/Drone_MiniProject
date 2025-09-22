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
!pip install rasterio

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def load_rasters(orthophoto_path, dsm_path, dem_path):
    """
    Load orthophoto, DSM, and DEM raster files.

    Parameters:
    -----------
    orthophoto_path : str
        Path to the orthophoto (RGB) TIFF file.
    dsm_path : str
        Path to the Digital Surface Model (DSM) TIFF file.
    dem_path : str
        Path to the Digital Elevation Model (DEM) TIFF file.

    Returns:
    --------
    orthophoto : numpy.ndarray
        Orthophoto raster (bands, rows, cols).
    dsm : numpy.ndarray
        DSM raster array.
    dem : numpy.ndarray
        DEM raster array.
    profile : dict
        Raster profile (from DSM, used later for saving/CHM).
    """
    with rasterio.open(orthophoto_path) as src:
        orthophoto = src.read()  # shape (bands, rows, cols)
    
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)  # first band only
        profile = src.profile
    
    with rasterio.open(dem_path) as src:
        dem = src.read(1)  # first band only
    
    return orthophoto, dsm, dem, profile


def generate_chm(dsm, dem):
    """
    Generate Canopy Height Model (CHM) = DSM - DEM.

    Parameters:
    -----------
    dsm : numpy.ndarray
        DSM raster array.
    dem : numpy.ndarray
        DEM raster array.

    Returns:
    --------
    chm : numpy.ndarray
        Canopy Height Model (DSM - DEM).
    """
    chm = np.where((dsm != 0) & (dem != 0), dsm - dem, np.nan)
    return chm


def visualize_raster(raster, title, cmap="viridis"):
    """
    Quick visualization of a single-band raster.

    Parameters:
    -----------
    raster : numpy.ndarray
        2D raster array.
    title : str
        Plot title.
    cmap : str
        Colormap (default: viridis).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(raster, cmap=cmap)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.axis("off")
    plt.show()


