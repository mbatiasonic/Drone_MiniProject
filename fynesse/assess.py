from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


# def data() -> Union[pd.DataFrame, Any]:
#     """
#     Load the data from access and ensure missing values are correctly encoded as well as
#     indices correct, column names informative, date and times correctly formatted.
#     Return a structured data structure such as a data frame.

#     IMPLEMENTATION GUIDE FOR STUDENTS:
#     ==================================

#     1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
#        - Load data using the access module
#        - Check for missing values and handle them appropriately
#        - Validate data types and formats
#        - Clean and prepare data for analysis

#     2. ADD ERROR HANDLING:
#        - Handle cases where access.data() returns None
#        - Check for data quality issues
#        - Validate data structure and content

#     3. ADD BASIC LOGGING:
#        - Log data quality issues found
#        - Log cleaning operations performed
#        - Log final data summary

#     4. EXAMPLE IMPLEMENTATION:
#        df = access.data()
#        if df is None:
#            print("Error: No data available from access module")
#            return None

#        print(f"Assessing data quality for {len(df)} rows...")
#        # Your data assessment code here
#        return df
#     """
#     logger.info("Starting data assessment")

#     # Load data from access module
#     df = access.data()

#     # Check if data was loaded successfully
#     if df is None:
#         logger.error("No data available from access module")
#         print("Error: Could not load data from access module")
#         return None

#     logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

#     try:
#         # STUDENT IMPLEMENTATION: Add your data assessment code here

#         # Example: Check for missing values
#         missing_counts = df.isnull().sum()
#         if missing_counts.sum() > 0:
#             logger.info(f"Found missing values: {missing_counts.to_dict()}")
#             print(f"Missing values found: {missing_counts.sum()} total")

#         # Example: Check data types
#         logger.info(f"Data types: {df.dtypes.to_dict()}")

#         # Example: Basic data cleaning (students should customize this)
#         # Remove completely empty rows
#         df_cleaned = df.dropna(how="all")
#         if len(df_cleaned) < len(df):
#             logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

#         logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
#         return df_cleaned

#     except Exception as e:
#         logger.error(f"Error during data assessment: {e}")
#         print(f"Error assessing data: {e}")
#         return None


# def query(data: Union[pd.DataFrame, Any]) -> str:
#     """Request user input for some aspect of the data."""
#     raise NotImplementedError


# def view(data: Union[pd.DataFrame, Any]) -> None:
#     """Provide a view of the data that allows the user to verify some aspect of its quality."""
#     raise NotImplementedError


# def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
#     """Provide a labelled set of data ready for supervised learning."""
#     raise NotImplementedError

##Drone_Mini_Project
# assess.py

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from shapely.geometry import Polygon
import geopandas as gpd
from skimage.segmentation import find_boundaries
from rasterio.features import shapes
from rasterio.plot import show


def detect_tree_tops(chm, min_distance=1):
    """
    Detect local maxima in the CHM as tree tops.
    
    Parameters:
    - chm: 2D array (CHM raster)
    - min_distance: minimum number of pixels between peaks
    
    Returns:
    - coordinates: list of (row, col) of detected tree tops
    """
    # Mask nodata
    chm_masked = np.where(chm <= 0, np.nan, chm)
    
    # Detect local maxima
    coordinates = peak_local_max(chm_masked, min_distance=min_distance, indices=True, labels=~np.isnan(chm_masked))
    
    return coordinates


def segment_tree_crowns(chm, orthophoto=None):
    """
    Segment tree crowns using watershed algorithm on CHM.
    
    Parameters:
    - chm: 2D array (Canopy Height Model)
    - orthophoto: optional 3D array (RGB) to guide segmentation
    
    Returns:
    - labeled_crowns: 2D array with unique labels for each crown
    """
    # Mask nodata
    chm_masked = np.where(chm <= 0, 0, chm)
    
    # Compute distance transform of inverted CHM (peaks = tree tops)
    distance = ndi.distance_transform_edt(chm_masked)
    
    # Find coordinates of local maxima
    coordinates = peak_local_max(chm_masked, min_distance=1, exclude_border=False)
    
    # Create a marker image
    markers = np.zeros_like(chm_masked, dtype=int)
    for i, (r, c) in enumerate(coordinates, start=1):
        markers[r, c] = i
    
    # Apply watershed
    labeled_crowns = watershed(-chm_masked, markers, mask=(chm_masked > 0))

    
    return labeled_crowns


def crowns_to_polygons(labeled_crowns, profile, min_area=1):
    """
    Convert labeled crown raster to vector polygons.
    
    Parameters:
    - labeled_crowns: 2D labeled array
    - profile: raster profile for CRS
    - min_area: minimum area (in pixels) to keep a crown
    
    Returns:
    - GeoDataFrame with polygons and crown IDs
    """
    polygons = []
    crown_ids = []

    for geom, value in shapes(labeled_crowns.astype(np.int32), mask=(labeled_crowns>0), transform=profile['transform']):
        if value <= 0:
            continue
        poly = Polygon(geom['coordinates'][0])
        if poly.area < min_area:
            continue
        polygons.append(poly)
        crown_ids.append(value)
    
    gdf = gpd.GeoDataFrame({'crown_id': crown_ids, 'geometry': polygons}, crs=profile['crs'])
    return gdf


def compute_crown_metrics(gdf, chm, profile):
    """
    Compute crown metrics: height, area, diameter.
    
    Adds columns to GeoDataFrame:
    - max_height, mean_height, crown_area, crown_diameter
    """
    chm_masked = np.where(chm <= 0, np.nan, chm)
    
    heights = []
    areas = []
    diameters = []
    
    for geom in gdf.geometry:
        # Mask CHM pixels within the polygon
        mask = rasterio.features.geometry_mask([geom], out_shape=chm.shape,
                                               transform=profile['transform'], invert=True)
        values = chm_masked[mask]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            max_h = 0
            mean_h = 0
        else:
            max_h = np.max(values)
            mean_h = np.mean(values)
        heights.append((max_h, mean_h))
        areas.append(geom.area)
        diameters.append(2 * np.sqrt(geom.area / np.pi))  # approximate as circle

    gdf['max_height'] = [h[0] for h in heights]
    gdf['mean_height'] = [h[1] for h in heights]
    gdf['crown_area'] = areas
    gdf['crown_diameter'] = diameters
    
    return gdf


def assess_forest_structure(chm, profile, orthophoto=None, min_distance=1, min_area=1):
    """
    Complete workflow:
    - detect tree tops
    - segment crowns
    - convert to polygons
    - compute crown metrics
    
    Returns:
    - GeoDataFrame with tree crowns and metrics
    """
    labeled_crowns = segment_tree_crowns(chm, orthophoto)
    gdf = crowns_to_polygons(labeled_crowns, profile, min_area=min_area)
    gdf = compute_crown_metrics(gdf, chm, profile)
    return gdf


def save_crowns_shapefile(gdf, output_path):
    """
    Save tree crowns GeoDataFrame to shapefile
    """
    gdf.to_file(output_path)

def plot_gdf_on_orthophoto(gdf, orthophoto, profile, title="Tree Crowns over Orthophoto"):
    """
    Overlay tree crown polygons (GeoDataFrame) onto the orthophoto.
    
    Parameters:
    - gdf: GeoDataFrame with crown polygons
    - orthophoto: 3D numpy array (RGB)
    - profile: rasterio profile of orthophoto
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display orthophoto
    show(orthophoto, transform=profile['transform'], ax=ax)
    
    # Plot crown polygons
    gdf.boundary.plot(ax=ax, color='red', linewidth=1)
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def save_gdf_to_csv(gdf, output_path):
    """
    Save GeoDataFrame attribute table (metrics) as CSV.
    
    Parameters:
    - gdf: GeoDataFrame
    - output_path: string path to save CSV
    """
    gdf.drop(columns='geometry').to_csv(output_path, index=False)
def plot_crown_metrics(gdf, metrics=['max_height', 'mean_height', 'crown_area', 'crown_diameter']):
    """
    Plot histograms and boxplots for tree crown metrics.
    
    Parameters:
    - gdf: GeoDataFrame with crown metrics
    - metrics: list of column names to plot
    """
    for metric in metrics:
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(gdf[metric], bins=20, color='green', alpha=0.7)
        plt.title(f"{metric} Histogram")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(gdf[metric], vert=True)
        plt.title(f"{metric} Boxplot")
        
        plt.tight_layout()
        plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def analyze_metric_correlations(csv_path, columns=None, output_path=None):
    """
    Analyze correlations between tree structural metrics.

    Parameters:
    - csv_path: str, path to the CSV file containing tree metrics
    - columns: list of column names to include in the correlation analysis
               Default: ["tree_height", "crown_diameter", "crown_area", "mean_height"]
    - output_path: optional str, path to save the correlation heatmap as PNG

    Returns:
    - corr_matrix: pandas DataFrame containing the correlation matrix
    - significance: dict with correlation coefficients and p-values
    """

    if columns is None:
        columns = ["max_height", "crown_diameter", "crown_area", "mean_height"]

    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter to selected columns
    df_metrics = df[columns]

    # Correlation matrix
    corr_matrix = df_metrics.corr()

    # Compute p-values for correlations
    significance = {}
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                corr, pval = pearsonr(df_metrics[col1].dropna(), df_metrics[col2].dropna())
                significance[(col1, col2)] = {"correlation": corr, "p_value": pval}

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation between Tree Structural Metrics", fontsize=14)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Pairplot
    sns.pairplot(df_metrics, diag_kind="kde")
    plt.suptitle("Pairwise Relationships between Tree Metrics", y=1.02, fontsize=14)
    plt.show()

    return corr_matrix, significance

