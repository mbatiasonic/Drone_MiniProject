MINI PROJECT

Samuel Mbatia Gachana

TITLE: Using Canopy Height Model (CHM) to Assess the Forest Structure of a Reforested Stand.

ABSTRACT
This project presents a pipeline for assessing forest structure in a reforested area using high-resolution drone imagery and derived digital surface and elevation models (DSM and DEM). By computing the Canopy Height Model (CHM), individual treetops were detected through local maxima analysis, and tree crowns were delineated using a watershed segmentation algorithm. From each segmented crown, quantitative metrics including maximum and mean tree height, crown area, and crown diameter were extracted, and results were aggregated to provide stand-level forest structure information. All outputs were converted to vector polygons and a CSV file compatible with GIS for further spatial analysis. This pipeline demonstrates a scalable approach for detailed forest monitoring, offering tree-level structural data to support reforestation assessment, forest management, and ecological research.

This notebook uses drone imagery and associated digital surface/elevation models (DSM/DEM) to analyze forest structure in a reforested area. The workflow leverages two custom Python modules:

* access.py – for loading, processing, and visualizing raster datasets, including orthophotos, DSM, DEM, and derived Canopy Height Model (CHM).

* assess.py – for assessing forest structure, segmenting individual tree crowns, extracting crown metrics, and saving results as vector polygons.


The project notebook is organized as follows:

* Introduction
* access.py Data Loading and visualisation
* STEP ONE
* STEP TWO
* STEP THREE
* STEP FOUR
* assess.py Forest Structure Assessment
* Conclusion
* References

The datasets include ;

* 🌍 DEM → ground elevation

* 🌲 DSM → surface elevation (including vegetation/buildings)

* 🌳 CHM → tree height (derived DSM – DEM)

* 🖼 Orthophoto → true-color RGB

* To be able to ascertain the extent and see how the dataset looks in the eye, there was visualization of the dataset.

  The assessment of forest structure using drone imagery and associated raster datasets begins with the derivation of the Canopy Height Model (CHM). The CHM is a raster representing the vertical structure of vegetation and is computed by subtracting the Digital Elevation Model (DEM), which represents the bare earth surface, from the Digital Surface Model (DSM), which captures the elevation of the ground plus any objects on it such as trees and buildings. Mathematically, this is expressed as:

***CHM = DSM - DEM***


This model isolates vegetation height from the underlying terrain, allowing for the identification of individual trees within a forested area. Within the CHM, local maxima are used to approximate the positions of tree apices. Local maxima are pixels whose values are higher than all neighboring pixels within a specified distance. Detecting these maxima is crucial for identifying potential tree tops and serves as the starting point for tree crown segmentation. Methods such as the peak_local_max function from the scikit-image library are commonly used for this purpose (Vincent & Soille, 1991; Dalponte & Coomes, 2016).

After detecting tree tops, the watershed segmentation algorithm is applied to delineate individual tree crowns. The CHM is treated analogously to a topographic surface, where peaks correspond to tree apices and valleys represent gaps between crowns. Watershed segmentation conceptually “floods” the CHM from the identified peaks, allowing regions to expand until they meet neighboring crowns. This approach is particularly effective in dense forests with uneven canopy structure, as it can separate closely spaced crowns and handle irregular shapes (Beucher & Meyer, 1993; Popescu et al., 2002).

Once crowns are delineated as discrete polygons, crown metrics are computed to quantify tree-level attributes. Each polygon corresponds to a single tree crown, and within each polygon, CHM pixels are analyzed to derive maximum and mean tree heights. The crown area is calculated as the total area of pixels within the polygon:


***Crown Area = Number of pixels × (pixel resolution)²***



The crown diameter is often approximated as the diameter of a circle with equivalent area:

***Crown Diameter = 2 × √(Crown Area / π)***
	​


These metrics are essential for understanding tree size, growth, and overall forest structure.

Integration with orthophoto data provides additional spectral information that can refine crown segmentation and enable species differentiation. Orthophotos are high-resolution, georeferenced aerial images in which the RGB channels can be analyzed to distinguish species based on color and spectral signature differences (Féret et al., 2018). This is particularly useful when trees of different species are adjacent but may have similar height profiles.


Finally, aggregating tree-level metrics allows for stand-level analysis, which provides insights into forest density, mean and median crown size, and structural complexity. Tree density is calculated as the number of detected trees divided by the area of the forest stand. Structural complexity indices, such as the coefficient of variation of tree heights or crown size distribution, provide information on forest heterogeneity, which is important for biodiversity assessment and monitoring reforestation success (Zhou et al., 2019). This workflow thus enables a scalable, quantitative assessment of forest structure, supporting forest management, ecological monitoring, and conservation efforts.

In conclusion, the assessment of the reforested stand in Kenya provides a vital snapshot of restoration in progress, revealing the nascent development of a functional forest structure. The data collected offers more than just an inventory; it serves as a benchmark for measuring future growth and a tool for evaluating the effectiveness of the restoration methodology employed. The findings underscore the importance of moving from planting trees to cultivating ecosystems, highlighting the need for continued monitoring and care. This study not only contributes valuable insights for the specific management of this Kenyan site but also reinforces the broader principle that successful reforestation is a long-term commitment. By understanding and nurturing the structural complexity of these young forests, we can better ensure they mature into resilient havens for biodiversity and robust carbon sinks for generations to come.

