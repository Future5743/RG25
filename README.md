# 🛰️ Semi-Automatic Algorithm for Crater Analysis in the Reiner Gamma Region

This repository provides a comprehensive pipeline for morphometric analysis and degradation classification of lunar impact craters using Digital Terrain Model (DTM) data and vector shapefiles. While optimized for the Reiner Gamma region, the framework is adaptable for other lunar surfaces.

---

## 🚀 Pipeline Overview

For each crater, the pipeline performs:

1. **Load and crop raster (DTM) data**
2. **Extract geometric features:**
   - Maximum and minimum elevation points  
   - Crater diameter  
   - Circularity index (Miller index)
3. **Calculate rim and interior slopes**
4. **Estimate terrain roughness using TRI (Terrain Ruggedness Index)**
5. **Generate a fitted crater buffer**
6. **Classify crater degradation (Classes: A, AB, B, C)**
7. **Generate a detailed PDF report per crater**
8. **Export results as shapefiles and visual outputs**

---

## 📦 Setup Instructions

This project was developed using **Python 3.12.7**.

### Required Libraries

Install dependencies using either `conda` or `pip`.

- **Using Anaconda:**
  ```bash
  conda install geopandas matplotlib numpy rasterio shapely tqdm scikit-learn scikit-image scipy reportlab
  ```

- **Using pip:**
  ```bash
  pip install geopandas matplotlib numpy rasterio shapely tqdm scikit-learn scikit-image scipy reportlab
  ```

| Library        | Version  | Documentation                                           |
|----------------|----------|----------------------------------------------------------|
| geopandas      | v1.0.1   | [Docs](https://geopandas.org/en/stable/docs.html)        |
| matplotlib     | v3.10.0  | [Docs](https://matplotlib.org/stable/index.html)         |
| numpy          | v1.26.4  | [Docs](https://numpy.org/doc/1.26/)                      |
| rasterio       | v1.4.3   | [Docs](https://rasterio.readthedocs.io/en/latest/)       |
| shapely        | v2.0.6   | [Docs](https://shapely.readthedocs.io/en/stable/)        |
| tqdm           | v4.67.1  | [PyPI](https://pypi.org/project/tqdm/)                   |
| scikit-learn   | v1.5.1   | [Docs](https://scikit-learn.org/stable/)                 |
| scikit-image   | v0.25.0  | [Docs](https://scikit-image.org/docs/stable/)            |
| scipy          | v1.13.1  | [Docs](https://docs.scipy.org/doc/scipy/)                |
| reportlab      | v3.6.13  | [PyPI](https://pypi.org/project/reportlab/)              |

---

## 🌐 Required External Data

You must download NAC DTM datasets from the [LROC DTM Portal](https://wms.lroc.asu.edu/lroc/rdr_product_select?filter%5Btext%5D=rein&filter%5Bprefix%5D%5B%5D=NAC_DTM).

### Download the following tiles:

| Tile              | Download Link                                                                                                                          | README |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------|
| Reiner Gamma 2 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER2/NAC_DTM_REINER2.TIF) | [README_RG2](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER2/NAC_DTM_REINER2_README.TXT) |
| Reiner Gamma 3 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER3/NAC_DTM_REINER3.TIF) | [README_RG4](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER4/NAC_DTM_REINER4_README.TXT) |
| Reiner Gamma 5 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER5/NAC_DTM_REINER5.TIF) | [README_RG5](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER5/NAC_DTM_REINER5_README.TXT) |
| Reiner Gamma 6 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER6/NAC_DTM_REINER6.TIF) | [README_RG6](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER6/NAC_DTM_REINER6_README.TXT) |
| Reiner Gamma 7 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER7/NAC_DTM_REINER7.TIF) | [README_RG7](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER7/NAC_DTM_REINER7_README.TXT) |
| Reiner Gamma 8 | [NAC_DTM_REINER2.TIF](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER8/NAC_DTM_REINER8.TIF) | [README_RG8](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/REINER8/NAC_DTM_REINER8_README.TXT) |
### Recommended directory structure:
```
📁 data
└──📁 RG
    ├── NAC_DTM_REINER2.tiff
    ├── NAC_DTM_REINER3.tiff
    └── ...
📁 RG25 (this repo)
├── data/
│   ├── Buffer_crateres/
│   ├── HIESINGER2011_MARE_AGE_UNITS_180/
│   ├── Swirl/
│   └── Centers/
├── results/
├── main.py
├── Circularity.py
├── Graph_dtoD.py
├── Maximum_search.py
├── Slopes.py
├── TRI.py
├── Topographical_profiles.py
└── README.md
```

---

## 📂 Project Structure

### 📁 Key Folders

- `data/`: Contains input shapefiles and downloaded DTM data.
- `results/`: Output folders for crater images, reports, shapefiles, and TRI calculations.
- `logo/`: Optional branding assets.

The data folder is organized this way :

```
📁 data -- This folder contains some of the data needed to launch python scripts  
├── 📁 Buffer_crateres -- This folder contains the crater buffers generated by the Yolov5 algorithm and ArcGIS  
│	├── 📁 Buffer_RG2 -- Crater buffer RG2  
│	├── 📁 ... 
│	└── 📁 Buffer_RG8 -- Crater buffer RG8  
├── 📁 HIESINGER2011_MARE_AGE_UNITS_180 -- Contains Hiesinger soil dating shapefile and other files  
├── 📁 Swirl -- Contains swirl shapefiles  
└── 📁 Centers -- Contains centers shapefiles 
```

The results folder is organized that way :
```
📁 results -- This folder contains all the results of the python scripts  
└── 📁 RGi -- This folder contains results of RGi  
    ├── 📁 crater_img -- This folder contains images of the DEM of the craters
    ├── 📁 profiles -- This folder contains the craters profiles of RGi  
    ├── 📁 reports -- This folder contains reports on the craters studied of RGi
    └── 📁 TRI -- This folder contains the TRI images and GeoTiff of the craters of RGi  

```

### 🐍 Python Scripts

| Script                     | Description                                                |
|----------------------------|------------------------------------------------------------|
| `main.py`                  | Main execution pipeline calling all components             |
| `Circularity.py`           | Computes crater circularity index                          |
| `Graph_dtoD.py`            | Plots depth/diameter trends with uncertainty               |
| `Maximum_search.py`        | Finds max elevation at 10° intervals around crater rims     |
| `Slopes.py`                | Calculates slope angles of crater rims                     |
| `TRI.py`                   | Computes terrain ruggedness index                          |
| `Topographical_profiles.py`| Generates radial topographic profiles                      |

---

## 📊 Outputs

All generated files are saved in `results/RG{zone}/`, including:

- **Shapefiles**:
  - `0_results_global_RG{zone}.shp`: Crater classification and morphometry
  - `1_results_rim_RG{zone}.shp`: Rim approximation
  - `2_results_slopes_RG{zone}.shp`: Rim slopes
  - `3_results_max_RG{zone}.shp`: Max elevation points
  - `4_results_low_RG{zone}.shp`: Lowest floor points
  - `5_results_centers_RG{zone}.shp`: Crater centers
  - `6_results_90_RG{zone}.shp`: Profile points every 90°

- **Visuals**:
  - PDF reports per crater in `/reports/`
  - TRI rasters in `/TRI/`
  - Profile plots in `/profiles/`

---

## ⚙️ How to Run

Set zones to analyze in `main.py`:
```python
zones = [2, 3, 4, 5, 6, 7, 8]
```
Verify the path used in `main.py`:
```python
crater_shapefile_path = os.path.join('data', 'Buffer_crateres', f'Buffer_RG{zone}')
raster_path = os.path.join('..', 'data', 'RG', 'DTM', f'NAC_DTM_REINER{zone}.tiff')
hiesinger_path = os.path.join('data', 'HIESINGER2011_MARE_AGE_UNITS_180', 'HIESINGER2011_MARE_AGE_UNITS_180.SHP')
swirls_path = os.path.join('data', 'Swirl', 'REINER_GAMMA.shp')
```
Then launch the main script:
```bash
python main.py
```

---

## 🛠️ Troubleshooting

- **Missing TIF or shapefile?** Make sure all external data is downloaded and correctly placed in the `data/` directory **outside of this repository**.
- **Path errors?** Adjust file paths in `main.py` to match your folder structure.
- **Incorrect versions?** Check library versions listed above.

---

## 🤝 Contributing

Feel free to fork the project, open issues, or submit pull requests! Whether it’s bug fixes or feature suggestions, contributions are welcome.

---

## 👥 Contributors

- **Lemelin Myriam** – Project Lead and Guidance
- **Calas Guilhem** – Project Lead and Guidance
- **Lagneaux Abigaëlle** – Developer

