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
6. **Estimate approximately crater degradation (Classes: A, AB, B, C)**
7. **Generate a detailed PDF report per crater**
8. **Export results as shapefiles and visual outputs**

---

## 📦 Setup Instructions

This project was developed using **Python 3.12.7**.

### Required Libraries

Install dependencies using either `conda` or `pip`.

- **Using Anaconda:**
  ```bash
  conda install geopandas matplotlib numpy pillow rasterio reportlab shapely scikit-image scikit-learn scipy tkinter tqdm
  ```

- **Using pip:**
  ```bash
  pip install geopandas matplotlib numpy pillow rasterio reportlab shapely scikit-image scikit-learn scipy tkinter tqdm
  ```

| Library      | Version | Documentation                                                        |
|--------------|---------|----------------------------------------------------------------------|
| geopandas    | v1.0.1  | [Docs](https://geopandas.org/en/stable/docs.html)                    |
| matplotlib   | v3.10.0 | [Docs](https://matplotlib.org/stable/index.html)                     |
| numpy        | v1.26.4 | [Docs](https://numpy.org/doc/1.26/)                                  |
| pillow       | v10.4.0 | [Docs](https://pillow.readthedocs.io/en/stable/reference/Image.html) |
| rasterio     | v1.4.3  | [Docs](https://rasterio.readthedocs.io/en/latest/)                   |
| reportlab    | v3.6.13 | [PyPI](https://pypi.org/project/reportlab/)                          |
| shapely      | v2.0.5  | [Docs](https://shapely.readthedocs.io/en/stable/)                    |
| scikit-image | v0.25.0 | [Docs](https://scikit-image.org/docs/stable/)                        |
| scikit-learn | v1.5.1  | [Docs](https://scikit-learn.org/stable/)                             |
| scipy        | v1.15.3 | [Docs](https://docs.scipy.org/doc/scipy/)                            |
| tkinter      | v8.6.14 | [Docs](https://docs.python.org/3/library/tkinter.html)                                                             |
| tqdm         | v4.67.1 | [PyPI](https://pypi.org/project/tqdm/)                               |

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
    ├── DTM
    │   ├── NAC_DTM_REINER2.tiff
    │   ├── NAC_DTM_REINER3.tiff
    │   └── ...
    └── Orthophoto
        ├── Othophoto_RG2.tiff
        ├── Othophoto_RG3.tiff
        └── ...

📁 RG25 (this repo)
├── data/
│   ├── Buffer_crateres/
│   ├── HIESINGER2011_MARE_AGE_UNITS_180/
│   ├── Swirl/
│   └── Centers/
├── results/
├── main.py
├── Circularity_and_barycenter.py
├── Maximum_search.py
├── PDF_report.py
├── Slopes.py
├── TRI.py
├── Topographical_profiles.py
├── Wanted_morph.py
└── README.md
```

---

## 📂 Project Structure

### 📁 Key Folders

- `data/`: Contains input shapefiles.
- `results/`: Output folders for crater images, reports, shapefiles, and TRI calculations.
- `logo/`: Optional branding assets.

The data folder is organized this way :

```
📁 data                                      -- This folder contains some of the data needed to launch python scripts  
├── 📁 Buffer_crateres                       -- This folder contains the crater buffers generated by the Yolov5 algorithm and ArcGIS  
│	├── 📁 Buffer_RG2                    -- Crater buffer RG2  
│	├── 📁 ... 
│	└── 📁 Buffer_RG8                    -- Crater buffer RG8  
├── 📁 HIESINGER2011_MARE_AGE_UNITS_180      -- Contains Hiesinger soil dating shapefile and other files  
├── 📁 Swirl                                 -- Contains swirl shapefiles  
└── 📁 Centers                               -- Contains centers shapefiles 
```

The results folder is organized that way :
```
📁 results               -- This folder contains all the results of the python scripts  
└── 📁 RGi               -- This folder contains results of RGi  
    ├── 📁 crater_img    -- This folder contains images of the DEM of the craters
    ├── 📁 profiles      -- This folder contains the craters profiles of RGi  
    ├── 📁 reports       -- This folder contains reports on the craters studied of RGi
    └── 📁 TRI           -- This folder contains the TRI images and GeoTiff of the craters of RGi 

```

### 🐍 Python Scripts

| Script                          | Description                                                              |
|---------------------------------|--------------------------------------------------------------------------|
| `main.py`                       | Main execution pipeline calling all components                           |
| `Circularity_and_barycenter.py` | Computes crater circularity index and barycenter                         |
| `Maximum_search.py`             | Finds max elevation at 10° intervals around crater rims                  |
| `PDf_report.py`                 | Generates a pdf report for every crater                                  |
| `Slopes.py`                     | Calculates slope angles of crater rims                                   |
| `Topographical_profiles.py`     | Generates radial topographic profiles                                    |
| `TRI.py`                        | Computes terrain ruggedness index                                        |
| `Wanted_morph`                  | Ask the user what morphology he wants to study (all or bowl-shaped only) |


---

## 📊 Outputs

All generated files are saved in `results/RG{zone}/`, including:

- **Shapefiles**:
  - `YYMMDD_RGdD_userInitials_centers_v1.shp`: Crater centers
  - `YYMMDD_RGdD_userInitials_global_results_v1.shp`: Crater classification and morphometry
  - `YYMMDD_RGdD_userInitials_highest_points_v1.shp`: Max elevation points
  - `YYMMDD_RGdD_userInitials_lowest_points_v1.shp`: Lowest floor points
  - `YYMMDD_RGdD_userInitials_rim_v1.shp`: Rim approximation
  - `YYMMDD_RGdD_userInitials_slopes_v1.shp`: Rim slopes
  

- **Visuals**:
  - PDF reports per crater in `/reports/`
  - TRI rasters in `/TRI/`
  - Profile plots in `/profiles/`
  - Picture in `/crater_img/`

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

