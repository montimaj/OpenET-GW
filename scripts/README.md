## Toward Sustainable Groundwater Management: Harnessing Remote Sensing and Climate Data to Estimate Field-Scale Groundwater Pumping

### PI: [Justin Huntington](https://www.dri.edu/directory/justin-huntington/)
### Software authors/maintainers: [Thomas Ott](https://www.dri.edu/directory/thomas-ott/), [Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/)

<img src="../Readme_Figures/official-dri-logotag-trans-bkgd.png" height="80"/><img src="../Readme_Figures/nv_state_logo.png" height="90"/> <img src="../Readme_Figures/owrd.jpg" height="80"/>

This software has been successfully tested on the [Apple MacBook Pro 2023](https://www.apple.com/macbook-pro/) (macOS Ventura 13.6.2) and on standard Windows 11 laptop/desktops.

## Citations
**Journal**: [Ott, T.J.](https://www.dri.edu/directory/thomas-ott/), [Majumdar, S.](https://www.dri.edu/directory/sayantan-majumdar/), [Huntington, J.L.](https://www.dri.edu/directory/justin-huntington/), 
[Pearson, C.](https://www.dri.edu/directory/chris-pearson/), [Bromley, M.](https://www.dri.edu/directory/matthew-bromley/), 
[Minor, B.A.](https://www.dri.edu/directory/blake-minor/), [Morton, C.G.](https://www.dri.edu/directory/charles-morton/), 
[Sueki, S.](https://www.dri.edu/directory/sachiko-sueki/), [Beamer, J.P.](https://www.linkedin.com/in/jordan-beamer-89ba8020/), & 
[Jasoni, R.](https://www.dri.edu/directory/richard-jasoni/) (2024). 
Toward Sustainable Groundwater Management: Harnessing Remote Sensing and Climate Data to Estimate Field-Scale Groundwater Pumping and Irrigation Efficiencies. _Submitted to Elsevier [Agricultural Water Management](https://www.sciencedirect.com/journal/agricultural-water-management) special issue on _Irrigation monitoring through Earth Observation (EO) data__.


## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If you have Anaconda or miniconda installed already, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. You may have to restart the current shell session to add conda to path.

You could update to the latest conda package manager by running ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help
you install the correct packages and Python version to run the codes.

### 2. Clone or download the repository

Download the repository from the compressed file link at the top right of the repository webpage, or clone the repository using Git.

#### Repository disk space requirements
This repository is very lightweight and will only take ~117 MB of disk space.

### 3. Creating the conda environment and installing packages
Open Linux/Mac terminal or Windows PowerShell and run the following:
```
conda create -y -n openetgw python=3.11
conda activate openetgw
conda install -y -c conda-forge rioxarray geopandas lightgbm earthengine-api rasterstats seaborn openpyxl swifter
```

Once the above steps are successfully executed, run the following to load the GDAL_DATA environment variable which is needed by 
rasterio.

```
conda deactivate
conda activate openetgw
```

### 4. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk).

### 5. Running the scripts
From the Linux/Mac terminal or Windows PowerShell:
1. Make sure that the `openetgw` conda environment is active. If not, run ```conda activate openetgw``` before running the following codes.
2. Enter any of the `scripts` directory, e.g., `scripts/figures/`
3. Run `python dv_figures.py` to generate the DV figures shown in Ott et al. (2024)


