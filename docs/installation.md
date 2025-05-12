!!! warning "Prérequis"

    === "Installation de java 11"
    
        [java Windows OS et MacOS](https://www.oracle.com/fr/java/technologies/javase/jdk11-archive-downloads.html#license-lightbox)
    
    === "Installation de miniforge"

        [This repository holds the minimal installers for Conda and Mamba](https://github.com/conda-forge/miniforge)

        [miniforge Windows OS => Miniforge3-Windows-x86_64.exe](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
    

    === "Installation de micromamba"
    
        [micromamba Windows OS et MacOS](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
    
    === "Installation de git"
    
        [git Windows OS et MacOS](hhttps://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## 1. Installation de pymdu

### 1.1. Rapatrier le dépôt git

```bash
git clone https://github.com/rupeelab17/pymdu.git
```

### 1.2. Installation de l'environnement python

!!! warning "Comme vous voulez !!!"

    === "Conda"
    
        ```bash
        cd pymdu
        conda env create -f environment.yml # (1)!
        conda activate pymdu
        conda install git pip uv
        conda install qgis -c conda-forge
        conda install fiona -c conda-forge
        ```    

    === "Micromamba"
    
        ```bash
        cd pymdu
        micromamba env create -f environment.yml # (1)!
        micromamba activate pymdu
        micromamba install git pip uv
        micromamba install qgis -c conda-forge
        micromamba install fiona -c conda-forge
        ```

1. :man_raising_hand: L'environnement python est créé en local.

???+ info "environment.yml"

    ```yaml 
    name: pymdu
    channels:
      - conda-forge
        - defaults
        - numba
    dependencies:
        - python=3.11
        - rasterio
        - gdal
        - openjpeg
        - llvmlite
    ```

### 1.3. Installation du package pour le développement

```bash
cd pymdu
uv pip install poetry
python -m poetry install --with docs
```

!!! warning

    Pour une installation classique avec conda, il faut dé-zipper [https://github.com/rupeelab17/pymdu/blob/main/docker/processing_umep.zip](https://github.com/rupeelab17/pymdu/blob/main/docker/processing_umep.zip)
    dans le dossier share/qgis/python/plugins de l’installation de l’environment python créé (voir ligne 89 du fichier [https://github.com/rupeelab17/pymdu/blob/main/Dockerfile](https://github.com/rupeelab17/pymdu/blob/main/Dockerfile))

## 2. Installation de pymdu avec Docker

### 2.1. Installation de Docker

=== "Windows OS"
    [https://docs.docker.com/desktop/install/windows-install/](https://docs.docker.com/desktop/install/windows-install/)

=== "MacOS"
    [https://docs.docker.com/desktop/install/mac-install/](https://docs.docker.com/desktop/install/mac-install/)

### 2.2. Rapatrier le dépôt git

```bash
git clone https://github.com/rupeelab17/pymdu.git
cd pymdu
```

### 2.3. Construction de l'image Docker

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t tipee/pymdu:latest .
```

### 2.4. Exécution du conteneur

#### Sur MacOS ou Linux

=== "Première méthode"

    ```bash
    docker run --name pymdu --rm -it -p 8898:8898 -v "$(pwd)"/demos:/app/demos tipee/pymdu:latest
    ```

=== "Seconde méthode"

    ```bash
    docker create --name pymdu -p 8898:8898 -v "$(pwd)"/demos:/app/demos tipee/pymdu:latest
    docker start pymdu
    docker stop pymdu
    ```

#### Sur Windows

=== "Première méthode"

    ```bash
    docker run --name pymdu --rm -it -p 8898:8898 -v %cd%/demos:/app/demos tipee/pymdu:latest
    ```

=== "Seconde méthode"

    ```bash
    docker create --name pymdu -p 8898:8898 -v %cd%:/app tipee/pymdu:latest
    docker start pymdu
    docker stop pymdu
    ```

!!! note "Utilisation du container"

    [http://localhost:8898/](http://localhost:8898/)

## 3. Compression et conversion des images Tiff

```bash
gdal_translate -of GTiff Tmrt_1997_157_0700D.tif  Tmrt_1997_157_0700D_comp.tif -co COMPRESS=DEFLATE -co PREDICTOR=2 -co DISCARD_LSB=2

```

### 3.1. Convertir Tiff en Jpeg2000

```bash
micromamba install -c micromamba-forge openjpeg
pip install git+https://github.com/bodleian/image-processing.git
```

### 3.2. Convertir Tiff 32bit en 16bit

```bash
gdal_calc.py -A Tmrt_1997_157_0700D_comp.tif --outfile=out_round_multiplie.tif --calc="round(A,3)*1000" 
gdal_translate -of GTiff -ot int16 out_round_multiplie.tif out_round_multiplie_16bits.tif
opj_compress -i out_round_multiplie_16bits.tif -o out_round_multiplie_16bits.jp2
gdal_translate -of GTiff -ot float32 out_round_multiplie_16bits.jp2 Tmrt_1997_157_0700D_32bit.tif
gdal_calc.py -A Tmrt_1997_157_0700D_32bit.tif --outfile=Tmrt_1997_157_0700D_final.tif --calc="A/1000"
```

### 3.3. Convertir Tiff 16bit en Jpeg2000 avec Python

```python
from image_processing import openjpeg

opj = openjpeg.OpenJpeg(openjpeg_base_path="/Users/Boris/anamicromamba3/envs/pymdu/bin/opj_compress")
opj.opj_compress("Tmrt_1997_157_0700D_16bit.tif", "Tmrt_1997_157_0700D.jp2",
                 openjpeg_options=openjpeg.DEFAULT_LOSSLESS_COMPRESS_OPTIONS)
```

