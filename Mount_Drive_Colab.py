# Montaggi GDrive
import os, sys
import xarray as xr
from google.colab import drive
drivedir='/content/drive'
drive.mount(drivedir)
os.chdir(drivedir)
datadir=drivedir+'/MyDrive/Esempi_ML_Github/'


# Aprire dataset
ds=xr.open_dataset(datadir+'nome_dataset')
