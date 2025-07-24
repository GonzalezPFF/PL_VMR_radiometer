# PL_VMR_radiometer
Reconstruction of vertical profile of Pressure-H2O content from 20-26 GHz spectra 

MAIN FOLDER: https://drive.google.com/drive/folders/1LMMYTr-idTX-5k1gvMP9bISbvzEOUP43?usp=sharing

# FOR THE AM SIMULATIONS

From NASA's main file (NASA's Modern-Era Retrospective analysis for Research and Applications version 2, MERRA2 reanalysis), MERRA-2 inst3_3d_asm_Nv: 3d, 3-Hourly, Instantaneous, Model-Level, Assimilation, Assimilated Meteorological Fields V5.12.4 (M2I3NVASM) database (https://disc.gsfc.nasa.gov/datasets/M2I3NVASM_5.12.4/summary):

'0Filter_savefromMERRA.ipynb' shows, from the subset time: 24-06-20; lon: [-180.0, 179.4]; lat [-90.0, 90.0]; lev: [1.0, 72.0];  in 'MERRA_nv.nc', extracted from the main MERRA-2 aforementioned datafile, the filtering and re-transforming border pressure (PL[mbar]), water vapor content (H2O[vmr]), temperature (T[K]) and mid-layer differential pressure (DELP[mbar]) data to layers from 5 to 72, all time indexes, latitude indexes from 35 to 326 separated by 2 indexes one each other, and longitude indexes from 0 to 576 separated by 4 indexes one each other, to relational database. Output file: 'merradf_output.txt'

'1stFilter.ipynb' shows chunking columns from 'merradf_output.txt' (object-measurement creation), creating output file 'merradf_output_structured.csv'. 

'2ndFilter+AMC.ipynb' shows layer separation finding the nearest averaged-index closest to 225 [mbar] to tropopause and 1 [mbar] stratopause, and creating AMC configfiles for simulations. Output files: 'x.amc', x ranging from 1 to 21024.

'3rdFilter_MakeSims.py' shows, for a given directory containing 'x.amc' files, construct the simulated spectra for each '.amc' file. Each spectra has resolution of 0.5 MHz, from 20 to 26 GHz. Output files: 'x.txt', x ranging from 1 to 21024.


# FOR THE MODEL TRAINING:

Once spectra is simulated from parameters previously extracted from MERRA-2, it's time for regression problem: From the noised simulated spectra, we need to return the vertical profile of H2O[vmr] v/s PL[mbar]. We try simply Fully-Connected Neural Network (FCNN) model, and Residual Neural Network (RESNET) model, as well as ML models of Decision Tree and Random Forest (under construction):

'4thFilter_AttachingDF.ipynb' shows, from 'x.txt' and 'merradf_output_structured.csv', the construction of sorted trios of (spectra, PL[mbar], H2O[vmr]), pkl-packaging the arrays. Output file: 'input_nn_ver1.pkl'.

'5thFilter_Handling_N_Training_sampleXX.ipynb', once noise is added to spectras, shows FCNN and RESNET, and '5thFilter_Handling_N_Training_sampleXX.py'shows DecisionTree models trying to solve the regression problem.

Best RMSE metrics but not too much sensitivity to variance are shown in XXX_sample25RES_copy.ipynb.
Best RMSE metrics in ML models, sensitives to variance, are shown in XXX_sample32dtree.py
