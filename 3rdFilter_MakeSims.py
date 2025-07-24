import os
import subprocess

input_directory = r"C:\DATABASES_NICOCASTRO\AM_software\am_sims"
output_directory = r"C:\DATABASES_NICOCASTRO\AM_software\am_sims\espectros"


os.makedirs(output_directory, exist_ok=True)

amc_files = [f for f in os.listdir(input_directory) if f.endswith('.amc')]

for amc_file in amc_files:
    amc_file_path = os.path.join(input_directory, amc_file)

    output_file_path = os.path.join(output_directory, os.path.splitext(amc_file)[0] + '.txt')

    command = f"am {amc_file_path} 20 GHz 26 GHz 0.5 MHz 0 deg 0.9 >> {output_file_path}"

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully processed {amc_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {amc_file}: {e}")
        
        #%%
#import numpy as np
#import matplotlib.pyplot as plt

#txt_file_path = r"C:\DATABASES_NICOCASTRO\AM_software\am_sims\espectros\1.txt"

#try:
#    data = np.loadtxt(txt_file_path)

#    if data.ndim != 2 or data.shape[1] < 2:
#        raise ValueError("Unexpected file format")

#    freq = data[:, 0]  
#    spectrum = data[:, 1]  
#    noise = data[:, 2] if data.shape[1] > 2 else None  

#    plt.figure(figsize=(10, 5))
#    plt.plot(freq, spectrum, label="Spectrum")
#    if noise is not None:
#        plt.plot(freq, noise, label="Noise", linestyle='--')

#    plt.xlabel("Frequency [GHz]")
#    plt.ylabel("Tb / Trj [K]")
#    plt.ylim(7.5, 11)
#    plt.title("AM Spectrum Output")
#    plt.legend()
#    plt.grid(True)
#    plt.show()

#except Exception as e:
#    print("Error reading or plotting file:", e)