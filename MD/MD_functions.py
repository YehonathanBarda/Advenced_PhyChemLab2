import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def read_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Find the start of the table
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "c_ke" in line and "c_pe" in line and "v_etotal" in line:
            header_index = i
            break
    # Extract the table data
    table_data = []
    for line in lines[header_index + 1:]:
        if line.strip(" ") == "" or not line.strip()[0].isdigit():
            break
        table_data.append(line.split())
    # Create a DataFrame
    df = pd.DataFrame(table_data, columns=["Step", "ke", "pe", "etotal"])
    df = df.astype({"Step": int, "ke": float, "pe": float, "etotal": float})
    return df

def read_lammpstrj(filename):
    """
    Reads a LAMMPS trajectory file (.lammpstrj) and returns a Pandas DataFrame.
    Each row represents an atom at a specific timestep.
    """
    data = []
    columns = []
    timestep = None
    num_atoms = 0
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if "ITEM: TIMESTEP" in lines[i]:
                timestep = int(lines[i+1].strip())
                i += 2
            elif "ITEM: NUMBER OF ATOMS" in lines[i]:
                num_atoms = int(lines[i+1].strip())
                i += 2
            elif "ITEM: ATOMS" in lines[i]:
                columns = lines[i].strip().split()[2:]  # Extract column names
                i += 1
                for j in range(num_atoms):
                    values = lines[i].strip().split()
                    data.append([timestep] + list(map(float, values)))
                    i += 1
            else:
                i += 1
    
    df = pd.DataFrame(data, columns=["Timestep"] + columns)
    return df

def sin_func(x, a):
    return a[0] * np.sin(a[2] * x + a[2])

def find_amplitude(x,t, toplot = False):
    """
    Finds the amplitude of the potential energy oscillations.
    """
    # Fit a sine function to the potential energy data

    initial_guess = [max(x), np.sqrt(0.02), 0]
    popt, _ = curve_fit(sin_func, t, x, p0=initial_guess)
    if toplot:
        plt.plot(t, x,'.', label="Data")
        plt.plot(t, sin_func(t, *popt), label="Fit")
        plt.legend()
        plt.show()
    return popt[0]

def Leonard_Jones(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

def plot_rdf(file_path, title=r"RDF vs R/$\sigma$", Plot = True):
    """
    Reads an LAMMPS RDF output file and plots RDF vs. R.
    
    Parameters:
        file_path (str): Path to the .out file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Find the start of data by skipping header lines
    data_start = None
    for i, line in enumerate(lines):
        if not line.startswith("#") and len(line.split()) > 1:
            data_start = i + 1  # The next line is data
            break

    if data_start is None:
        print("Error: Could not find RDF data in file.")
        return

    # Read RDF data
    data = np.loadtxt(file_path, skiprows=data_start)

    # Extract relevant columns: R (column 1) and RDF (column 2)
    R = data[:, 1]
    RDF = data[:, 2]

    # Plot
    if Plot:
        plt.figure(figsize=(8, 6))
        plt.plot(R, RDF, linestyle='-', label='RDF')
        plt.xlabel(r"R/$\sigma$")
        plt.ylabel("Radial Distribution Function (RDF)")
        plt.vlines(2**(1/6), 0, max(RDF), linestyle='--', color='red', label=r"$R_{W} = 2^{1/6}\sigma$")
        plt.title(title, fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.show()
    return R, RDF

if __name__ == "__main__":
    log_file = r'C:\Users\yaniv\Yehonathan TAU\Advenced_PhyChemLab\MD\log_harmonic.lammps'

    harm_erg_df = read_log(log_file)
    print(harm_erg_df)