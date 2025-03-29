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


def compute_msd(oxygen_df, box_bounds):
    """
    Computes the Mean Squared Displacement (MSD) considering Periodic Boundary Conditions (PBCs).
    
    Parameters:
    oxygen_df (DataFrame): Sorted DataFrame containing oxygen atom positions over time.
    box_bounds (dict): Dictionary with 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi' defining the simulation box.
    
    Returns:
    DataFrame: MSD values over time.
    """
    # Define box dimensions
    box_size = np.array([box_bounds['xhi'] - box_bounds['xlo'], 
                          box_bounds['yhi'] - box_bounds['ylo'], 
                          box_bounds['zhi'] - box_bounds['zlo']])
    
    # Get initial positions
    oxygen_initial = oxygen_df[oxygen_df['Timestep'] == 0].set_index('id')[['x', 'y', 'z']]

    
    # Compute displacements considering PBCs
    diff_oxygen = oxygen_df[['Timestep', 'id', 'x', 'y', 'z']].copy()
    
    for coord, length in zip(['x', 'y', 'z'], box_size):
        diff_oxygen[coord] -= diff_oxygen['id'].map(oxygen_initial[coord])
        diff_oxygen[coord] = apply_pbc(diff_oxygen[coord], length)
    
    # Compute Mean Squared Displacement (MSD)
    MSD = diff_oxygen.groupby('Timestep').agg({
        'x': lambda x: (x**2).mean(), 
        'y': lambda y: (y**2).mean(), 
        'z': lambda z: (z**2).mean()
    }).reset_index()
    
    MSD['MSD'] = MSD[['x', 'y', 'z']].sum(axis=1)
    
    return MSD

# Function to apply PBC corrections
def apply_pbc(displacement, box_length):
    return displacement - box_length * np.round(displacement / box_length)

# Linear fit function
def linear_fit(x, a, b):
    return a * x + b

def compute_VACF(oxygen_df):
    # Extract the initial velocity components at t=0
    initial_vx = oxygen_df[oxygen_df["Timestep"] == 0]["vx"].values
    initial_vy = oxygen_df[oxygen_df["Timestep"] == 0]["vy"].values
    initial_vz = oxygen_df[oxygen_df["Timestep"] == 0]["vz"].values
    
    # Create a new DataFrame to store the velocity autocorrelation function for each timestep
    vacf = []
    
    # Loop through each timestep from t=0 to the maximum timestep in the DataFrame
    timesteps = sorted(oxygen_df["Timestep"].unique())
    
    for t in timesteps:
        # Get the velocities at timestep t
        vx_t = oxygen_df[oxygen_df["Timestep"] == t]["vx"].values
        vy_t = oxygen_df[oxygen_df["Timestep"] == t]["vy"].values
        vz_t = oxygen_df[oxygen_df["Timestep"] == t]["vz"].values
        
        # Average over all particles <v(0) * v(t)> 
        vacf_t = np.mean(initial_vx * vx_t + initial_vy * vy_t + initial_vz * vz_t)
        vacf.append(vacf_t)
    
    return vacf, timesteps

def Calculate_Diffusion_coefficient_from_VACF(vacf, timesteps):
    # time step is in 1 fs
    # vacf is in (A/fs)^2
    D = 1/3 * np.trapz(vacf, timesteps) # in A^2/fs
    D *= 1e-5 # in m^2/s
    return D

if __name__ == "__main__":
    log_file = r'C:\Users\yaniv\Yehonathan TAU\Advenced_PhyChemLab\MD\log_harmonic.lammps'
    harm_erg_df = read_log(log_file)
    print(harm_erg_df)