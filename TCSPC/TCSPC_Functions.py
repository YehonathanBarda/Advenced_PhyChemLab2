import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad

def get_data(file_path):
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError('File format not supported')
    
    wavelength = data['Wavelength']
    wavelength_min = 400
    wavelength_max = 700
    filtered_data = data[(wavelength >= wavelength_min) & (wavelength <= wavelength_max)]
    return filtered_data

def normalize(data):
    return data / np.max(data)

def set_zero(data, col):
    return  data[col] - np.min(data[col])

def plot_together(data_F,data_R,data_F_R, Type):
    if Type == 'Abs':
        col = 'Abs'
        Title = 'Absorption'
    elif Type == 'Ems':
        col = 'S1c/R1'
        Title = 'Emission'
    elif Type == 'Ext':
        col = 'S1/R1c'
        Title = 'Extinction'
    else:
        raise ValueError('Type not supported. Choose Abs, Ems or Ext')
    data_F_zero = set_zero(data_F, col)
    data_R_zero = set_zero(data_R, col)
    data_F_R_zero = set_zero(data_F_R, col)

    sumed_spec = data_F_zero + data_R_zero
    plt.figure(figsize=(10, 6))
    plt.plot(data_F['Wavelength'], data_F_zero, label=r'Fluorescein 4.24 $\mu M$')
    plt.plot(data_R['Wavelength'], data_R_zero, label=r'Rose Bengal 6.0 $\mu M$')

    plt.plot(data_F['Wavelength'], sumed_spec, label='Sumed Fluorescein + Rose Bengal')
    plt.plot(data_F_R['Wavelength'], data_F_R_zero, label='Fluorescein + Rose Bengal')
    plt.xlabel('Wavelength')
    plt.ylabel(Title)
    plt.legend()
    plt.title(Title + ' Spectra', fontsize=16)
    plt.show()

def plot_Abs_and_Ems(data_AbsF, data_AbsR, data_EmsF, data_EmsR):
    plt.figure(figsize=(10, 6))
    data_AbsF_norm = Normalize_by_area(data_AbsF)
    data_AbsR_norm = Normalize_by_area(data_AbsR)
    data_EmsF_norm = Normalize_by_area(data_EmsF)
    data_EmsR_norm = Normalize_by_area(data_EmsR)

    plt.plot(data_AbsF_norm['Wavelength'], data_AbsF_norm['Abs'], '-C0', label=r'Absorption Fluorescein 4.24 $\mu M$')
    plt.plot(data_AbsR_norm['Wavelength'], data_AbsR_norm['Abs'], '-C3', label=r'Absorption Rose Bengal 6.0 $\mu M$')
    plt.plot(data_EmsF_norm['Wavelength'], data_EmsF_norm['S1c/R1'], '--C0', label=r'Emission Fluorescein 4.24 $\mu M$')
    plt.plot(data_EmsR_norm['Wavelength'], data_EmsR_norm['S1c/R1'], '--C3', label=r'Emission Rose Bengal 6.0 $\mu M$')
    plt.xlabel('Wavelength')

    plt.legend()
    plt.title('Absorption and Emission Spectra', fontsize=16)
    plt.xlim(400,650)
    plt.show()

def interpolate_data(x, y):
    return interp1d(x, y, kind='linear', fill_value="extrapolate")

def calc_abs_coeff(Absorption,c, l):
    Absorption['Abs'] = Absorption['Abs'] / (c * l)
    return Absorption

def calc_concentration(Absorption, Abs_coeff, l):
    max_abs = np.max(Absorption['Abs'])
    return max_abs / (Abs_coeff * l)

def calculate_overlap_integral(Emission, abs_coeff, limits, plot=False):
    Emission_norm = Normalize_by_area(Emission)
    Emission_interpolated = interpolate_data(Emission_norm['Wavelength'], Emission_norm['S1c/R1'])
    abs_coeff_interpolated = interpolate_data(abs_coeff['Wavelength'], abs_coeff['Abs'])
    integrad = lambda x: Emission_interpolated(x) * abs_coeff_interpolated(x) * (x) ** 4
    # integral, error = quad(integrad, limits[0], limits[1])

    x = np.linspace(limits[0], limits[1], int(limits[1] - limits[0]))
    y = integrad(x)
    print(x)
    plt.plot(x, y, label='Product')
    dx = x[1] - x[0]
    print(dx)
    integral = np.sum(y) 
    # print(y)
    plt.show()
    # if plot:
    #     x = np.linspace(limits[0], limits[1], 1000)
    #     # plt.plot(Emission_norm['Wavelength'], Emission_norm['S1c/R1'], label='Emission')
    #     # plt.plot(x, Emission_interpolated(x), label='Emission')
    #     plt.plot(x, abs_coeff_interpolated(x), label='abs_coeff_interpolated')
    #     plt.plot(abs_coeff['Wavelength'], abs_coeff['Abs'], label='abs_coeff_real')
    #     # plt.plot(x, integrad(x), label='Product')
    #     plt.legend()
    #     plt.show()

    return integral

def Normalize_by_area(data):
    col = data.columns[1]
    # plt.plot(data['Wavelength'], data[col], label='Original')
    data[col] = data[col] / np.trapz(data[col], data['Wavelength'])
    # plt.plot(data['Wavelength'], data[col], label='Normalized')
    # plt.legend()
    # plt.show()
    area = np.trapz(data[col], data['Wavelength'])
    print('Area under the curve is: {:.2f}'.format(area))
    return data

def calc_R0(overlap_int, QY, n, K2 = 2/3):
    return 0.2108 * (K2 * n ** -4 * QY * overlap_int) ** (1/6) * 0.1 # in nm

if __name__ == "__main__":
    Abs_folder = r'TCSPC\Data\day1 data\Absorption'
    file_pathF = os.path.join(Abs_folder, 'abs_flouresciene_4.3_micM.xlsx')
    file_pathR = os.path.join(Abs_folder, 'abs_rose_b_6micM.xlsx')
    file_pathF_R = os.path.join(Abs_folder, 'abs_roseb_flourescien_together.xlsx')
    # dataF = get_data(file_pathF)
    # dataR = get_data(file_pathR)
    # dataF_R = get_data(file_pathF_R)
    # plot_together(dataF,dataR,dataF_R, Type='Abs')