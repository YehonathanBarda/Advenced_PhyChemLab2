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
    
    wavelength = data['Wavelength'].astype(float)
    wavelength_min = 400
    wavelength_max = 700
    filtered_data = data[(wavelength >= wavelength_min) & (wavelength <= wavelength_max)]
    return filtered_data

def normalize(data):
    return data / np.max(data)

def set_zero(data, col):
    return  data[col] - np.min(data[col])

def plot_together(data_F,data_R,data_F_R, Type, Title_add = '', save = False, concentrations = [4.24, 6.0, r'$\mu M$']):
    if Type == 'Abs':
        col = 'Abs'
        Title = 'Absorption'
    elif Type == 'Ems':
        col = 'S1c/R1'
        Title = 'Emission'
    elif Type == 'Ext':
        col = 'S1/R1c'
        Title = 'Excitation'
    else:
        raise ValueError('Type not supported. Choose Abs, Ems or Ext')
    data_F_zero = set_zero(data_F, col)
    data_R_zero = set_zero(data_R, col)
    data_F_R_zero = set_zero(data_F_R, col)

    sumed_spec = data_F_zero + data_R_zero
    plt.figure(figsize=(10, 6))
    plt.plot(data_F['Wavelength'], data_F_zero, label=r'Fluorescein {:.2f} '.format(concentrations[0]) + concentrations[2])
    plt.plot(data_R['Wavelength'], data_R_zero, label=r'Rose Bengal {:.2f} '.format(concentrations[1]) + concentrations[2])

    plt.plot(data_F['Wavelength'], sumed_spec, label='Sumed Fluorescein + Rose Bengal')
    plt.plot(data_F_R['Wavelength'], data_F_R_zero, label='Fluorescein + Rose Bengal')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(Title)
    plt.legend()
    plt.title(Title + ' Spectra' + Title_add, fontsize=16)
    if type(save) == str:
        plt.savefig(save) 
    elif save:
        graph_folder = os.path.join(os.getcwd(), 'TCSPC', 'graphs')
        plt.savefig(os.path.join(graph_folder, Title + ' Spectra'+ Title_add + '.png'))
    plt.show()

def plot_Abs_and_Ems(data_AbsF, data_AbsR, data_EmsF, data_EmsR, save, concentrations = [4.24, 6.0, r'$\mu M$']):
    plt.figure(figsize=(10, 6))
    data_AbsF_norm = Normalize_by_area(data_AbsF)
    data_AbsR_norm = Normalize_by_area(data_AbsR)
    data_EmsF_norm = Normalize_by_area(data_EmsF)
    data_EmsR_norm = Normalize_by_area(data_EmsR)

    plt.plot(data_AbsF_norm['Wavelength'], data_AbsF_norm['Abs'], '-C0', label=r'Absorption Fluorescein {:.1f} '.format(concentrations[0]) + concentrations[2])
    plt.plot(data_AbsR_norm['Wavelength'], data_AbsR_norm['Abs'], '-C3', label=r'Absorption Rose Bengal {:.1f} '.format(concentrations[1]) + concentrations[2])
    plt.plot(data_EmsF_norm['Wavelength'], data_EmsF_norm['S1c/R1'], '--C0', label=r'Emission Fluorescein {:.1f} '.format(concentrations[0]) + concentrations[2])
    plt.plot(data_EmsR_norm['Wavelength'], data_EmsR_norm['S1c/R1'], '--C3', label=r'Emission Rose Bengal {:.1f} '.format(concentrations[1]) + concentrations[2])
    plt.xlabel('Wavelength [nm]')

    plt.legend()
    Title = 'Absorption and Emission Spectra'
    plt.title(Title, fontsize=16)
    plt.xlim(400,650)
    if type(save) == str:
        graph_folder = os.path.join(os.getcwd(), 'TCSPC', 'graphs')
        plt.savefig(save)
    elif save:
        graph_folder = os.path.join(os.getcwd(), 'TCSPC', 'graphs')
        os.makedirs(graph_folder, exist_ok=True)
        plt.savefig(os.path.join(graph_folder, Title + '.png'))
    plt.show()

def interpolate_data(x, y):
    return interp1d(x, y, kind='linear', fill_value="extrapolate")

def calc_abs_coeff(Absorption,c, l):
    Absorption['Abs'] = Absorption['Abs'] / (c * l)
    return Absorption

def calc_concentration(Absorption, Abs_coeff, l):
    max_abs = np.max(Absorption['Abs'])
    return max_abs / (Abs_coeff * l) # in M

def calculate_overlap_integral(Emission, abs_coeff, limits, plot=False):
    Emission_norm = Normalize_by_area(Emission)
    Emission_interpolated = interpolate_data(Emission_norm['Wavelength'], Emission_norm['S1c/R1'])
    abs_coeff_interpolated = interpolate_data(abs_coeff['Wavelength'], abs_coeff['Abs'])
    integrad = lambda x: Emission_interpolated(x) * abs_coeff_interpolated(x) * (x) ** 4
    integral, error = quad(integrad, limits[0], limits[1], limit=100) # In M^-1cm^-1nm^4, limit is the maximum number of subintervals

    # x = np.linspace(limits[0], limits[1], int(limits[1] - limits[0]-1))
    # y = integrad(x)
    # print(x)
    # plt.plot(x, y, label='Product')
    # dx = x[1] - x[0]
    # integral = np.sum(y) * dx
    # error = N
    # if plot:
    #     x = np.linspace(limits[0], limits[1], 1000)
    #     # plt.plot(Emission_norm['Wavelength'], Emission_norm['S1c/R1'], label='Emission')
    #     # plt.plot(x, Emission_interpolated(x), label='Emission')
    #     plt.plot(x, abs_coeff_interpolated(x), label='abs_coeff_interpolated')
    #     plt.plot(abs_coeff['Wavelength'], abs_coeff['Abs'], label='abs_coeff_real')
    #     # plt.plot(x, integrad(x), label='Product')
    #     plt.legend()
    #     plt.show()

    return integral, error

def Normalize_by_area(data):
    data_copy = data.copy()
    col = data_copy.columns[1]
    # plt.plot(data['Wavelength'], data[col], label='Original')
    data_copy[col] = data_copy[col] / np.trapezoid(data_copy[col], data_copy['Wavelength'])
    # area = np.trapezoid(data[col], data['Wavelength'])
    # print('Area under the curve is: {:.2f}'.format(area))
    return data_copy

def calc_R0(overlap_int: float, QY: float, n: float, K2: float = 2/3, delta_overlap: float = None, delta_QY: float = 0, delta_n: float = 0) -> float:
    """
    Calculate the Förster distance R0 in nm.
    Parameters:
    overlap_int : float
        The overlap integral in M^-1 cm^-1 nm^4.
        QY : float
        The quantum yield of the donor (between 0 and 1).
            for Fluorescein QY = 0.97
            for Rose Bengal QY =  0.11
        n : float
        The refractive index of the medium. for ethanol n = 1.3617
        K2 : float, optional
        The orientation factor (default is 2/3 for random orientation).
        Returns:
        float
        The Förster distance R0 in nm.
    """
    R0 = 0.2108 * (K2 * n ** -4 * QY * overlap_int) ** (1/6) * 0.1  # in nm
    if delta_overlap is not None:
        delta_R0 = R0 * np.sqrt(
            (1/6 * delta_overlap / overlap_int) ** 2 +
            (1/6 * delta_QY / QY) ** 2 +
            (4/6 * delta_n / n) ** 2
        )
    else:
        delta_R0 = None
    return R0, delta_R0

if __name__ == "__main__":
    function =  lambda x: x**2
    x = np.linspace(0,10,100)
    integral1 = quad(function, 0, 10)
    integral2 = np.trapezoid(function(x), x)
    integral3 = np.sum(function(x)) * (x[1]-x[0])
    print(integral1)
    print(integral2)
    print(integral3)
    print('Real value is 333.33')
