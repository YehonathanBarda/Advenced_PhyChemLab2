from TCSPC_Functions import *
import os
from scipy.constants import Avogadro

'''
Day 2
High concentration - FRET
'''
# Absorption
Abs_folder = r'TCSPC\Data\day2 data\Absorption'
file_pathAbsF = os.path.join(Abs_folder, 'Flu cuvette500um 0.1mM absorption.xlsx')
file_pathAbsR = os.path.join(Abs_folder, 'RosB cuvette500um 0.1mM absorption.xlsx')
file_pathAbsF_R = os.path.join(Abs_folder, 'RosB and Flu cuvette500um 0.1mM absorption.xlsx')
data_AbsF = get_data(file_pathAbsF)
data_AbsR = get_data(file_pathAbsR)
data_AbsF_R = get_data(file_pathAbsF_R)
data_AbsF -= data_AbsF[data_AbsF['Wavelength'] > 610]['Abs'].mean() # background subtraction
data_AbsR -= data_AbsR[data_AbsR['Wavelength'] > 610]['Abs'].mean() # background subtraction
data_AbsF_R -= data_AbsF_R[data_AbsF_R['Wavelength'] > 610]['Abs'].mean() # background subtraction
concentration_F = calc_concentration(data_AbsF, 92300, 0.05) * 1E3 # in mM, Absorption coefficient of fluorescein is 92,300 M-1cm-1 and l = 0.05 cm
concentration_R = calc_concentration(data_AbsR, 90400, 0.05) * 1E3 # in mM, Absorption coefficient of Rose Bengal is 90,400 M-1cm-1 and l = 0.05 cm
day2_concentrations = [concentration_F, concentration_R, 'mM']
# plot_together(data_AbsF, data_AbsR, data_AbsF_R,Type='Abs', Title_add=' 500 um cuvette', save=True, concentrations=day2_concentrations)

# Emission
Ems_folder = r'TCSPC\Data\day2 data\Emission\cuvette500um'
file_pathEmsF = os.path.join(Ems_folder, 'Flourescein Emission at 390 nm cuvette 500um c0.1mM.csv')
file_pathEmsR = os.path.join(Ems_folder, 'roseb_Emission at 390 nm cuvette 500um c0.1mM.csv')
file_pathEmsF_R = os.path.join(Ems_folder, 'resb and Flu Emission at 390 nm cuvette 500um 0.05 mM.csv')
data_EmsF = get_data(file_pathEmsF)
data_EmsR = get_data(file_pathEmsR)
data_EmsF_R = get_data(file_pathEmsF_R)
# plot_together(data_EmsF, data_EmsR, data_EmsF_R,Type='Ems', Title_add=' 500 um cuvette', save=True, concentrations=day2_concentrations)

# Excitation
Ext_folder = r'TCSPC\Data\day2 data\Excitation'
file_pathExtF = os.path.join(Ext_folder, 'Flourescein Excitation at 600 nm cuvette 500um c0.1mM.csv')
file_pathExtR = os.path.join(Ext_folder, 'roseb_Excitation at 600 nm cuvette 500um c0.1mM.csv')
file_pathExtF_R = os.path.join(Ext_folder, 'roesb and Flu Excitation at 600 nm cuvette 500um c0.1mM.csv')
data_ExtF = get_data(file_pathExtF)
data_ExtR = get_data(file_pathExtR)
data_ExtF_R = get_data(file_pathExtF_R)
# plot_together(data_ExtF, data_ExtR, data_ExtF_R,Type='Ext', Title_add=' 500 um cuvette', save=True, concentrations=day2_concentrations)

'''
Results:
'''

'''
Overlap Integral
'''

# Fluorescein - Fluorescein
print('Fluorescein - Fluorescein')
print('Fluorescein concentration is: {:.2f} mM'.format(concentration_F))
F_Abs_coeff = calc_abs_coeff(data_AbsF,concentration_F * 1E-3, 0.05) # in M^-1cm-1
plt.plot(F_Abs_coeff['Wavelength'], F_Abs_coeff['Abs'], label=r'Abs Coeff Fluorescein 4.24 $\mu M$')
plt.plot(data_EmsF['Wavelength'],normalize(data_EmsF['S1c/R1']), label=r'Normalized Emission Fluorescein 4.24 $\mu M$')
plt.vlines([480, 530], 0, max(F_Abs_coeff['Abs']), colors='r', linestyles='dashed')
plt.legend()
plt.xlabel('Wavelength')
plt.show()

F_FoverlapInt, F_FoverlapIntErr = calculate_overlap_integral(data_EmsF, F_Abs_coeff, [480, 530], plot=True) # In M^-1cm^-1nm^4
R0_F_F, delta_R0_F_F = calc_R0(F_FoverlapInt, 0.97, 1.3617, delta_overlap=F_FoverlapIntErr, delta_n=0.0001) # in nm
distance_between_molecules = 1/(concentration_F * 1E-3 * Avogadro) ** (1/3) * 1E8 # in nm, Concentration is in mM
print('J(F-F) = {:.2e} ± {:.2e} M^-1cm^-1nm^4    R0 = {:.2f} ± {:.2f} nm'.format(F_FoverlapInt, F_FoverlapIntErr, R0_F_F, delta_R0_F_F))
print('Distance between molecules (F -> F) is: {:.2f} nm'.format(distance_between_molecules))

# Fosecein - Rose Bengal
print('Fluorescein - Rose Bengal')
print('Rose Bengal concentration is: {:.2f} mM'.format(concentration_R))
R_Abs_coeff = calc_abs_coeff(data_AbsR,concentration_R * 1E-3, 0.05) # in M^-1cm-1
plt.plot(R_Abs_coeff['Wavelength'], R_Abs_coeff['Abs'], label=r'Abs Coeff Rose Bengal 0.08 $mM$')
plt.plot(data_EmsF['Wavelength'],normalize(data_EmsF['S1c/R1']), label=r'Normalized Emission Fluorescein 0.1 $mM$')
plt.vlines([460, 600], 0, max(R_Abs_coeff['Abs']), colors='r', linestyles='dashed')
plt.legend()
plt.xlabel('Wavelength')
plt.show()

F_RoverlapInt, F_RoverlapIntErr = calculate_overlap_integral(data_EmsF, R_Abs_coeff, [460, 600], plot=True) # In M^-1cm^-1nm^4
R0_F_R, delta_R0_F_R = calc_R0(F_RoverlapInt, 0.97, 1.3617, delta_overlap=F_RoverlapIntErr, delta_n=0.0001) # in nm
print('J(F-R) = {:.2e} ± {:.2e} M^-1cm^-1nm^4    R0 = {:.2f} ± {:.2f} nm'.format(F_RoverlapInt, F_RoverlapIntErr, R0_F_R, delta_R0_F_R))
distance_between_molecules = 1/(concentration_R * 1E-3 * Avogadro) ** (1/3) * 1E8 # in nm, Concentration is in mM
print('Distance between molecules (F -> R) is: {:.2f} nm'.format(distance_between_molecules))

# Rose Bengal - Fluorescein
print('Rose Bengal - Fluorescein')
F_Abs_coeff = calc_abs_coeff(data_AbsF,concentration_F * 1E-3, 0.05) # in M^-1cm-1
plt.plot(F_Abs_coeff['Wavelength'], F_Abs_coeff['Abs'], label=r'Abs Coeff Fluorescein 0.1 $mM$')
plt.plot(data_EmsR['Wavelength'],normalize(data_EmsR['S1c/R1']), label=r'Normalized Emission Rose Bengal 0.08 $mM$')
plt.vlines([460, 600], 0, max(F_Abs_coeff['Abs']), colors='r', linestyles='dashed')
plt.legend()
plt.xlabel('Wavelength')
plt.show()

R_FoverlapInt, R_FoverlapIntErr = calculate_overlap_integral(data_EmsR, F_Abs_coeff, [480, 580], plot=True) # In M^-1cm^-1nm^4
R0_R_F, delta_R0_R_F = calc_R0(R_FoverlapInt, 0.97, 1.3617, delta_overlap=R_FoverlapIntErr, delta_n=0.0001) # in nm
print('J(R-F) = {:.2e} ± {:.2e} M^-1cm^-1nm^4    R0 = {:.2f} ± {:.2f} nm'.format(R_FoverlapInt, R_FoverlapIntErr, R0_R_F, delta_R0_R_F))


'''FRET Efficiency'''
F_DA = data_EmsF_R['S1c/R1'].max() # Fluorescein emission in the presence of Rose Bengal
F_D = data_EmsF['S1c/R1'].max() # Fluorescein emission without Rose Bengal
FRET_efficiency = 1 - (F_DA / F_D)
print('FRET efficiency is: {:.2f} %'.format(FRET_efficiency * 100))

# Distance between donor and acceptor
R_F_R = R0_F_R * ( (1 / FRET_efficiency) - 1 ) ** (1/6)
print('Distance between Fluorescein and Rose Bengal is: {:.2f} nm'.format(R_F_R))
