from TCSPC_Functions import *
import os
from scipy.constants import Avogadro
'''
Day 1
Low concentration - No FRET
cuvette length: 1 cm
'''
# Absorption
Abs_folder = r'TCSPC\Data\day1 data\Absorption'
file_pathAbsF = os.path.join(Abs_folder, 'abs_flouresciene_4.3_micM.xlsx')
file_pathAbsR = os.path.join(Abs_folder, 'abs_rose_b_6micM.xlsx')
file_pathAbsF_R = os.path.join(Abs_folder, 'abs_roseb_flourescien_together.xlsx')
data_AbsF = get_data(file_pathAbsF)
data_AbsR = get_data(file_pathAbsR)
data_AbsF_R = get_data(file_pathAbsF_R)
# plot_together(data_AbsF, data_AbsR, data_AbsF_R,Type='Abs', Title_add=' 1 cm cuvette', save=True)

# Emission
Ems_folder = r'TCSPC\Data\day1 data\Emission'
file_pathEmsF = os.path.join(Ems_folder, 'Flourescein Emission at 390 nm 4.3 um P.csv')
file_pathEmsR = os.path.join(Ems_folder, 'RoseB Emission at 390 nm 6 um P.csv')
file_pathEmsF_R = os.path.join(Ems_folder, 'RoseB 6 um and Flour 4.3 Emission at 390 nm P.csv')
data_EmsF = get_data(file_pathEmsF)
data_EmsR = get_data(file_pathEmsR)
data_EmsF_R = get_data(file_pathEmsF_R)
# plot_together(data_EmsF, data_EmsR, data_EmsF_R,Type='Ems', Title_add=' 1 cm cuvette', save=True)

# Excitation
Ext_folder = r'TCSPC\Data\day1 data\Excitation'
file_pathExtF = os.path.join(Ext_folder, 'Flourescein Excitation at 600 nm  4.3 um P.csv')
file_pathExtR = os.path.join(Ext_folder, 'RoseB Excitation at 600 nm 6 um P.csv')
file_pathExtF_R = os.path.join(Ext_folder, 'RoseB 6 um and Flour 4.3 Excitation at 600 nm P.csv')
data_ExtF = get_data(file_pathExtF)
data_ExtR = get_data(file_pathExtR)
data_ExtF_R = get_data(file_pathExtF_R)
# plot_together(data_ExtF, data_ExtR, data_ExtF_R,Type='Ext', Title_add=' 1 cm cuvette', save=True)

# Absorption and Emission
# plot_Abs_and_Ems(data_AbsF, data_AbsR, data_EmsF, data_EmsR, save=True)

'''Inner filter effect was detected, we will use smaller cuvette for the rest of the experiments'''

'''
Overlap Integral
'''
# # Fluorescein
# Fluoresceib_concentration = calc_concentration(data_AbsF, 92E3, 1) # in M, extinction coefficient of fluorescein is 92,000 M-1cm-1 and l = 1 cm
# print('Fluorescein concentration is: {:.2f} uM'.format(Fluoresceib_concentration * 1E6))
# F_Abs_coeff = calc_abs_coeff(data_AbsF,Fluoresceib_concentration, 1)
# plt.plot(F_Abs_coeff['Wavelength'], F_Abs_coeff['Abs'], label=r'Abs Coeff Fluorescein 4.24 $\mu M$')
# plt.plot(data_EmsF['Wavelength'],normalize(data_EmsF['S1c/R1']), label=r'Normalized Emission Fluorescein 4.24 $\mu M$')
# plt.vlines([490, 520], 0, max(F_Abs_coeff['Abs']), colors='r', linestyles='dashed')
# plt.legend()
# plt.xlabel('Wavelength')
# plt.show()

# F_FoverlapInt = calculate_overlap_integral(data_EmsF, F_Abs_coeff, [480, 540], plot=True)
# R0_F_F = calc_R0(F_FoverlapInt, 0.97, 1.3617)
# distance_between_molecules = 1/(Fluoresceib_concentration * Avogadro) ** (1/3) * 1E8 # in nm
# print('J(F-F) = {:.2e} M^-1cm^-1nm^4    R0 = {:.2f} nm'.format(F_FoverlapInt * 1E6, R0_F_F))
# print('Distance between molecules is: {:.2f} nm'.format(distance_between_molecules))

'''
Day 2
## Low concentration - No FRET
cuvette length: 1 mm
Checking that there is no inner filter effect
'''

# Emission
Ems_folder = r'TCSPC\Data\day2 data\Emission\cuvatte1mm'
file_pathEmsF = os.path.join(Ems_folder, 'fluor_emission390nm_4.6um_1mm.csv')
file_pathEmsR = os.path.join(Ems_folder, 'roseb_emission390nm_6um_1mm.csv')
file_pathEmsF_R = os.path.join(Ems_folder, 'roseb_fluor_emission390nm_1mm.csv')
data_EmsF = get_data(file_pathEmsF)
data_EmsR = get_data(file_pathEmsR)
data_EmsF_R = get_data(file_pathEmsF_R)
# plot_together(data_EmsF, data_EmsR, data_EmsF_R,Type='Ems', Title_add=' 1 mm cuvette', save=True)

'''Inner filter effect was not detected with 1 mm cuvette, but this cuvette broke so we improvised an 500 um cuvette'''

'''
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
concentration_F = calc_concentration(data_AbsF, 92300, 0.05) * 1E6 # in uM, Absorption coefficient of fluorescein is 92,300 M-1cm-1 and l = 0.05 cm
concentration_R = calc_concentration(data_AbsR, 90400, 0.05) * 1E6 # in uM, Absorption coefficient of Rose Bengal is 90,400 M-1cm-1 and l = 0.05 cm
plot_together(data_AbsF, data_AbsR, data_AbsF_R,Type='Abs', Title_add=' 500 um cuvette', save=True, concentrations=[concentration_F, concentration_R])

# Emission
Ems_folder = r'TCSPC\Data\day2 data\Emission\cuvette500um'
file_pathEmsF = os.path.join(Ems_folder, 'Flourescein Emission at 390 nm cuvette 500um c0.1mM.csv')
file_pathEmsR = os.path.join(Ems_folder, 'roseb_Emission at 390 nm cuvette 500um c0.1mM.csv')
file_pathEmsF_R = os.path.join(Ems_folder, 'roseb and Flu Emission at 390 nm cuvette 500um 0.05mM.csv')
data_EmsF = get_data(file_pathEmsF)
data_EmsR = get_data(file_pathEmsR)
data_EmsF_R = get_data(file_pathEmsF_R)
plot_together(data_EmsF, data_EmsR, data_EmsF_R,Type='Ems', Title_add=' 500 um cuvette', save=False, concentrations=[concentration_F, concentration_R])

# Excitation
Ext_folder = r'TCSPC\Data\day2 data\Excitation'
file_pathExtF = os.path.join(Ext_folder, 'Flourescein Excitation at 600 nm cuvette 500um c0.1mM.csv')
file_pathExtR = os.path.join(Ext_folder, 'roseb_Excitation at 600 nm cuvette 500um c0.1mM.csv')
file_pathExtF_R = os.path.join(Ext_folder, 'roesb and Flu Excitation at 600 nm cuvette 500um c0.1mM.csv')
data_ExtF = get_data(file_pathExtF)
data_ExtR = get_data(file_pathExtR)
data_ExtF_R = get_data(file_pathExtF_R)
plot_together(data_ExtF, data_ExtR, data_ExtF_R,Type='Ext', Title_add=' 500 um cuvette', save=False, concentrations=[concentration_F, concentration_R])

'''
Results:
'''

'''
Overlap Integral
'''
# Overlap Integral
## Fluorescein
# Fluorescein_concentration = calc_concentration(data_AbsF, 92E3, 1) # Absorption coefficient of fluorescein is 92,000 M-1cm-1 and l = 1 cm
# print('Fluorescein concentration is: {:.2f} uM'.format(Fluorescein_concentration * 1E6))
# F_Abs_coeff = calc_abs_coeff(data_AbsF,Fluorescein_concentration, 1)
# plt.plot(F_Abs_coeff['Wavelength'], F_Abs_coeff['Abs'], label=r'Abs Coeff Fluorescein 4.24 $\mu M$')
# plt.plot(data_EmsF['Wavelength'],normalize(data_EmsF['S1c/R1']), label=r'Normalized Emission Fluorescein 4.24 $\mu M$')
# plt.vlines([490, 520], 0, max(F_Abs_coeff['Abs']), colors='r', linestyles='dashed')
# plt.legend()
# plt.xlabel('Wavelength')
# plt.show()

# F_FoverlapInt = calculate_overlap_integral(data_EmsF, F_Abs_coeff, [480, 540], plot=True)
# R0_F_F = calc_R0(F_FoverlapInt, 0.97, 1.3617)
# print('J(F-F) = {:.2e} M^-1cm^-1nm^4    R0 = {:.2f} nm'.format(F_FoverlapInt * 1E6, R0_F_F))

