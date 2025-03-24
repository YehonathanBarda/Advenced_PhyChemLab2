import os
import json
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks, peak_widths
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.constants import Boltzmann


def read_tracked_particles(file_path):
    """
    Read the tracked particles from a json file
    Dict format: {frame:{particle:(x,y)}
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def pixel_to_um(data, pixel_to_um = 0.005787): # 1 pixel = 0.05787 um
    if type(data) is int:
        return data * pixel_to_um
    else:
        converted_data = {}
        for frame, particles in data.items():
            converted_data[frame] = {}
            for particle, (x, y) in particles.items():
                converted_data[frame][particle] = (x * pixel_to_um, y * pixel_to_um)
        return converted_data
def fix_units(data, ratio = 10): # 10 um / 1 um = 10
    if type(data) is float or type(data) is int:
        return data * ratio
    else:
        converted_data = {}
        for frame, particles in data.items():
            converted_data[frame] = {}
            for particle, (x, y) in particles.items():
                converted_data[frame][particle] = (x * ratio, y * ratio)
        return converted_data

def um_to_pixel(data, um_to_pixel = 1/0.005787): # 1 pixel = 0.05787 um
    converted_data = {}
    for frame, particles in data.items():
        converted_data[frame] = {}
        for particle, (x, y) in particles.items():
            converted_data[frame][particle] = (x * um_to_pixel, y * um_to_pixel)
    return converted_data

def flip_particles_and_time(data, frame_cut = np.inf):
    """
    Flip the dict from {frame:{particle:(x,y)}} to {particle:{frame:(x,y)}}
    """
    particles = {}
    # for frame, particles_data in data.items():
    #     for particle, (x, y) in particles_data.items():
    #         if particle not in particles:
    #             particles[particle] = []
    #         particles[particle].append([float(frame), x, y])

    for frame, particles_data in data.items():
        if float(frame) > frame_cut:
            break
        for particle in sorted(particles_data.keys(), key=lambda x: int(x)):
            if particle not in particles:
                particles[particle] = []
            particles[particle].append([float(frame), particles_data[particle][0], particles_data[particle][1]])
    return particles

def flip_particles_and_frmae(data, time_cut = np.inf):
    """
    Flip the dict from {frame:{particle:(x,y)}} to {particle:{frame:(x,y)}}
    """
    particles = {}
    frame_num = 0
    last_frame = -1
    for time, particles_data in sorted(data.items(), key = lambda x: float(x[0])):
        if float(time) > time_cut:
            break

        if last_frame >= frame_num:
            raise ValueError(f'Frame number is not increasing: {last_frame} -> {frame_num}')
        last_frame = frame_num
        
        for particle in sorted(particles_data.keys(), key=lambda x: int(x)):
            if particle not in particles:
                particles[particle] = []
            particles[particle].append([frame_num, particles_data[particle][0], particles_data[particle][1]])
        frame_num += 1
    return particles

def gen_particles_array(particles_dict):
    """
    Convert the dict to a 3D numpy array with shape (n_particles, n_frames)
    array looks like:
    [[[t1, x1_p1, y1_p1], [t2, x2_p1, y2_p1], ...],
     [[t1, x1_p2, y1_p2], [t2, x2_p2, y2_p2], ...],
     ...
     [[t1, x1_pn, y1_pn], [t2, x2_pn, y2_pn], ...]]
    """
    particles_array = []
    for particle, positions in particles_dict.items():
        particles_array.append(positions)
    return np.array(particles_array)
    # return particles_array

def calculate_displacements(particles_dict):
    displacements_dict = {}
    for particle, positions in particles_dict.items():
        positions = sorted(positions)  # Sort by frame
        Nframe = len(positions) # Number of frames for this particle
        displacements = np.zeros((3,Nframe))
        for frame in range(Nframe):
            displacements[:,frame] = np.array(positions[frame]) - np.array(positions[0])
        displacements_dict[particle] = displacements
    return displacements_dict

def plot_particles_trjectory(particles_dict, particles_to_plot=None):
    count_particles = 0
    for particle in particles_dict.keys():
        if particles_to_plot is not None and particle not in particles_to_plot:
            continue
        if len(particles_dict[particle]) < 15:
            continue
        count_particles += 1
        # if count_particles > 100 and count_particles < 150:
        #     break
        particle_data = particles_dict[particle]
        particle_data = np.array(particle_data)
        x_positions = particle_data[:, 1]
        y_positions = particle_data[:, 2]

        # if max(x_positions) - min(x_positions) < 5 or max(y_positions) - min(y_positions) < 5:
        #     print('particle too small')
        #     continue
        plt.plot(x_positions, y_positions, label=f'Particle {particle}')
        
            # plt.plot(np.mean(x_positions), np.mean(y_positions), 'bo', alpha=0.2, markersize=10)
            # circle1 = plt.Circle((np.mean(x_positions), np.mean(y_positions)), 50, color='b', fill=True, alpha=0.1)
            # circle2 = plt.Circle((np.mean(x_positions), np.mean(y_positions)), 2, color='r', fill=True, alpha=0.2)
            # plt.gca().add_patch(circle1)
            # plt.gca().add_patch(circle2)
            # print('particle too small')


        # if np.std(np.diff(x_positions)) < 2 or np.std(np.diff(y_positions)) < 2:
        #     # plt.plot(np.mean(x_positions), np.mean(y_positions[0]), 'ro', alpha=0.2)
        #     # plt.plot(x_positions[-1], y_positions[-1], 'bo')
        #     print(np.diff(x_positions))
        #     print(np.diff(y_positions))

    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Trajectory of particles', fontsize=16)
    # plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def filter_particles(particles_dict, min_frames, min_displacement, min_avg_step = None):
    filtered_particles = {}
    for particle, positions in particles_dict.items():
        if len(positions) >= min_frames:
            particle_data = np.array(positions)
            x_positions = particle_data[:, 1]
            y_positions = particle_data[:, 2]
            # if min_avg_step is not None:
            #     if np.mean(abs(np.diff(x_positions))) < 0.5 and np.mean(abs(np.diff(y_positions))) < min_avg_step:
            #         print(np.mean(abs(np.diff(x_positions))))
            if max(x_positions) - min(x_positions) >= min_displacement and max(y_positions) - min(y_positions) >= min_displacement:
                filtered_particles[particle] = positions
    return filtered_particles

def combin_displacements(displacements_dict):
    longest_time = 0
    longest_time_particle = ''  # The particle with the longest time
    for particle, displacements in displacements_dict.items():
        if displacements.shape[1] > longest_time:
            longest_time = displacements.shape[1]
            longest_time_particle = particle
    time_array = displacements_dict[longest_time_particle][0]
    combined_displacements = np.full((len(displacements_dict), 2, longest_time), np.nan)
    for i, (particle, displacements) in enumerate(displacements_dict.items()):
        combined_displacements[i, :, :displacements.shape[1]] = displacements[1:]
    return time_array, combined_displacements

def save_displacements_to_csv(combined_displacements, filename='displacements.csv'):
    """
    Save the x displacements of the particles to a CSV file
    """
    x_displacements = combined_displacements[:, 0, :]
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Particle'] + [f'Time_{i}' for i in range(x_displacements.shape[1])])
        for i, row in enumerate(x_displacements):
            csvwriter.writerow([f'Particle_{i+1}'] + row.tolist())

def calc_MSD(combined_displacements):
    """
    Calculate the mean square displacement (MSD) of the particles at each time point
    """
    r_squared = np.sum(combined_displacements ** 2, axis=1)
    MSD = np.nanmean(r_squared, axis=0)
    return MSD

def calc_diffusion_coefficient(MSD, time_array):
    """
    Calculate the diffusion coefficient from the MSD
    """

    popt, pcov = curve_fit(linear_func, time_array, MSD)
    D = popt[0] / 4  # Diffusion coefficient MDS=2nDt / 4 (2 Dimensions)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation errors
    Derr = perr[0] / 4

    plt.plot(time_array, MSD, '.', label='MSD')
    plt.plot(time_array, linear_func(time_array, *popt), '-', label=f'Linear Fit')
    plt.xlabel('Time [s]')
    plt.ylabel(r'MSD [$\mu m^2$]')
    plt.legend()
    print(f'Diffusion coefficient: {D:.2e} ± {Derr:.2e} um^2/s')
    plt.show()
    return D, Derr

def linear_func(x, a, b):
    return a * x + b

def plot_trajectory_lengths_histogram(particles_dict):
    """
    Plot a histogram of the lengths of all particle trajectories
    """
    trajectory_lengths = [len(positions) for positions in particles_dict.values()]
    
    plt.hist(trajectory_lengths, bins=20, edgecolor='black')
    plt.xlabel('Trajectory Length (number of frames)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Particle Trajectory Lengths')
    plt.grid(True)
    plt.show()

def find_dt(data,plot = False):
    """
    Find the time difference between each frame
    """
    dt = []
    for frame, particles in data.items():
        dt.append(float(frame))
    dt.sort()
    dt = np.diff(dt)
    if plot:
        plt.hist(dt, bins=50, edgecolor='black')
        plt.xlabel('Time difference between frames (s)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time Differences Between Frames')
        plt.grid(True)
        plt.show()
        print(f'Mean time difference between frames: {np.mean(dt):.5f} ± {np.std(dt):.5f} s')
    return np.mean(dt)

def OT_data_extarction(data):
    """
    Extract the x, y position and time of the particles from the OT data
    """
    t = []
    x = []
    y = []
    for time, position in data.items():
            t.append(float(time))
            x.append(fix_units(position[0]))
            y.append(fix_units(position[1]))
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    t = np.array(t)
    return t, x, y

def FT_calc(x_data,y_data, N= 10000):
    inter = CubicSpline(x_data, y_data)
    x = np.linspace(x_data[0], x_data[-1], N, endpoint=True)
    y = inter(x)

    N = N
    T = abs(x[1] - x[0])
    yf = np.abs(fft(y)[:N//2])
    xf = fftfreq(N,T)[:N//2]
    Tf = abs(xf[1] - xf[0])
    try:
        peaks, _ = find_peaks(yf, height = 1e5, distance = 150, width = 2)
        results_half = peak_widths(yf, peaks, rel_height = 0.5)
        xf_max = xf[peaks]
        yf_max = yf[peaks]
        xf_err = results_half[0] * Tf
        FWHM_data = (results_half[1], results_half[2] * Tf, results_half[3] * Tf)

        return xf, yf, xf_max, xf_err, yf_max, FWHM_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return xf, yf, None, None, None, None

def k_with_equipatiton(x,t):
    m = 1 # mass in kg
    v = np.diff(x) / np.diff(t)
    Kinetic_energy = 0.5 * m * np.mean(v ** 2)
    Total_energy = Boltzmann * 300 # Temp 300 K
    k = (Total_energy - Kinetic_energy)/(0.5 * np.mean(x ** 2))
    if True:
        print(f'Kinetic energy: {Kinetic_energy:.3e} J')
        print(f'Total energy: {Total_energy:.3e} J')
        print(f'K_B T: {Boltzmann * 300:.3e} J')
        print(f'k: {k:.3e} N/m')


# if __name__ == "__main__":
    # dict_path = os.path.join('SPT\Json file', 'mes2_2025-01-26-180833-0000tracked_particles_min_frames_um.json')
    # data = read_tracked_particles(dict_path)
    # data = fix_units(data)

    # # data = um_to_pixel(data)
    # particles_dict = flip_particles_and_frmae(data, frame_cut=270)
    # # particles_dict = filter_particles(particles_dict, 50, pixel_to_um(10))
    # # particles_dict = filter_particles(particles_dict, 100, 500)


    # plot_particles_trjectory(particles_dict)

    # displacements_dict = calculate_displacements(particles_dict)
    # time_array, combined_displacements = combin_displacements(displacements_dict)
    # MSD = calc_MSD(combined_displacements)
    # n = 500
    # relevant_time = time_array[:n]
    # relevant_msd = MSD[:n]
    # D, Derr = calc_diffusion_coefficient(relevant_msd, relevant_time)
    # results_filename = 'SPT\Diffusion_results.txt'
    # with open(results_filename, 'a') as file:
    #     x = False
    #     # x =True
    #     if x:
    #         file.write(f'{os.path.basename(dict_path)}:\nD = {D:.2e} ± {Derr:.2e} um^2/s\n')

    
    
    # # delta_t = time_array[1] - time_array[0]
    # # save_displacements_to_csv(combined_displacements)
    # # plt.plot(relevant_time, relevant_msd)
    # # plt.xlabel('Time [sec]')
    # # plt.ylabel(r'MSD [$\mu m ^2$]')
    # # plt.title('Mean square displacement')
    # # plt.show()
