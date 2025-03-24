from SPT_functions import *

dict_path = os.path.join('SPT\Json file', 'mes2_2025-01-26-180833-0000tracked_particles_min_frames_um.json')
data = read_tracked_particles(dict_path)
data = fix_units(data)
dt = find_dt(data)

particles_dict = flip_particles_and_frmae(data) 
particles_dict = filter_particles(particles_dict, 500, pixel_to_um(60))
plot_particles_trjectory(particles_dict)
# plot_trajectory_lengths_histogram(particles_dict)
# print(particles_dict)