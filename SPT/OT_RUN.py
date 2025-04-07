from SPT_functions import *
json_folder = r'SPT\Json file'
dict_path = os.path.join(json_folder, 'I0_03_2025-02-26-164514-0000_tracked_particle.json')
data = read_tracked_particles(dict_path)
t, x, y = OT_data_extarction(data)

# plt.plot(t, y)
plt.plot(t, x)
dt = np.diff(t)
# print(np.mean(dt), np.std(dt))
# plt.hist(y, bins=10)
plt.show()