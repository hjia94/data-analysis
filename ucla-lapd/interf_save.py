import h5py
import numpy as np
import time
import datetime

from read_scope_data import read_trc_data_simplified
from interf_raw import density_from_phase
import os

#===============================================================================================================================================
file_path = "/home/interfpi/"
shot_number = 0
#===============================================================================================================================================

def hdf5_file(filename):

	f = h5py.File(filename, "w")
	neA_group = f.create_group("ne_p20")
	neB_group = f.create_group("ne_p29")
	tarr_group = f.create_group("time_array")

	if os.path.exists(filename):
		print("HDF5 file opened ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	else:
		print("HDF5 file created ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

	return f, neA_group, neB_group, tarr_group


def create_sourcefile_dataset(grp, data, saved_time):
	""" worker for below:
		create an HDF5 dataset containing the contents of the specified file
		name and modified time
	"""
	fds_name = saved_time
	fds = grp.create_dataset(fds_name, data=data)
	fds.attrs['modified'] = time.ctime(fds_name)


def get_density_data(shot_number):
	ifn = f"{file_path}C1-interf-shot{shot_number:05d}.trc"
	refchA, tarr = read_trc_data_simplified(ifn)

	ifn = f"{file_path}C2-interf-shot{shot_number:05d}.trc"
	plachA, tarr = read_trc_data_simplified(ifn)

	ifn = f"{file_path}C3-interf-shot{shot_number:05d}.trc"
	refchB, tarr = read_trc_data_simplified(ifn)

	ifn = f"{file_path}C4-interf-shot{shot_number:05d}.trc"
	plachB, tarr = read_trc_data_simplified(ifn)

	t_ms, neA = density_from_phase(tarr, refchA, plachA)
	t_ms, neB = density_from_phase(tarr, refchB, plachB)

	saved_time = os.path.getmtime(ifn)
	
	return t_ms, neA, neB, saved_time

#===============================================================================================================================================
def main():
	# Create an HDF5 file to store the data
	today = datetime.date.today()
	filename = f"interf_data_{today}.hdf5"
	f, neA_group, neB_group, tarr_group = hdf5_file(filename)

	while True:
		ifn = f"{file_path}C1-interf-shot{shot_number:05d}.trc"
		if not os.path.exists(ifn):
			continue

		try :
			t_ms, neA, neB, saved_time = get_density_data(shot_number)

			create_sourcefile_dataset(neA_group, neA, saved_time)
			create_sourcefile_dataset(neB_group, neB, saved_time)
			create_sourcefile_dataset(tarr_group, t_ms, saved_time)

			print("Interferometer shot at", time.ctime(saved_time))
			
			shot_number += 1

		except KeyboardInterrupt:
			print("Keyboard interrupt detected. Exiting...")
			break
		except Exception as e:
			print("Error: ", e)
			break

	
#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':

	main()
