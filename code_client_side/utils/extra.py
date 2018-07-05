
import os

def get_latest_file_number(path,name):
	full_name = path + name 
	current_file = 1
	while os.path.exists(full_name + '_' + str(current_file)):
		current_file+=1

	return current_file
		