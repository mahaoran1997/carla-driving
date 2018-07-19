import os
import re
count =0
import h5py
from shutil import copyfile
from shutil import move

def rename_folder(file_list,folder_name,initial_count=0):
	count = initial_count
	for filename in file_list:
		print filename
		if "data" in filename:
			newfilename = folder_name+"/data_"+str(count).zfill(5) + ".h5"

			os.rename(folder_name +'/'+ filename,newfilename )
			count +=1




def join_folders(src_path, folder_name_vec,dest_folder,per_val_files=0.10,use_dirty_data=False):
	if not os.path.exists(src_path+dest_folder):
		os.mkdir(src_path+dest_folder)
		#os.mkdir(dest_folder +  '/SeqTrain')
		#os.mkdir(dest_folder +  '/SeqVal')
	count =0
	with open('clean_results','w') as f:
		for folder,number_files in folder_name_vec.iteritems():
			file_list = sorted(os.listdir(src_path+folder))
			count_int=0
			count_good_files=0
			for filename in file_list:
				if "data" in filename:
					print filename

					newfilename =  str(count)+"_data_"+str(count_int).zfill(5) + ".h5"
					print folder +'/'+filename
					print dest_folder +'/'+newfilename
					#if count_int > len(file_list)*per_val_files: 
					#copyfile(folder_list[count] +'/'+filename,dest_folder+'/SeqTrain/'+newfilename)
					#else:
					#	copyfile(folder_list[count] +'/'+filename,dest_folder+'/SeqVal/'+newfilename)
					copyfile(src_path+folder +'/'+filename,src_path+dest_folder+'/'+newfilename)
					count_int+=1
				if count_int >= number_files:
					break


			count+=1

	rename_folder(sorted(os.listdir(src_path+dest_folder)),src_path+dest_folder)
	

	
src_path = '/home/muellem/Downloads/Desktop/'

#RC7_w_1h_
#folder_list ={'20171220_W1_w600_1':540,'20171220_W1_w700_1':540,'20171220_W1_1':540,'20171220_W1_w900_1':540,'20171221_W1_w1000_1':540,'20171221_W1_w1100_1':540,'20171221_W1_w1200_1':540}

#RC3_p_1h_
#folder_list ={'20171220_W1_1':540,'20171221_W1_pm5_1':540,'20171220_W1_pp5_1':540}

#RC3_z_1h_
#folder_list ={'20171220_W1_1':540,'20171221_W1_z50_1':540,'20171221_W1_z150_1':540}

#RC11_wpz_1h_
#folder_list ={'20171220_W1_w600_1':540,'20171220_W1_w700_1':540,'20171220_W1_1':540,'20171220_W1_w900_1':540,'20171221_W1_w1000_1':540,'20171221_W1_w1100_1':540,'20171221_W1_w1200_1':540,'20171221_W1_pm5_1':540,'20171220_W1_pp5_1':540,'20171221_W1_z50_1':540,'20171221_W1_z150_1':540}

#RC7_w_2h_
#folder_list ={'20171220_W1_w600_1':1080,'20171220_W1_w700_1':1080,'20171220_W1_1':1080,'20171220_W1_w900_1':1080,'20171221_W1_w1000_1':1080,'20171221_W1_w1100_1':1080,'20171221_W1_w1200_1':1080}

#RC3_p_2h_
#folder_list ={'20171220_W1_1':1080,'20171221_W1_pm5_1':540,'20171223_W1_pm5_1':540,'20171220_W1_pp5_1':1080}

#RC3_z_2h_
#folder_list ={'20171220_W1_1':1080,'20171221_W1_z50_1':540,'20171221_W1_z150_1':540,'20171224_W1_z50_1':540,'20171224_W1_z150_1':540}

#RC11_wpz_2h_
#folder_list ={'20171220_W1_w600_1':1080,'20171220_W1_w700_1':1080,'20171220_W1_1':1080,'20171220_W1_w900_1':1080,'20171221_W1_w1000_1':1080,'20171221_W1_w1100_1':1080,'20171221_W1_w1200_1':1080,'20171221_W1_pm5_1':540,'20171223_W1_pm5_1':540,'20171220_W1_pp5_1':1080,'20171221_W1_z50_1':540,'20171221_W1_z150_1':540,'20171224_W1_z50_1':540,'20171224_W1_z150_1':540}

#RC6_pz_1h
#folder_list ={'RC3_p_1h':1620,'RC3_z_1h':1620}

#RC6_pz_2h
#folder_list ={'RC3_p_2h':3240,'RC3_z_2h':3240}

#RC3_p_1h_N_
#folder_list ={'20171226_W1_N_1_clean':540,'20171226_W1_pm5_N_1_clean':540,'20171226_W1_pp5_N_1_clean':540}

#RC3_z_1h_N_
#folder_list ={'20171226_W1_N_1_clean':540,'20171227_W1_z50_N_1_clean':540,'20171227_W1_z150_N_1_clean':540}

#RC6_pz_1h_N
#folder_list ={'RC3_p_1h_N':3240,'RC3_z_1h_N':3240}

#RC17_wpz_M
#folder_list ={'RC11_wpz_1h':5940,'RC6_pz_1h_N':3240}

#RC28_wpz_M
#folder_list ={'RC11_wpz_2h':11880,'RC6_pz_1h_N':3240}

#RC20_wpz_M
#folder_list ={'RC11_wpz_1h':5940,'RC6_pz_1h_N':3240,'RSS_W3_1h_WP':540,'RSS_W6_1h_WP':540,'RSS_W8_1h_WP':540}

#RC27_wpz_M
#folder_list ={'RSS_W1_1h_N_WP':540,'RSS_W1_2h_WP':1080,'RSS_W1_pm5_1h_N_WP':540,'RSS_W1_pm5_2h_WP':1080,'RSS_W1_pp5_1h_N_WP':540,'RSS_W1_pp5_2h_WP':1080,'RSS_W1_w600_2h_WP':1080,'RSS_W1_w700_2h_WP':1080,'RSS_W1_w900_2h_WP':1080,'RSS_W1_w1000_2h_WP':1080,'RSS_W1_w1100_2h_WP':1080,'RSS_W1_w1200_2h_WP':1080,'RSS_W1_z50_1h_N_WP':540,'RSS_W1_z50_2h_WP':1080,'RSS_W1_z150_1h_N_WP':540,'RSS_W1_z150_2h_WP':1080}

#RC28_wpz_M_DR
folder_list ={'W1_1':90,'W1_N_1':45,'W1_pm5_1':90,'W1_pm5_N_1':45,'W1_pp5_1':90,'W1_pp5_N_1':45,'W1_w600_1':90,'W1_w700_1':90,'W1_w900_1':90,'W1_w1000_1':90,'W1_w1100_1':90,'W1_w1200_1':90,'W1_z50_1':90,'W1_z50_N_1':45,'W1_z150_1':90,'W1_z150_N_1':45,
'W2_1':90,'W2_N_1':45,'W2_pm5_1':90,'W2_pm5_N_1':45,'W2_pp5_1':90,'W2_pp5_N_1':45,'W2_w600_1':90,'W2_w700_1':90,'W2_w900_1':90,'W2_w1000_1':90,'W2_w1100_1':90,'W2_w1200_1':90,'W2_z50_1':90,'W2_z50_N_1':45,'W2_z150_1':90,'W2_z150_N_1':45,
'W3_1':90,'W3_N_1':45,'W3_pm5_1':90,'W3_pm5_N_1':45,'W3_pp5_1':90,'W3_pp5_N_1':45,'W3_w600_1':90,'W3_w700_1':90,'W3_w900_1':90,'W3_w1000_1':90,'W3_w1100_1':90,'W3_w1200_1':90,'W3_z50_1':90,'W3_z50_N_1':45,'W3_z150_1':90,'W3_z150_N_1':45,
'W5_1':90,'W5_N_1':45,'W5_pm5_1':90,'W5_pm5_N_1':45,'W5_pp5_1':90,'W5_pp5_N_1':45,'W5_w600_1':90,'W5_w700_1':90,'W5_w900_1':90,'W5_w1000_1':90,'W5_w1100_1':90,'W5_w1200_1':90,'W5_z50_1':90,'W5_z50_N_1':45,'W5_z150_1':90,'W5_z150_N_1':45,
'W6_1':90,'W6_N_1':45,'W6_pm5_1':90,'W6_pm5_N_1':45,'W6_pp5_1':90,'W6_pp5_N_1':45,'W6_w600_1':90,'W6_w700_1':90,'W6_w900_1':90,'W6_w1000_1':90,'W6_w1100_1':90,'W6_w1200_1':90,'W6_z50_1':90,'W6_z50_N_1':45,'W6_z150_1':90,'W6_z150_N_1':45,
'W7_1':90,'W7_N_1':45,'W7_pm5_1':90,'W7_pm5_N_1':45,'W7_pp5_1':90,'W7_pp5_N_1':45,'W7_w600_1':90,'W7_w700_1':90,'W7_w900_1':90,'W7_w1000_1':90,'W7_w1100_1':90,'W7_w1200_1':90,'W7_z50_1':90,'W7_z50_N_1':45,'W7_z150_1':90,'W7_z150_N_1':45,
'W8_1':90,'W8_N_1':45,'W8_pm5_1':90,'W8_pm5_N_1':45,'W8_pp5_1':90,'W8_pp5_N_1':45,'W8_w600_1':90,'W8_w700_1':90,'W8_w900_1':90,'W8_w1000_1':90,'W8_w1100_1':90,'W8_w1200_1':90,'W8_z50_1':90,'W8_z50_N_1':45,'W8_z150_1':90,'W8_z150_N_1':45,
'W9_1':90,'W9_N_1':45,'W9_pm5_1':90,'W9_pm5_N_1':45,'W9_pp5_1':90,'W9_pp5_N_1':45,'W9_w600_1':90,'W9_w700_1':90,'W9_w900_1':90,'W9_w1000_1':90,'W9_w1100_1':90,'W9_w1200_1':90,'W9_z50_1':90,'W9_z50_N_1':45,'W9_z150_1':90,'W9_z150_N_1':45,
'W10_1':90,'W10_N_1':45,'W10_pm5_1':90,'W10_pm5_N_1':45,'W10_pp5_1':90,'W10_pp5_N_1':45,'W10_w600_1':90,'W10_w700_1':90,'W10_w900_1':90,'W10_w1000_1':90,'W10_w1100_1':90,'W10_w1200_1':90,'W10_z50_1':90,'W10_z50_N_1':45,'W10_z150_1':90,'W10_z150_N_1':45,
'W11_1':90,'W11_N_1':45,'W11_pm5_1':90,'W11_pm5_N_1':45,'W11_pp5_1':90,'W11_pp5_N_1':45,'W11_w600_1':90,'W11_w700_1':90,'W11_w900_1':90,'W11_w1000_1':90,'W11_w1100_1':90,'W11_w1200_1':90,'W11_z50_1':90,'W11_z50_N_1':45,'W11_z150_1':90,'W11_z150_N_1':45,
'W12_1':90,'W12_N_1':45,'W12_pm5_1':90,'W12_pm5_N_1':45,'W12_pp5_1':90,'W12_pp5_N_1':45,'W12_w600_1':90,'W12_w700_1':90,'W12_w900_1':90,'W12_w1000_1':90,'W12_w1100_1':90,'W12_w1200_1':90,'W12_z50_1':90,'W12_z50_N_1':45,'W12_z150_1':90,'W12_z150_N_1':45,
'W13_1':90,'W13_N_1':45,'W13_pm5_1':90,'W13_pm5_N_1':45,'W13_pp5_1':90,'W13_pp5_N_1':45,'W13_w600_1':90,'W13_w700_1':90,'W13_w900_1':90,'W13_w1000_1':90,'W13_w1100_1':90,'W13_w1200_1':90,'W13_z50_1':90,'W13_z50_N_1':45,'W13_z150_1':90,'W13_z150_N_1':45}

print folder_list


join_folders(src_path, folder_list,'RC28_wpz_M_DR')




