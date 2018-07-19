import csv
import os

exp = 'TrainT_TrainW1'
#exp = 'TrainT_TestW4'
#exp = 'TestT_TrainW1'
#exp = 'TestT_TestW4'

src_path = '/media/matthias/7E0CF8640CF818BB/Github/carla_chauffeur/results/RSS/250k'
csv_file = 'RSS_250k_' + exp + '.csv'
fname_filter = exp + '.summary'
file_list = sorted(os.listdir(src_path))

with open(csv_file, 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['Model'] + ['Success'] + ['Trespass'] + ['Collision'] + ['Infraction'])

	for fname in file_list:
	    if fname_filter in fname:
		print 'Processing:', fname
		with open(os.path.join(src_path, fname)) as f:
		    content = f.readlines()
		    # you may also want to remove whitespace characters like `\n` at the end of each line
		    content = [x.strip() for x in content] 

		    success = content[3][35:40]
		    print content[3][16:25], success

		    trespass = content[13][29:38]
		    print content[13][19:27], trespass

		    collision = content[14][30:39]
		    print content[14][19:28], collision

		    infraction = content[15][31:40]
		    print content[15][19:29], infraction

		    writer.writerow([fname, success, trespass, collision, infraction])


