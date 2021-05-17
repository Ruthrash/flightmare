file_path = "/home/ruthz/dodge_drone_challenge/flightmare/flightrl/example.log"
viz_dict = dict()

with open(file_path) as fp:
	for line in fp: 
		words = line.split(':')[-1]
		name = words.split(' ')[0]
		word = words.split(' ')[-1]
		if name[0:-1] == "onpolicy":
			viz_dict(name) = dict()		
			obs = word.split('\n')[0].split(',')
			obs = [float(ob) for ob in obs]
			viz_dict
