import csv

def datareader(datadict, filename):
	with open(filename) as f:
		cf = csv.reader(f, delimiter=',')
		i = 0
		for line in cf:
			print(line)
			if line[0] == 'DataID' or line[0] == 'ID':
				continue
			datadict[str(i)] = {}
			datadict[str(i)]['0'] = line[4]
			datadict[str(i)]['1'] = line[5]
			datadict[str(i)]['2'] = line[6]
			if int(line[7]) == 1:
				datadict[str(i)]['label'] = [0, 1]
			else:
				datadict[str(i)]['label'] = [1, 0]
			i+=1


def datareaderb(datadict, filename):
	with open(filename) as f:
		cf = csv.reader(f, delimiter=',')
		i = 0
		for line in cf:
#			print(line[6])
			if line[0] == 'DataID' or line[0] == 'ID':
				continue
			datadict[str(i)] = {}
			datadict[str(i)]['0'] = line[4]
			datadict[str(i)]['1'] = line[5]
			if (line[6]) != 'None':
				datadict[str(i)]['label'] = [0, 1]
			else:
				datadict[str(i)]['label'] = [1, 0]
			i+=1
			if i == 512:
				return

