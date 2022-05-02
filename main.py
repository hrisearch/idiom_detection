from classifier import CNN, RNN, Siamese
from data_reader import datareader, datareaderb
import argparse

def main():
	p = argparse.ArgumentParser(description='for ML algo')
	p.add_argument('--model', required=True, help='ML algo: cnn, rnn')
	p.add_argument('--task', required=True, help='a or b')
	args = p.parse_args()


	if args.task == 'a':
		traindata = {}
		datareader(traindata, 'data1.csv')
		testdata = {}
		datareader(testdata, 'data2.csv')

		if args.model == 'cnn':
			cnn = CNN(traindata)
			cnn.train()
			cnn.test(testdata)
		elif args.model == 'rnn':
			rnn = RNN(traindata)
			rnn.train()
			rnn.test(testdata)

	elif args.task == 'b':
		traindata = {}
		datareaderb(traindata, 'db1.csv')
		testdata = {}
		datareaderb(testdata, 'db2.csv')

		sia = Siamese(traindata)
		sia.train()
		sia.test(testdata)


if __name__ == '__main__':
	main()