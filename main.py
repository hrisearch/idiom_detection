from classifier import CNN, RNN
from data_reader import datareader
import argparse

def main():
	p = argparse.ArgumentParser(description='for ML algo')
	p.add_argument('--model', required=True, help='ML algo: cnn, rnn')
	args = p.parse_args()

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

if __name__ == '__main__':
	main()