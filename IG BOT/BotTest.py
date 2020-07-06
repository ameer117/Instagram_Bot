import numpy as np
import pickle
import tensorflow as tf
import numpy as np
import random
from collections import Counter
import itertools

def idsToSentence(ids, wList):
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    myStr = ""
    listOfResponses=[]
    for num in ids:
        #print(num[0])
        #print(wList[num[0]])
        if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
            listOfResponses.append(myStr)
            myStr = ""
        else:
            myStr = myStr + (wList[num[0]]) + " "
    if myStr:
        listOfResponses.append(myStr)
    listOfResponses = [i for i in listOfResponses if i]
    listOfResponses = list(set(listOfResponses))
    #chosenString = ''.join(listOfResponses)
    chosenString = random.choice(listOfResponses)
    #chosenString = max(listOfResponses, key=len)
    return chosenString

def getTestInput(inputMessage, wList, maxLen):
	encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		try:
			encoderMessage[index] = wList.index(word)
		except ValueError:
			continue
	encoderMessage[index + 1] = wList.index('<EOS>')
	encoderMessage = encoderMessage[::-1]
	encoderMessageList=[]
	for num in encoderMessage:
		encoderMessageList.append([num])
	return encoderMessageList

def processDataset(filename):
    openedFile = open(filename, 'r')
    allLines = openedFile.readlines()
    myStr = ""
    for line in allLines:
        myStr += line
    finalDict = Counter(myStr.split())
    #finalDict = {key:val for key, val in finalDict.items() if val != 1} #remove words that appear only once
    return finalDict

# with open("wordList.txt", "r") as fp:
#     wordList = pickle.load(fp)

datasetDictionary = processDataset('conversationData.txt') #full corpus is not used
print ('Finished parsing and cleaning dataset')
wordList = list(datasetDictionary.keys()) #word list is list of all the words. full corpus is the text file to a string

#wordList.insert(0, "<UNK>") #0 element is unknown
wordList.append('<pad>')
wordList.append('<EOS>')
print(wordList)
vocabSize = len(wordList)
print(vocabSize)
tf.reset_default_graph()

batchSize = 24
maxEncoderLength = 15
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000


# Create placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)
#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
                                                            vocabSize, vocabSize, lstmUnits, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

# Start session and get graph
sess = tf.Session()
#y, variables = model.getModel(encoderInputs, decoderLabels, decoderInputs, feedPrevious)

# Load in pretrained model
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models'))
saver.restore(sess, "models/pretrained_seq2seq.ckpt-10000")
zeroVector = np.zeros((1), dtype='int32')
#pretrained_seq2seq.ckpt-190000.index

while (True):
    inputString = input("Type a message: ") 
    inputVector = getTestInput(inputString, wordList, maxEncoderLength)
    feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    ids = (sess.run(decoderPrediction, feed_dict=feedDict))
    print(str(idsToSentence(ids, wordList)))

