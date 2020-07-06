import tensorflow as tf
import numpy as np
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os
from collections import Counter
import itertools


def splitDict(d):
    n = (4*len(d)) //  5         # length of smaller half
    i = iter(d.items())      # alternatively, i = d.iteritems() works in Python 2

    d1 = dict(itertools.islice(i, n))   # grab first n items
    d2 = dict(i)                        # grab the rest

    return d1, d2

def createTrainingMatrices(dic, wList, maxLen):
    conversationDictionary = dic
    numExamples = len(dic) #num examples of training data
    xTrain = np.zeros((numExamples, maxLen), dtype='int32')
    yTrain = np.zeros((numExamples, maxLen), dtype='int32')
    for index,(key,value) in enumerate(conversationDictionary.items()):
        # Will store integerized representation of strings here (initialized as padding)
        encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
        decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
        # Getting all the individual words in the strings
        keySplit = key.split()
        valueSplit = value.split()
        keyCount = len(keySplit)
        valueCount = len(valueSplit)
        # Throw out sequences that are too long or are empty
        if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1) or valueCount == 0 or keyCount == 0):
            continue
        # Integerize the encoder string
        for keyIndex, word in enumerate(keySplit):
            try:
                encoderMessage[keyIndex] = wList.index(word)
            except ValueError:
                encoderMessage[keyIndex] = 0 #this means it is <UNK>
        encoderMessage[keyIndex + 1] = wList.index('<EOS>')
        # Integerize the decoder string
        for valueIndex, word in enumerate(valueSplit):
            try:
                decoderMessage[valueIndex] = wList.index(word)
            except ValueError:
                decoderMessage[valueIndex] = 0 #this means it is <UNK>
        decoderMessage[valueIndex + 1] = wList.index('<EOS>')
        xTrain[index] = encoderMessage
        yTrain[index] = decoderMessage
    # Remove rows with all zeros
    yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
    xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
    numExamples = xTrain.shape[0]
    return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen, numexamples):
    num = randint(0,numexamples - localBatchSize - 1)
    arr = localXTrain[num:num + localBatchSize]
    labels = localYTrain[num:num + localBatchSize]
    # Reversing the order of encoder string apparently helps as per 2014 paper
    reversedList = list(arr)
    for index,example in enumerate(reversedList):
        reversedList[index] = list(reversed(example))

    # Lagged labels are for the training input into the decoder
    laggedLabels = []
    EOStokenIndex = wordList.index('<EOS>')
    padTokenIndex = wordList.index('<pad>')
    for example in labels:
        eosFound = np.argwhere(example==EOStokenIndex)[0]
        shiftedExample = np.roll(example,1)
        shiftedExample[0] = EOStokenIndex
        # The EOS token was already at the end, so no need for pad
        if (eosFound != (maxLen - 1)):
            shiftedExample[eosFound+1] = padTokenIndex
        laggedLabels.append(shiftedExample)

    # Need to transpose these
    reversedList = np.asarray(reversedList).T.tolist()
    labels = labels.T.tolist()
    laggedLabels = np.asarray(laggedLabels).T.tolist()
    return reversedList, labels, laggedLabels

def translateToSentences(inputs, wList, encoder=False):
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    numStrings = len(inputs[0])
    numLengthOfStrings = len(inputs)
    listOfStrings = [''] * numStrings
    for mySet in inputs:
        for index,num in enumerate(mySet):
            if (num != EOStokenIndex and num != padTokenIndex):
                if (encoder):
                    # Encodings are in reverse!
                    listOfStrings[index] = wList[num] + " " + listOfStrings[index]
                else:
                    listOfStrings[index] = listOfStrings[index] + " " + wList[num]
    listOfStrings = [string.strip() for string in listOfStrings]
    return listOfStrings

def getTestInput(inputMessage, wList, maxLen):
    encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
    inputSplit = inputMessage.lower().split()
    for index,word in enumerate(inputSplit):
        try:
            encoderMessage[index] = wList.index(word) #finds the number mapped to the word in encoderMessage[index]
        except ValueError:
            continue
    encoderMessage[index + 1] = wList.index('<EOS>')
    encoderMessage = encoderMessage[::-1]
    encoderMessageList=[]
    for num in encoderMessage:
        encoderMessageList.append([num])
    return encoderMessageList

def idsToSentence(ids, wList):
    #TODO: don't output UNKS
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    myStr = ""
    listOfResponses=[]
    for num in ids:
        if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
            listOfResponses.append(myStr)
            myStr = ""
        else:
            myStr = myStr + wList[num[0]] + " "
    if myStr:
        listOfResponses.append(myStr)
    listOfResponses = [i for i in listOfResponses if i]
    return listOfResponses

def processDataset(filename):
    openedFile = open(filename, 'r')
    allLines = openedFile.readlines()
    myStr = ""
    for line in allLines:
        myStr += line
    finalDict = Counter(myStr.split())
    finalDict = {key:val for key, val in finalDict.items() if val != 1} #remove words that appear only once
    return finalDict

##################################################### BEGIN #################################################

# Hyperparamters
batchSize = 24
maxEncoderLength = 15
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000

#IF WE ARE NOT LOADING 
datasetDictionary = processDataset('conversationData.txt') #full corpus is not used
print ('Finished parsing and cleaning dataset')
#only use words that appear more than once. last index is UNK
wordList = list(datasetDictionary.keys()) #word list is list of all the words. full corpus is the text file to a string
wordList.insert(0, "<UNK>") #0 element is unknown
print(wordList[0])
vocabSize = len(wordList) #vocab size is all words that appear more than once

# If you've run the entirety of word2vec.py then these lines will load in
# the embedding matrix.
if (os.path.isfile('embeddingMatrix.npy')):
    wordVectors = np.load('embeddingMatrix.npy')
    wordVecDimensions = wordVectors.shape[1]
else:
    print("making word vectors")
    wordVecDimensions = 100

# Add two entries to the word vector matrix. One to represent padding tokens,
# and one to represent an end of sentence token
padVector = np.zeros((1, wordVecDimensions), dtype='int32')
EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
if (os.path.isfile('embeddingMatrix.npy')):
    wordVectors = np.concatenate((wordVectors,padVector), axis=0)
    wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

# Need to modify the word list as well
wordList.append('<pad>')
wordList.append('<EOS>')
vocabSize = vocabSize + 2
if (os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy')):
    xTrain = np.load('Seq2SeqXTrain.npy')
    yTrain = np.load('Seq2SeqYTrain.npy')
    xTest = np.load('Seq2SeqXTest.npy')
    yTest = np.load('Seq2SeqYTest.npy')
    print ('Finished loading training matrices')
    numTrainingExamples = xTrain.shape[0]
    numTestExamples = xTest.shape[0]
else:
    conversationDictionary = np.load('conversationDictionary.npy', allow_pickle = True).item()
    train, test = splitDict(conversationDictionary) #splits dictionary into train and test
    numTrainingExamples, xTrain, yTrain = createTrainingMatrices(train, wordList, maxEncoderLength)
    numTestExamples, xTest, yTest = createTrainingMatrices(test, wordList, maxEncoderLength)
    np.save('Seq2SeqXTrain.npy', xTrain)
    np.save('Seq2SeqYTrain.npy', yTrain)
    np.save('Seq2SeqXTest.npy', xTest)
    np.save('Seq2SeqYTest.npy', yTest)
    print ('Finished creating training matrices')

print(len(xTest))

tf.reset_default_graph()

# Create the placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)

#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
# Architectural choice of of whether or not to include ^

decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM,
                                                            vocabSize, vocabSize, embeddingDim, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

lossWeights = [tf.ones_like(l, dtype=tf.float32) for l in decoderLabels]

loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

ValidLoss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
ValidOptimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
# If you're loading in a saved model, uncomment the following line and comment out line 202
#saver.restore(sess, tf.train.latest_checkpoint('models/'))
sess.run(tf.global_variables_initializer())

# Uploading results to Tensorboard
tf.summary.scalar('Training Loss', loss)
tf.summary.scalar('Validation Loss', ValidLoss)
tf.summary.scalar('difference', loss-ValidLoss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Some test strings that we'll use as input at intervals during training
encoderTestStrings = ["hello",
					"hi",
					"how r u",
					"wyd",
					"sup"
					]

zeroVector = np.zeros((1), dtype='int32')

for i in range(numIterations):

    encoderTrain, decoderTargetTrain, decoderInputTrain = getTrainingBatch(xTrain, yTrain, batchSize, maxEncoderLength, numTrainingExamples)
    feedDict = {encoderInputs[t]: encoderTrain[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: decoderTargetTrain[t] for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: decoderInputTrain[t] for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: False})

    curLoss, _, pred = sess.run([loss, optimizer, decoderPrediction], feed_dict=feedDict)

    ####DIVIDE BETWEEN TRAIN AND TEST########
    encoderTest, decoderTargetTest, decoderInputTest = getTrainingBatch(xTest, yTest, batchSize, maxEncoderLength, numTestExamples)
    feedDict = {encoderInputs[t]: encoderTest[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: decoderTargetTest[t] for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: decoderInputTest[t] for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    ValLoss, _, pred = sess.run([ValidLoss, ValidOptimizer, decoderPrediction], feed_dict=feedDict)

    if (i % 50 == 0):
        print('Current loss:', curLoss, ' Validation Loss:', ValLoss, 'at iteration', i)
        summary = sess.run(merged, feed_dict=feedDict)
        writer.add_summary(summary, i)
    if (i % 25 == 0 and i != 0):#only need this to TEST
        num = randint(0,len(encoderTestStrings) - 1)
        print(encoderTestStrings[num])
        inputVector = getTestInput(encoderTestStrings[num], wordList, maxEncoderLength)
        feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
        feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
        feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
        feedDict.update({feedPrevious: True})
        ids = (sess.run(decoderPrediction, feed_dict=feedDict))
        print(idsToSentence(ids, wordList))

    if (i % 10000 == 0 and i != 0):
        print("save")
        savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
