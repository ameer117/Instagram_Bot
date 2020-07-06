from selenium import webdriver
from time import sleep
import numpy as np
import pickle
import tensorflow as tf
import numpy as np
import random

class InstaBot:
    def __init__(self, username, pw, num):
        self.msgnum = num
        self.driver = webdriver.Chrome()
        self.username = username
        self.driver.get("https://instagram.com")
        sleep(2)
        self.driver.find_element_by_xpath("//input[@name=\"username\"]")\
            .send_keys(username)
        self.driver.find_element_by_xpath("//input[@name=\"password\"]")\
            .send_keys(pw)
        self.driver.find_element_by_xpath('//button[@type="submit"]')\
            .click()
        sleep(4)
        self.driver.find_element_by_xpath("//button[contains(text(), 'Not Now')]")\
            .click()
        sleep(2)

    def get_unfollowers(self):
        self.driver.find_element_by_xpath("//a[contains(@href,'/{}')]".format(self.username))\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//a[contains(@href,'/following')]")\
            .click()
        following = self._get_names()
        self.driver.find_element_by_xpath("//a[contains(@href,'/followers')]")\
            .click()
        followers = self._get_names()
        not_following_back = [user for user in following if user not in followers]
        print("LIST OF NON FOLLOWERS: " + str(not_following_back))

    def _get_names(self):
        sleep(2)
        #sugs = self.driver.find_element_by_xpath('//h4[contains(text(), Suggestions)]')
        #self.driver.execute_script('arguments[0].scrollIntoView()', sugs)
        sleep(2)
        scroll_box = self.driver.find_element_by_xpath("/html/body/div[4]/div/div[2]/ul")
        last_ht, ht = 0, 1
        # while last_ht != ht:
        #     last_ht = ht
        #     sleep(1)
        #     ht = self.driver.execute_script("""
        #         arguments[0].scrollTo(0, arguments[0].scrollHeight); 
        #         return arguments[0].scrollHeight;
        #         """, scroll_box)
        fBody = self.driver.find_element_by_xpath("//div[@class='isgrP']")
        #scroll = 0
        for i in range(1,6):
            self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight;', fBody)
            sleep(0.1)
        links = scroll_box.find_elements_by_tag_name('a')
        names = [name.text for name in links if name.text != '']
        # close button
        self.driver.find_element_by_xpath("/html/body/div[4]/div/div[1]/div/div[2]/button")\
            .click()
        #print(names)
        return names
    def dmFromHome(self, name):
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[3]/div/div[2]/a')\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//a[@class='-qQT3 rOtsg']")\
            .click()
        #text = self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/span').text
    def getText(self, num):
        input = ""
        try:
            input = self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[{}]/div[2]/div/div/div/div/div/div/div/div/span'.format(str(num))).text
        except:
            print("no message")
            input = ""
        return input
    def sendMessage(self, msg):
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div/div/div[2]/textarea').send_keys(msg)
        sleep(1)
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div/div/div[3]/button').click()
        self.msgnum += 2
        print("MSGNUM = ", self.msgnum)
        
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
            myStr = myStr + (wList[num[0]]).decode('utf-8') + " "
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


with open("wordList.txt", "r+b") as fp:
    wordList = pickle.load(fp, encoding="bytes")
wordList.append('<pad>')
wordList.append('<EOS>')
print(wordList)
vocabSize = len(wordList)

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
saver.restore(sess, "models/pretrained_seq2seq.ckpt-149000")
zeroVector = np.zeros((1), dtype='int32')
user = '' #IG USER
pw = '' #IG PASSWORD
name = '' #USERNAME OF WHO TO MESSAGE
my_bot = InstaBot(user, pw, 45) 
sleep(2)
my_bot.dmFromHome(name)
#inp = my_bot.getText(my_bot.msgnum)

while (True):
    inp = my_bot.getText(my_bot.msgnum)
    while (inp == ""):
        sleep(1)
        inp = my_bot.getText(my_bot.msgnum)
    inputString = inp
    inputVector = getTestInput(inputString, wordList, maxEncoderLength)
    feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    ids = (sess.run(decoderPrediction, feed_dict=feedDict))
    my_bot.sendMessage(str(idsToSentence(ids, wordList)))
    sleep(2)



#print(my_bot.getText(19))




#//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/span
#//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[19]/div[2]/div/div/div/div/div/div/div/div/span
