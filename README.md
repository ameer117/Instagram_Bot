# Instagram_Bot
An instagram bot with the ability to reply to DMs using machine learning. Used 35,000+ of my personal text messages to train the bot using the TensorFlow library to train a seq2seq model. This bot replies to messages based on what it's learning from how I reply to my texts. Bot also has the ability to find people who don't follow you back.

#####
Instructions:

Go to IMESSAGE_COLLECTOR FOLDER

Use messagescrape to gather imessages from an apple computer. clean_messages.ipynb is used to clean the messages

Return to root folder

Run seq2seqpy3 to train the model

Input your username and password and the person you want to reply to into the appropriate locations in main.py

Run Main.py
