
import sqlite3
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# find your chat.db and establish a connection
conn = sqlite3.connect('/Users/ameerhussain/Library/Messages/chat.db')
cur = conn.cursor()

# query the database to get all the table names
cur.execute(" select name from sqlite_master where type = 'table' ")

for name in cur.fetchall():
    print(name)
('_SqliteDatabaseProperties',)
('deleted_messages',)
('sqlite_sequence',)
('chat_handle_join',)
('chat_message_join',)
('message_attachment_join',)
('handle',)
('message',)
('chat',)
('attachment',)
('sync_deleted_messages',)
('message_processing_task',)
('sync_deleted_chats',)
('sync_deleted_attachments',)
('kvtable',)
('sqlite_stat1',)

# create pandas dataframe with all the tables you care about.

## Mac OSX versions below High Sierra
#messages = pd.read_sql_query('''select *, datetime(date + strftime("%s", "2001-01-01") ,"unixepoch","localtime")  as date_utc from message''', conn) 

## High Sierra and above
messages = pd.read_sql_query('''select *, datetime(date/1000000000 + strftime("%s", "2001-01-01") ,"unixepoch","localtime")  as date_utc from message''', conn) 

handles = pd.read_sql_query("select * from handle WHERE ROWID = '2'", conn)
chat_message_joins = pd.read_sql_query("select * from chat_message_join", conn)
# these fields are only for ease of datetime analysis (e.g., number of messages per month or year)
messages['message_date'] = messages['date']
messages['timestamp'] = messages['date_utc'].apply(lambda x: pd.Timestamp(x))
messages['date'] = messages['timestamp'].apply(lambda x: x.date())
messages['month'] = messages['timestamp'].apply(lambda x: int(x.month))
messages['year'] = messages['timestamp'].apply(lambda x: int(x.year))


# rename the ROWID into message_id, because that's what it is
messages.rename(columns={'ROWID' : 'message_id'}, inplace = True)

# rename appropriately the handle and apple_id/phone_number as well
handles.rename(columns={'id' : 'phone_number', 'ROWID': 'handle_id'}, inplace = True)
# merge the messages with the handles
merge_level_1 = pd.merge(messages[['text', 'handle_id', 'date','message_date' ,'timestamp', 'month','year','is_sent', 'message_id']],  handles[['handle_id', 'phone_number']], on ='handle_id', how='inner')

# and then that table with the chats
df_messages = pd.merge(merge_level_1, chat_message_joins[['chat_id', 'message_id']], on = 'message_id', how='left')


print(len(df_messages))
#print(df_messages.head())

# save the combined table for ease of read for future analysis!
df_messages.to_csv('./texts.csv', index = False, encoding='utf-8')
df_messages['date'].min(), df_messages['date'].max()
# number of messages per day
plt.plot(df_messages.groupby('date').size())
plt.xticks(rotation='45')
# how many messages you have sent versus received
df_messages.groupby('is_sent').size()
#number of messages per month and year
df_messages.groupby('month').size()
df_messages.groupby('year').size()

# close connections
cur.close()
conn.close()