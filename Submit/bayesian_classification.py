
# coding: utf-8

# In[235]:


from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle 


# In[2]:


gnb = GaussianNB()


# In[4]:


test1 = pd.read_csv('test.csv')
test2 = pd.read_csv('test2.csv')
test3 = pd.read_csv('test3.csv')
train = pd.read_csv('train.csv')


# In[184]:


#instantiate the classifier
used_features=["danceability",
               "energy",
               "key",
               "loudness",
               "mode",
               "speechiness",
               "acousticness",
               "instrumentalness",
               "liveness",
               "valence",
               "tempo",
               "time_signature"
              ]


# In[193]:


features=["danceability",
               "energy",
               "key",
               "loudness",
               "mode",
               "speechiness",
               "acousticness",
               "instrumentalness",
               "liveness",
               "valence",
               "tempo",
               "time_signature",
               "top"]


# In[248]:


non_top_song=pd.read_csv('non_top_460.csv', header=0)
non_top_song["top"] = 0
non_top = non_top_song[features]

top_song = pd.read_csv('top_140.csv', header=0)
top_song['top'] = 1
top = top_song[features]

concated_songs = pd.concat([top,non_top])

songs_train, songs_test = train_test_split(concated_songs, test_size = 0.20, random_state = 10)

gnb.fit(songs_train[used_features].values,
        songs_train["top"].values)

prediction = gnb.predict(songs_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
        .format(
            songs_test.shape[0],
            (songs_test["top"] != prediction).sum(),
            100*(1-(songs_test["top"] != prediction).sum()/songs_test.shape[0])
))

prediction


# In[253]:


non_top_song=pd.read_csv('non_top_460.csv', header=0)
non_top_song["top"] = 0
non_top = non_top_song[features]

top_song = pd.read_csv('top_140.csv', header=0)
top_song['top'] = 1
top = top_song[features]

concated_songs = pd.concat([top,non_top])
concated_songs = shuffle(concated_songs)

for i in range (0,5):

    songs_train1 = concated_songs[0:120*(i):]
    songs_train2 = concated_songs[120*(i+1):600:]
    songs_train = pd.concat([songs_train1,songs_train2])
    
    songs_test = concated_songs[120*(i):120*(i+1):]
    
    gnb.fit(songs_train[used_features].values,
            songs_train["top"].values)

    prediction = gnb.predict(songs_test[used_features])

    # Print results
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
          .format(
              songs_test.shape[0],
              (songs_test["top"] != prediction).sum(),
              100*(1-(songs_test["top"] != prediction).sum()/songs_test.shape[0])
    ))

    print(prediction)


# In[233]:


kf = KFold(n_splits=5, shuffle=True)
print(kf) 
songs_train = []
songs_test = []

for train_index, test_index in kf.split(concated_songs):
#     print("TRAIN:", train_index, "TEST:", test_index)
    songs_train.append(train_index)
    songs_test.append(test_index)
    print(songs_train)
    print(songs_test)


# In[207]:





# In[202]:


y_pred3 = gnb.predict(non_top_song[used_features])
y_pred3

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          non_top_song.shape[0],
          (non_top_song["top"] != y_pred3).sum(),
          100*(1-(non_top_song["top"] != y_pred3).sum()/non_top_song.shape[0])
))
y_pred3

