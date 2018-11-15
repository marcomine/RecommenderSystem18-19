# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:06:43 2018

@author: marco
"""
import numpy as np
import scipy.sparse as sps
import csv
import pandas as pd
from recommend import ItemCBFKNNRecommender





tracks = pd.read_csv("C:/Users/marco/.spyder-py3/tracks.csv")
train = pd.read_csv("C:/Users/marco/.spyder-py3/train.csv")
targetPlaylist = pd.read_csv("C:/Users/marco/.spyder-py3/target_playlists.csv")

targetPlaylistCol = targetPlaylist.playlist_id.tolist()

trackCol = tracks.track_id.tolist()

playlistCol = train.playlist_id.tolist()

tracklistCol = train.track_id.tolist()


albumIdCol = tracks.album_id.tolist()                                                           #column ALBUM_ID from tracks.csv
#albumIdCol.sort()                                   #column ALBUM_ID ordered
albumIdCol_unique = list(set(albumIdCol))           #column ALBUM_ID ordered, without replicated elements


artistIdCol = tracks.artist_id.tolist()              #column ARTIST_ID from tracks.csv
#artistIdCol.sort()                                   #column ARTIST_ID ordered
artistIdCol_unique = list(set(artistIdCol))          #column ARTIST_ID ordered, without replicated elements

 


durSecCol = tracks.duration_sec.tolist()              #column DURATION_SEC from tracks.csv
#durSecCol.sort()                                      #column DURATION_SEC ordered
durSecCol_unique = list(set(durSecCol))               #column DURATION_SEC ordered, without replicated elements

numTrack = len(trackCol)
numPlayList = len(playlistCol)

albumIdArtistIdCol = albumIdCol + artistIdCol
albumIdArtistIdCol

mat = sps.coo_matrix(((np.ones(numPlayList, dtype=int)), (playlistCol, tracklistCol)))
mat = mat.tocsr()


matTrack_Album = sps.coo_matrix(((np.ones(numTrack, dtype=int)), (trackCol , albumIdCol)))       #sparse matrix ROW: track_id COLUMN: album_id
matTrack_Album = matTrack_Album.tocsr()


matTrack_Artist = sps.coo_matrix(((np.ones(numTrack, dtype=int)), (trackCol, artistIdCol)))       #sparse matrix ROW: track_id COLUMN: artist_id
matTrack_Artist = matTrack_Artist.tocsr()

#matTrack_Dur = sps.coo_matrix(((np.ones(numTrack, dtype=int)), (trackCol, durSecCol)))       #sparse matrix ROW: track_id COLUMN: duration_sec
#matTrack_Dur = matTrack_Dur.tocsr()

trainjoined = train.set_index("track_id").join(tracks, on="track_id")
trainjoined

#matUser_Artist = sps.coo_matrix(((np.ones(numPlayList, dtype=int)), (playlistCol, trainjoined.artist_id)))       
#matUser_Artist = matUser_Artist.tocsr()

matUser_Album = sps.coo_matrix(((np.ones(numPlayList, dtype=int)), (playlistCol, trainjoined.album_id)))       
matUser_Album = matUser_Album.tocsr()

#NUOVA AGGIUNT
#mat_Train, mat_Test = train_test_holdout(mat, train_perc = 0.8)








recommender = ItemCBFKNNRecommender(mat, matTrack_Artist, matTrack_Album, matUser_Album)    
recommender.fit(shrink=22.0, topK=160)
with open('HybridFiltUserAlb.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['playlist_id', 'track_ids'])
    for playlist in targetPlaylistCol[0:]:
        thewriter.writerow([playlist, ' '.join(map(str, recommender.recommend(playlist, at=10))) ])