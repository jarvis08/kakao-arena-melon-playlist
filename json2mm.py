import json
import sys 

import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix


def parse_songs():
    with open('./res/train.json', 'r') as f:
        print('>> Opened train.json')
        train_data = json.load(f)
        with open('./res/songs.json', 'w') as s:
            print('>> Make songs.json with {} play lists..'.format(len(train_data)))
            songs = []
            for plist in train_data:
                songs.append(plist["songs"])
                print(plist["songs"])
            json.dump(songs, s)

def parse_id_songs():
    with open('./res/train.json', 'r') as f:
        train_data = json.load(f)
        with open('./res/id_songs.json', 'w') as s:
            id_songs = {}
            for plist in train_data:
                id_songs[plist["id"]] = plist["songs"]
            json.dump(id_songs, s)


#def plist2mm():
#    print('>> Start making mm file..')
#    with open('./res/train.mm', 'wb') as f:
#        with open('./res/songs.json', 'r') as s:
#            df = pd.DataFrame({"A": [1,2], "B": [3,0]})
#            scipy.io.mmwrite("mmout", df)
#            songs = json.load(s)
#            #np_array = np.asarray(songs)
#            #print(type(np_array))
#            #scipy.io.mmwrite(f, np_array)
#            print(songs[0])
#            scipy.io.mmwrite(f, songs)
#    print('>> Done.')


def plist2mm():
    print('>> Start making mm file..')
    with open('./res/id_songs.json', 'r') as s:
        id_songs = json.load(s)
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in id_songs.items() ]))
        df.fillna(0)
        #df = pd.DataFrame(id_songs)
        scipy.io.mmwrite("res/main", scipy.sparse.csr_matrix(df))
        #scipy.io.mmwrite("train", df)
    print('>> Done.')

def check_data():
    import collections
    print('>> Check json file')
    with open('./res/id_songs.json', 'r') as f:
        id_songs = json.load(f)
        ids = id_songs.keys()
        songs = id_songs.values()
        songs = [id for ids in songs for id in ids]
        songs = collections.Counter(songs)
        print('len_ids = ', len(ids))
        print('len_songs = ', len(songs))  


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        parse_songs()
    if sys.argv[1] == 'dict':
        parse_id_songs()
    if sys.argv[1] == 'convert':
        plist2mm()
    if sys.argv[1] == 'check':
        check_data()
