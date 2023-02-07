import os
import sys
from pathlib import Path



def prep(df_dir='flir.txt'):

    data_dic = {'bicycle':0,'car':1,'person':2}

    os.chdir(Path(df_dir))

    if str(df_dir).split('/')[-1]=='flir':
        df=open('flir2.txt','w')
    else:
        df=open('mscoco2.txt','w')

    for i in data_dic.items():
        directory = "./"+str(i[0])
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                df.write(f+" "+str(i[1]))
                df.write('\n')

    df.close()
df = sys.argv[1]
prep(df)