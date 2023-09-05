import os

file_all = ['hike','lucia','stunt_1','hockey_1']
#file_all = [file for file in sorted(os.listdir('../ytvos/train/JPEGImages'))]

with open('val_davis2017_test.txt','a',encoding='utf-8') as f:
    for ss in file_all:
        print(ss)
        files = [file for file in sorted(os.listdir('../davis2017/JPEGImages/480p/'+ss))]
        for jj in files:
            f.write('1 '+'davis2017/JPEGImages/480p/'+ss+'/'+jj+' davis2017/Annotations/480p/'+ss+'/'+jj[:-3]+'png\n')
    
