import random 


fname           = "ytdump.txt"

contents        = open(fname,'r',encoding='utf_8').readlines()
redo_file       = open(fname,'w',encoding='utf_8')

for line in contents:
    if 'watch?v=' in line:
        redo_file.write(line)
redo_file.close()