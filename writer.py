import time 
import random
import sys 

text        = open(sys.argv[1],'r')
skiplines   = 14
dropin      = "".join([text.readline() for _ in range(skiplines)])
write_file  = open("test.py",'a')
write_file.write(dropin)
write_file.close()
text = text.read()


while text:
    write_file  = open("test.py",'a')
    time.sleep(.004*random.randint(1,6))
    write_file.write(text[1])
    text = text[1:] 
    write_file.close()
