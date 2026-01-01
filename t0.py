a = open("C:/.txt","r").read()#
b = a.split("   ncalls  tottime  percall  cumtime  percall filename:lineno(function)")[1].split("\n")[:-2]
# = a.split('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')[1]
#b = a[a.index('"""BEGIN')+1:]

elp = [] 
for item in b[1:]:
  c = item.split(" ")
  if len(c) < 5:
    break
  c = [l for l in c if not l == ""]
  c[1] = float(c[1])
  if c[1] > .5:
    elp.append((c[1],"".join(c[5:])))
import pprint 
pprint.pp(elp)
