import sys
sys.path.insert(0,"/home/mattcspencer/DNSA/lib/")

from DN import DN, DN_load, DN_save 
import numpy as np

if len(sys.argv) < 3:
   sys.stderr.write('Usage: sys.argv[0] train_data_filename model_filename <architecture>')
   print "\n"
   sys.exit(1)

if len(sys.argv) >= 4:
   archstring = sys.argv[3]
else:
   archstring = "500,500,500,200,3"

intArch = []
Arch = archstring.split(',')
for a in Arch:
   intArch.append(int(a))

alen = len(intArch) - 1;

print "\nTraining File: {0}; Model name: {1}".format(sys.argv[1], sys.argv[2])
print "\nUsing Architecture: {0}; Target size {1}\n".format(archstring, intArch[alen])

data = np.load(sys.argv[1]) 
data_l1 = data[:,intArch[alen]:]
targets = data[:,0:intArch[alen]]

print data_l1.shape

dn = DN(intArch)

dn.set_param('num_epochs', 30)
dn.set_param('batch_size', 1000)

dn.train_cuda_legacy(data_l1, targets)

DN_save(dn, sys.argv[2])


