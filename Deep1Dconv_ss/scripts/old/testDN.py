import sys
#sys.path.append("/opt/deepnetwork-toolkit/")
#sys.path.append("/opt/cudamat-rbm/")
sys.path.append("/home/mattcspencer/DNSA/lib/")

from DN import DN, DN_load, DN_save 
import numpy as np

if len(sys.argv) < 5:
   sys.stderr.write('Usage: sys.argv[0] test_data_filename model_filename pred probs <targsize>')
   print "\n"
   sys.exit(1)

dn = DN_load(sys.argv[2])


test_data = np.load(sys.argv[1]) 
test_l1 = test_data[:,:]

probs = dn.calc_output_legacy(test_l1, 1000)

targsize = 3
if len(sys.argv) >= 6:
   targsize = sys.argv[5]

max_vals = np.reshape(np.repeat(probs.max(axis=1), targsize), (probs.shape[0], targsize))

print "".format(probs[0], max_vals[0], (probs[0] >= max_vals[0]))
preds = 1 * (probs > max_vals - .0001)

np.savetxt(sys.argv[3], preds, fmt="%d")
np.savetxt(sys.argv[4], probs, fmt="%.6f")


