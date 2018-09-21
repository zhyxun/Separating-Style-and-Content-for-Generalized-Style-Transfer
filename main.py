from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json

from train import *
from test import *
import random
######################################
# this is the version with discriminator and based on the stylebank
# this version does not resize the images and all images used are original size
############################################################

# if tf.gfile.Exists(a.output_dir):
#     tf.gfile.DeleteRecursively(a.output_dir)
#     tf.gfile.MakeDirs(a.output_dir)

def main():

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "train":
        train()
    elif a.mode == 'test':
        test()

main()



