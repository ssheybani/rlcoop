# all package-wide definitions. 
# This file is run every time a module in its subdirectories or sub-subdirectories is imported.
# Leave it empty, if the package’s modules and sub-packages do not need to share any code.


"""
Dependencies:
	util is used by: env_track_dyadic, torch_trainer .
	script files use all modules.
"""

from collections import namedtuple

Hyperparams = namedtuple('Hyperparams',
                        ('batch_size', 'lr', 'buffer_max_size', 'target_int', 'gamma'))

