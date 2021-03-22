# -*- coding: utf-8 -*-

import sys, os, inspect
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import sample


SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
PARENT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))

sys.path.insert(0, PARENT_PATH)

import rlcoop

DATA_PATH = os.path.join(PARENT_PATH,'data/')
CONFIG_PATH = os.path.join(PARENT_PATH,'rlcoop','configs/')
