"""
The point of this file is to serialize the data into a dilled fill.
such that it can be used in the future without having to re-run the
pre-processing steps.
-------------------------------------------------------------------
1. Make the classes across the training set have uniform amount of
   samples for each class.
2. Make the test set also uniform that way it is easier to see if the
   best model is good for all classes.
3. Use the rest of the data for the validation set.
4. Serialize all the data into dilled files.
   (test to make sure the RAM is enough)
"""

# Import file manipulation libraries and path handling libraries
import os
import dill
import pathlib



