#!/usr/bin/env python
# coding: utf-8
"""
Local application for Angelina Braille Reader inference
"""
import argparse
import os, time
from pathlib import Path

import local_config
import model.infer_retinanet as infer_retinanet

model_weights = 'model.t7'

recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
    model_weights_fn=os.path.join(local_config.data_path, 'weights', model_weights),
    create_script=None)
    
def cropFrom(img, results_dir):
    recognizer.run_and_save(img, results_dir, target_stem=None,
                                               lang="EN", extra_info=None,
                                               draw_refined=recognizer.DRAW_NONE,
                                               remove_labeled_from_filename=False,
                                               find_orientation=True,
                                               align_results=True,
                                               process_2_sides=False,
                                               repeat_on_aligned=False,
                                               save_development_info=False)
                                               
