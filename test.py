import time
import os
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

#load data
data_loader = CreateDataLoader(opt)
dataset_paired, paired_dataset_size = data_loader.load_data_test()

model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
rmse_x = []
rmse_y = []
for i, (images_a, images_b) in enumerate(dataset_paired):

    model.set_input(images_a, images_b)
    model.test()
    visuals = model.get_current_visuals()
    img_path = 'image'+ str(i)
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

print(np.mean(rmse_y))
print(np.mean(rmse_x))
print((np.mean(rmse_x)+np.mean(rmse_y))/2)
webpage.save()
