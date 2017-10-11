import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer


# Load data
data_loader = CreateDataLoader(opt)
dataset_paired, paired_dataset_size = data_loader.load_data_pair()
dataset_unpaired, unpaired_dataset_size = data_loader.load_data_unpair()

# Create Model
model = create_model(opt)
visualizer = Visualizer(opt)

# Start Training
print('Start training')

#################################################
# Step1: Autoencoder
#################################################
print('step 1')
pre_epoch_AE = 5 # number of iteration for autoencoder pre-training
total_steps = 0
for epoch in range(1, pre_epoch_AE+1):
    for i,(images_a, images_b) in enumerate(dataset_paired):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - unpaired_dataset_size * (epoch - 1)
        model.set_input(images_a, images_b)
        model.optimize_parameters_pretrain_AE()
    print('pretrain Autoencoder model (epoch %d, total_steps %d)' %
          (epoch, pre_epoch_AE))


#################################################
# Step2: CycleGAN
#################################################
print('step 2')
pre_epoch_cycle = 5 # number of iteration for CycleGAN training
total_steps = 0
for epoch in range(1, pre_epoch_cycle+1):
    epoch_start_time = time.time()
    for i,(images_a, images_b) in enumerate(dataset_unpaired):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - unpaired_dataset_size * (epoch - 1)
        model.set_input(images_a, images_b)
        model.optimize_parameters_pretrain_cycleGAN()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors_cycle()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/unpaired_dataset_size, opt, errors)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, pre_epoch_cycle, time.time() - epoch_start_time))

    if epoch > pre_epoch_cycle/2:
        model.update_learning_rate()


#################################################
# Step3:  VIGAN
#################################################
print('step 3')
total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
	# You can use paired and unpaired data to train. Here we only use paired samples to train.
    for i,(images_a, images_b) in enumerate(dataset_paired):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - paired_dataset_size * (epoch - 1)
        model.set_input(images_a, images_b)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/paired_dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()