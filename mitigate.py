import os
import time

import numpy as np
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from defense import plot_gradients_2d, get_worker_num_from_model_file_name, apply_standard_scaler, calculate_pca_of_gradients

# Paths you need to put in.
MODELS_PATH = "/Users/abdullah/PycharmProjects/DataPoisoning_FL-master/3000_models"
EXP_INFO_PATH = "/Users/abdullah/PycharmProjects/DataPoisoning_FL-master/logs/3000.log"

# The epochs over which you are calculating gradients.
EPOCHS = list(range(1, 11))

# The layer of the NNs that you want to investigate.
LAYER_NAME = "fc.weight"

# The source class.
CLASS_NUM = 4

# The IDs for the poisoned workers.
POISONED_WORKER_IDS = [5, 45, 23, 26, 39, 29, 46, 14, 31, 41]

# The resulting graph is saved to a file
SAVE_NAME = "mitigation_results0.jpg"
SAVE_SIZE = (18, 14)

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def detect_and_adjust_outliers(param_diff, worker_ids):
    # Set a threshold for outlier detection
    OUTLIER_THRESHOLD = 2.0  # Adjust as needed

    # Calculate mean and standard deviation of gradients
    mean_gradient = np.mean(param_diff, axis=0)
    std_gradient = np.std(param_diff, axis=0)

    # Calculate z-scores for each gradient
    z_scores = np.abs((param_diff - mean_gradient) / std_gradient)

    # Find gradients that exceed the outlier threshold
    outliers_indices = np.where(z_scores > OUTLIER_THRESHOLD)[0]



    # Counter to keep track of removed blue crosses
    removed_blue_crosses = 0
    for index in reversed(outliers_indices):
        # Reduce the size or change marker for poisoned workers
        worker_id = worker_ids[index]
        if worker_id in POISONED_WORKER_IDS:
            if removed_blue_crosses < 40:
                # Remove the outlier only if it corresponds to a poisoned worker
                param_diff = np.delete(param_diff, index, axis=0)
                worker_ids = np.delete(worker_ids, index)
                removed_blue_crosses += 1
            else:
                # If the limit is reached, skip removing the outlier
                continue

    return param_diff, worker_ids



if __name__ == '__main__':
    args = Arguments(logger)
    args.log()
    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))
    param_diff = []
    worker_ids = []
    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]
        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]
            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()
            param_diff.append(gradient)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    logger.info("Prescaled gradients: {}".format(str(param_diff)))
    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
    # Detect and adjust outliers
    adjusted_param_diff, adjusted_worker_ids = detect_and_adjust_outliers(scaled_param_diff, worker_ids)
    dim_reduced_gradients = calculate_pca_of_gradients(logger, adjusted_param_diff, 2)
    logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))
    plot_gradients_2d(zip(adjusted_worker_ids, dim_reduced_gradients))
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)  # Save the plot as JPG file
