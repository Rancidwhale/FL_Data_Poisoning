import os
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
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import server as s

# Paths you need to put in.
MODELS_PATH = "/Users/abdullah/PycharmProjects/DataPoisoning_FL-master/3002_models"
EXP_INFO_PATH = "/Users/abdullah/PycharmProjects/DataPoisoning_FL-master/logs/3002.log"

# The epochs over which you are calculating gradients.
EPOCHS = list(range(1,11))

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
LAYER_NAME = "fc.weight"

#df = pd.read_csv("data.csv")
# The source class.
CLASS_NUM = 4

# The IDs for the poisoned workers. This needs to be manually filled out.
# You can find this information at the beginning of an experiment's log file.
POISONED_WORKER_IDS = [14, 1, 45, 26, 24, 31, 23, 34, 36, 15, 13,18,5, 11 ]


# The resulting graph is saved to a file
SAVE_NAME = "defense_results_300212.jpg"
SAVE_SIZE = (18, 14)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


def plot_gradients_2d(gradients):
    fig = plt.figure()
    flag1=1
    flag2=1
    flag3=1
    flag4=1

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            if worker_id == 11:
                if flag1:
                    plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
                    flag1=0
                else:
                    continue
            if worker_id == 5:
                if flag2:
                    plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
                    flag2=0
                else:
                    continue

            if worker_id == 13:
                if flag3:
                    plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
                    flag3=0
                else:
                    continue
            if worker_id == 18:
                if flag4:
                    plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
                    flag4=0
                else:
                    continue
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
        else:
            plt.scatter(gradient[0], gradient[1], color="orange", s=180)

    fig.set_size_inches(SAVE_SIZE, forward=False)

    # Increase x and y axis coordinates
    plt.xlim(-30, 70)  # Set the x-axis limits
    plt.ylim(-30, 70)  # Set the y-axis limits

    plt.grid(False)
    plt.margins(0,0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)


def detect_poisoned_workers(param_diff, worker_ids):
    # Calculate average gradient update
    average_update = np.mean(param_diff, axis=0)

    # Calculate L2-norm of differences from average
    gradient_norms = [np.linalg.norm(diff - average_update) for diff in param_diff]

    # Set a threshold for anomaly detection
    anomaly_threshold = 3.5 * np.std(gradient_norms)  # Example: 2 standard deviations

    # Identify poisoned workers
    poisoned_worker_ids = [worker_ids[i] for i, norm in enumerate(gradient_norms) if norm > anomaly_threshold]
    print(poisoned_worker_ids)
    #exit(0)
    return poisoned_worker_ids


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
            worker_ids.append(worker_id)


    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))

    logger.info("Prescaled gradients: {}".format(str(param_diff)))
    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
    dim_reduced_gradients = calculate_pca_of_gradients(logger, scaled_param_diff, 2)
    logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))

    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))
    #POISONED_WORKER_IDS=detect_poisoned_workers(param_diff,worker_ids)
    #print(POISONED_WORKER_IDS)
    #exit(0)
    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients))
