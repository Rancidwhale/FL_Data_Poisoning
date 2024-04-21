import pickle

path = "data_loaders/fashion-mnist/train_data_loader.pickle"
with open(path, "rb") as f:
    x= pickle.load(f)

print(x)