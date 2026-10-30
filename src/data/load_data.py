from datasets import load_dataset

def load_data(cache_dir: str = "data/cub"):
    """
    Load the CUB dataset for training and testing.
    """
    train = load_dataset("Multimodal-Fatima/CUB_train", split="train", cache_dir=cache_dir)
    test = load_dataset("Multimodal-Fatima/CUB_test", split="test", cache_dir=cache_dir)
    return train, test


def get_class_names(dataset):
    return dataset.features["label"].names

if __name__ == "__main__":
    train_data, test_data = load_data()
    print(len(train_data), len(test_data))
    print(train_data.features)
    print("---------------------------------")
    # image, label, file_name, id, description 


    # Print the first example from the training data
    ex= train_data[0]
    print(ex["label"], ex["file_name"], ex["description"])
    print(ex["image"].mode, ex["image"].size, ex["image"].format)

