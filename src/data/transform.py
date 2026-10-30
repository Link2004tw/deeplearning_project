from torchvision import transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_test_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])