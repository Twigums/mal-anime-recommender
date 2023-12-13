from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# loads the data of the given path as an ImageFolder using pytorch's implementation of class values as a folder containing image data points
def load_data(path_to_data, split_percent, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((320, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.15, ), (0.3, )),
        ])

    dataset = datasets.ImageFolder(root = path_to_data, transform = transform)

    train_size = int(split_percent * len(dataset))
    test_size = int(len(dataset) - train_size)

    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
            dataset = train_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True,
            )
    test_loader = DataLoader(
            dataset = test_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True,
            )

    return train_data, test_data, train_loader, test_loader

def load_recommend_data(path_to_recommend_data, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((320, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.15, ), (0.3, )),
        ])

    dataset = datasets.ImageFolder(root = path_to_recommend_data, transform = transform)
    idxs = [key for key in dataset.class_to_idx]

    test_loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True,
            )

    return idxs, test_loader
