from torch.utils.data import TensorDataset, DataLoader, Subset


def get_turpy_dataloaders(dataset):

    base_dataset = TensorDataset(
        dataset["X"],
        dataset["Y"],
    )

    train_data = Subset(
        base_dataset,
        dataset["splits"]["train_idx"],
    )

    val_data = Subset(
        base_dataset,
        dataset["splits"]["val_idx"],
    )

    test_data = Subset(
        base_dataset,
        dataset["splits"]["test_idx"],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader    

