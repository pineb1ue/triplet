import torchvision.transforms as T


def get_transforms(phase):
    if phase not in ("train", "test"):
        raise ValueError

    transforms = [T.Resize((224, 224)), T.ToTensor()]

    return T.Compose(transforms)
