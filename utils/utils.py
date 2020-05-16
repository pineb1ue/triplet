import cv2
import pandas as pd
from PIL import Image


def add_id(row, num):
    """Add num to row["id]"""
    row["id"] += num
    return row


def add_path_column(df, image_paths):
    def add_path(row, paths):
        path = paths[int(row["frame"]) - 1]
        row["image_path"] = path
        return row

    return df.apply(add_path, axis=1, args=(image_paths,))


def get_image(row, crop_bbox=True, pil=False):
    """Return RGB array/PIL image."""
    image = cv2.imread(row["image_path"])

    if crop_bbox:
        l, t, w, h = list(map(int, row.iloc[2:6]))
        image = image[t : t + h, l : l + w, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image if not pil else Image.fromarray(image)


def load_mot(path, additional_columns=None):
    if additional_columns is None:
        additional_columns = []

    print("Loading {}.".format(path))

    df = pd.read_csv(
        path,
        names=(
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
            *additional_columns,
        ),
    )

    print("Loaded. Dataframe has {} data.".format(df.shape[0]))
    return df
