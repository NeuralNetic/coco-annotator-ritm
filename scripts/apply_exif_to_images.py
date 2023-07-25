from argparse import ArgumentParser, Namespace
import numpy as np
import os
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Apply EXIF transforms to images in a database folder'
    )
    parser.add_argument(
        '-i', '--input', required=True, type=str,
        help='Path to folder with images'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = args.input

    for file_name in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            pil_img = Image.open(file_path)
        except Exception as e:
            print('Delete image {}, because {}'.format(file_path, e))
            os.remove(file_path)

        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = Image.fromarray(np.array(pil_img.convert('RGB')))
        pil_img.save(file_path)


if __name__ == '__main__':
    main()
