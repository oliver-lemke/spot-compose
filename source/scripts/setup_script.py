import os.path
import shutil
import zipfile

from utils.recursive_config import Config


def _wandb_login():
    os.system("wandb login")


def _download_Chinese_MNIST(config: Config):
    # MNIST dataset
    dataset_path = config.get_subpath("data")
    dataset_download_path = os.path.join(dataset_path, "data_template")
    file_name = "chinese-mnist.zip"
    zip_path = os.path.join(dataset_download_path, file_name)

    os.makedirs(dataset_download_path, exist_ok=True)

    os.system("kaggle datasets download -d gpreda/chinese-mnist")
    shutil.move(file_name, dataset_download_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_download_path)

    os.remove(zip_path)

    csv_path = os.path.join(dataset_download_path, "chinese_mnist.csv")
    with open(csv_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    os.remove(csv_path)

    header = lines[0]
    train_lines = lines[1:12_000]
    val_lines = lines[12_000:]

    train_csv_path = os.path.join(dataset_download_path, "train.csv")
    val_csv_path = os.path.join(dataset_download_path, "val.csv")
    with open(train_csv_path, "wt", encoding="UTF-8") as train, open(
        val_csv_path, "wt", encoding="UTF-8"
    ) as val:
        train.write(header)
        train.writelines(train_lines)
        val.write(header)
        val.writelines(val_lines)


def main():
    config = Config()
    _wandb_login()
    _download_Chinese_MNIST(config)


if __name__ == "__main__":
    main()
