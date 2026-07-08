from .cic_iot_loader import load_cic_dataset
from .nba_iot_loader import load_nbaiot_dataset
from .common import get_dataloaders


def load_dataset(args):

    dataset = args.dataset.lower()

    if dataset == "cic":

        return load_cic_dataset(
            data_dir=args.data_path,
            num_parts=args.num_parts
        )

    elif dataset == "nbaiot":

        return load_nbaiot_dataset(
            data_dir=args.data_path,
            num_parts=args.num_parts
        )

    else:

        raise ValueError(
            f"Unsupported dataset : {dataset}"
        )