from .cic_iot_loader import (
    load_cic_dataset,
    get_dataloaders,
)

from .nba_iot_loader import load_nbaiot_dataset


def load_dataset(args):

    if args.dataset.lower() == "cic":

        return load_cic_dataset(
            args.data_path,
            num_parts=args.num_parts
        )

    elif args.dataset.lower() == "nbaiot":

        return load_nbaiot_dataset(
            args.data_path
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {args.dataset}"
        )