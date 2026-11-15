"""Compare the five replay tables against the original frozen account."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actual', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    for field in ('ledger', 'weights', 'pnl', 'trades', 'signals'):
        actual = pd.read_parquet(args.actual / f'{field}.parquet')
        expected = pd.read_parquet(args.reference / f'{field}.parquet')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        print(f'{field}: {len(actual)} rows, exact match')


if __name__ == '__main__':
    main()
