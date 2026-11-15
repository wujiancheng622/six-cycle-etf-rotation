"""Export frozen benchmark inputs from the original local research workspace."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.workspace.resolve() / 'guosheng_final_decision_20261114'))
    import study

    data = study.load()
    args.output.mkdir(parents=True, exist_ok=True)
    data.views.hist.to_parquet(args.output / 'returns.parquet')
    data.state.to_parquet(args.output / 'state.parquet')
    exclusions = data.views.hist.attrs.get('excluded_risk_dates', {})
    (args.output / 'risk_exclusions.json').write_text(json.dumps(exclusions, default=list), encoding='utf-8')


if __name__ == '__main__':
    main()
