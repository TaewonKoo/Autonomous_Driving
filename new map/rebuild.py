"""Rebuild this map using its frozen source; write inside this bundle by default."""
import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def main():
    if not os.environ.get('SUMO_HOME'):
        try:
            import sumo
            os.environ['SUMO_HOME'] = str(Path(next(iter(sumo.__path__))))
        except ImportError:
            pass
    from maps.seongsan import config
    from maps.seongsan.builder import build

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated')
    parser.add_argument('--laps', type=int, default=1)
    parser.add_argument('--route', choices=list(config.EGO_ROUTES), default=config.DEFAULT_EGO_ROUTE)
    parser.add_argument('--background-only', action='store_true')
    parser.add_argument('--no-traffic', action='store_true')
    args = parser.parse_args()
    if args.laps < 1:
        parser.error('--laps must be positive')
    path = build(args.output, laps=args.laps, traffic=not args.no_traffic,
                 include_ego=not args.background_only, route_name=args.route)
    print(f'Map: {path}')
    print(f'Preview: {path.parent.parent / "preview.svg"}')


if __name__ == '__main__':
    main()
