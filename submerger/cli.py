import argparse
from . import sync, align


def main(argv=None):
    parser = argparse.ArgumentParser(prog='submerger', description='Subtitle utilities')
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    p_pair = subparsers.add_parser('pair', help='Pair subtitle files')
    p_pair.add_argument('tl_file')
    p_pair.add_argument('sl_file')
    p_pair.add_argument('--tl-code', default='tl')
    p_pair.add_argument('--sl-code', default='sl')
    p_pair.add_argument('--out', default='synced_subs')

    p_align = subparsers.add_parser('align', help='LLM-assisted alignment')
    p_align.add_argument('tl_file')
    p_align.add_argument('sl_file')
    p_align.add_argument('--tl-code', default='tl')
    p_align.add_argument('--sl-code', default='sl')
    p_align.add_argument('--out', default='final_synced')

    args = parser.parse_args(argv)

    if args.cmd == 'pair':
        return sync.main(
            args.tl_file,
            args.sl_file,
            tl_code=args.tl_code,
            sl_code=args.sl_code,
            out=args.out,
        )
    else:
        return align.main(
            args.tl_file,
            args.sl_file,
            tl_code=args.tl_code,
            sl_code=args.sl_code,
            out=args.out,
        )


if __name__ == '__main__':
    main()
