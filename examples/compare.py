"""Choose an implementation and input; all measurement machinery is in the tool."""

from weft_bench import Inputs, WeftKernel, compare
from weft_bench.loading import load_config, select_target
from weft_bench.results import save, table


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--weft",
        help="file.py:symbol or artifact directory; omit for the preset kernel",
    )
    parser.add_argument("--output", default="runs/example")
    args = parser.parse_args()
    target = select_target(load_config(args.config), args.target)
    rows = compare(
        "rms_norm",
        target=target,
        candidate=WeftKernel.parse(args.weft) if args.weft else None,
        inputs=Inputs(preset="decode"),
        output=args.output + "/artifacts",
    )
    print(table(rows))
    save(rows, args.output)


if __name__ == "__main__":
    main()
