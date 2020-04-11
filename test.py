import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bs", default=None, required=True, help="batch size",
)
parser.add_argument(
    "--tiny", default=False, required=False, help="run on a sample data",
)
parser.add_argument(
    "--adaptive",
    default=False,
    required=True,
    action="store_false",
    help="Use Adaptive Attention Span",
)
args = parser.parse_args()
print(args)
