from transformers import HfArgumentParser
from t5_pretrainer.arguments import EvalArguments
from t5_pretrainer.index_dpr import dense_indexing_dpr, dense_retrieve_dpr
from t5_pretrainer.utils.utils import set_seed
from pdb import set_trace as st
set_seed(42)

def main():
    print("starting...")
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.index_only:
        dense_indexing_dpr(args)
    elif args.retrieve_only:
        dense_retrieve_dpr(args)
    else:
        dense_indexing_dpr(args)
        dense_retrieve_dpr(args)


if __name__ == "__main__":
    main()
