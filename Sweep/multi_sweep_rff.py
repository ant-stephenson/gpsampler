import argparse
import sweep_rff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_idx", default=0, type=int)
    parser.add_argument("--job_id", default=0, type=int)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--NO_TRIALS", default=1, type=int)
    parser.add_argument("--significance_threshold", default=0.1, type=float)
    parser.add_argument("--ncpus", default=1, type=int)
    parser.add_argument("--method", default="rff", type=str)

    args = parser.parse_args()

    param_set = {
        k: v
        for(k, v) in zip(
            sweep_rff.default_param_set.keys(),
            sweep_rff.param_sets[args.param_idx])}

    sweep_rff.run_sweep(
        **param_set,
        job_id=args.job_id,
        verbose=args.verbose,
        param_index=args.param_idx,
        NO_TRIALS=args.NO_TRIALS,
        significance_threshold=args.significance_threshold,
        ncpus=args.ncpus,
        method=args.method,
    )
