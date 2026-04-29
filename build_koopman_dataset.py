from pathlib import Path
from koopman_io import build_batch_npz


InputPatterns = ["runs/*.npz"]
OutputPath = "baseline_dataset.npz"


def main():
    run_paths = []
    for pattern in InputPatterns:
        run_paths.extend(sorted(Path().glob(pattern)))

    if not run_paths:
        raise FileNotFoundError("No run NPZ files matched the configured input patterns.")

    build_batch_npz(run_paths, OutputPath)
    print(f"Wrote batch dataset to {Path(OutputPath).resolve()}")


if __name__ == "__main__":
    main()
