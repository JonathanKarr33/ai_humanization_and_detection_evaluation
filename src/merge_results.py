import argparse
import os

import pandas as pd


def merge_csv_files(input_files: list[str], output_file: str):
    merged_df = None

    input_files = [f for f in input_files if f.endswith(".csv")]

    for file_path in input_files:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Get the filename without extension for prefixing
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = file_name.removeprefix("results_")

        # Identify join columns and data columns
        join_cols = ["paper_id", "process"]

        # Check if join columns exist
        assert set(join_cols).issubset(df.columns)

        # Prefix non-join columns
        rename_dict = {
            col: f"{file_name}_{col}" for col in df.columns if col not in join_cols
        }
        df = df.rename(columns=rename_dict)

        if merged_df is None:
            merged_df = df
        else:
            # Merge on paper_id and process
            merged_df = pd.merge(merged_df, df, on=join_cols, how="outer")

    assert merged_df is not None

    # Reorder columns to ensure paper_id and process are first
    cols = list(merged_df.columns)
    join_cols = ["paper_id", "process"]
    # Filter join_cols to only those present (though they should be) and remove from original list
    present_join_cols = [c for c in join_cols if c in cols]
    other_cols = [c for c in cols if c not in present_join_cols]

    merged_df = merged_df[present_join_cols + other_cols]

    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge CSV results on paper_id and process."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Input CSV files to merge"
    )
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    merge_csv_files(args.input, args.output)
