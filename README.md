# Dietary Health Risks Analysis from NHANES Data

This project analyzes the National Health and Nutrition Examination Survey (NHANES) dataset to explore the relationships between dietary factors, environmental exposures, and health outcomes. The primary goal is to identify and modularize thousands of potential mediating variables (biomarkers, environmental chemicals, etc.) to understand their collective impact.

The analysis is performed through a memory-efficient, multi-stage pipeline designed to handle a large number of variables without requiring massive amounts of RAM.

## Project Structure

The repository is structured to separate raw data and results from the source code. Key files and directories include:

- `*.py`: Python scripts for different stages of the analysis.
- `requirements.txt`: Required Python packages.
- `.gitignore`: Specifies files and directories to be ignored by Git (e.g., data, results, virtual environments).

All data and results are generated locally and are not tracked by Git.

## How to Run the Analysis

1.  **Setup Environment**:
    It's recommended to use a virtual environment.
    ```bash
    python3 -m venv gdm_env
    source gdm_env/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    Run the crawler to download the necessary NHANES data files. This step is currently manual, and the data should be placed in the `NHANES_Data` directory.

3.  **Run the Main Analysis Pipeline**:
    The core logic is in `modularize_mediators_iteratively.py`. This script performs the entire four-stage analysis.
    ```bash
    python3 modularize_mediators_iteratively.py
    ```
    The script supports restarting from a specific stage if interrupted:
    ```bash
    # Example: Restart from stage 3 (Clustering)
    python3 modularize_mediators_iteratively.py --start_from_stage 3
    ```

## Scripts Description

-   **`nhanes_data_crawler.py`**: A utility to crawl and download data files from the NHANES website.
-   **`organize_nhanes_files.py`**: Organizes the downloaded raw files into a structured directory layout.
-   **`preprocess_nhanes.py`**: Preprocesses the raw data files into a consistent format (e.g., CSV).
-   **`extract_all_variables.py`**: Parses the preprocessed files to create a comprehensive variable dictionary.
-   **`modularize_mediators_iteratively.py`**: The main pipeline script. It executes the four stages of analysis:
    1.  **Metadata Pre-computation**: Calculates metadata (missing rate, variance) for potential mediators and filters them.
    2.  **Correlation Matrix Construction**: Builds a large correlation matrix iteratively to conserve memory.
    3.  **Clustering**: Performs hierarchical clustering on the correlation matrix to group variables into modules.
    4.  **Module Representation**: Uses PCA to create a representative variable (first principal component) for each module.
-   **`nhanes_causal_analysis.py` & `nhanes_exhaustive_causal_analysis.py`**: Scripts for downstream causal analysis using the generated module representatives (work in progress).
-   **`validate_data_format.py`**: A utility script to check and validate data formats.

## Analysis Results

The final outputs of the pipeline are saved in the `mediator_analysis_results_v2/` directory, which includes:
-   The final list of mediator variables.
-   The full correlation matrix (`correlation_matrix.csv`).
-   A sorted list of variable correlations (`correlation_list_sorted.csv`).
-   Clustering results, including a dendrogram visualization and module mappings.
-   The final module representatives file (`module_representatives_pca_v2.parquet`), which serves as the input for downstream analyses.