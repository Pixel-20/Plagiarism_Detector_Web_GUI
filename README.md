# Plagiarism Detector

## Description

This project is a plagiarism detection tool designed to compare text documents and identify potential instances of plagiarism. It analyzes text similarity and provides a report on the findings.

## Features

*   Text comparison between multiple documents.
*   Similarity scoring.
*   Report generation.

## Directory Structure

```
.
├── plagiarism_detector/    # Main application logic
├── results/                # Output directory for plagiarism reports
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
├── requirements.txt        # Project dependencies
├── run.bat                 # Batch script to run the application on Windows
├── run.py                  # Main Python script to execute the plagiarism detector
├── plagiarism_detector.log # Log file for the application
└── README.md               # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the plagiarism detector, execute the `run.py` script or the `run.bat` batch file:

Using Python:
```bash
python run.py
```

Or, on Windows, using the batch file:
```bash
run.bat
```

The results of the plagiarism check will be stored in the `results/` directory.

## Technical Details

The plagiarism detection process involves several key steps:

*   **AST Parsing**: The C/C++ source code is first parsed into Abstract Syntax Trees (ASTs) using `clang`. This allows for a structured representation of the code, capturing its syntactic and semantic elements.
*   **Token Processing**: Tokens are extracted from the source code. These tokens are then filtered, weighted based on significance (e.g., keywords, identifiers), and identifiers can be normalized. Specific C++ patterns like memory management, template usage, and STL usage are also identified.
*   **Control Flow Graph (CFG) Generation**: For a deeper understanding of the program's logic, Control Flow Graphs are generated from the ASTs. This helps in analyzing the execution paths within the code.
*   **Similarity Analysis**: Various techniques are employed to compare the processed code representations (ASTs, token sequences, CFGs) to determine similarity scores between documents. This likely includes methods like cosine similarity on TF-IDF vectors of tokens, graph matching for CFGs, and tree edit distance for ASTs.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. 