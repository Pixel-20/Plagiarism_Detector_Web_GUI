# Plagiarism Detection System for C/C++ Code

A comprehensive system for detecting plagiarism in C/C++ code submissions using advanced code analysis techniques.

## Features

- **Multiple Detection Techniques**:
  - Token-based analysis
  - Abstract Syntax Tree (AST) comparison
  - Control flow graph analysis
  - Algorithmic complexity metrics

- **Web Interface**:
  - Upload and compare individual files
  - Analyze entire directories
  - View detailed similarity reports

- **Machine Learning Integration**:
  - Dataset generation for training
  - Model training and optimization
  - Improved detection accuracy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/plagiarism-detector.git
   cd plagiarism-detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run.py
   ```

## Usage

### Web Interface

Access the web interface by opening a browser and navigating to:
```
http://localhost:5000
```

### Command Line Interface

The system provides several command-line tools:

#### Compare two files:
```
python plagiarism_detector/apply_model.py compare file1.cpp file2.cpp
```

#### Analyze a directory:
```
python plagiarism_detector/apply_model.py analyze /path/to/directory
```

#### Compare two directories:
```
python plagiarism_detector/apply_model.py compare-dirs /path/to/dir1 /path/to/dir2
```

### Dataset Generation and Model Training

The system includes tools for generating training datasets and training models:

#### Generate a training dataset:
```
python plagiarism_detector/dataset_generator.py --output dataset --originals 20 --variants 7
```

#### Train a model:
```
python plagiarism_detector/model_trainer.py --dataset dataset --output trained_model
```

#### Run the complete training pipeline:
```
python plagiarism_detector/run_training.py --output-dir training --originals 20 --variants 7
```

## How It Works

### Detection Process

1. **Code Parsing**: Source files are parsed to extract tokens and build AST representations
2. **Feature Extraction**: Multiple features are extracted from the code
3. **Similarity Analysis**: Various similarity metrics are computed
4. **Result Aggregation**: Results are combined using optimized weights

### Training Process

1. **Dataset Generation**: Creates original code samples and plagiarized variants
2. **Feature Extraction**: Extracts features from all samples
3. **Model Training**: Trains a classifier to detect plagiarism
4. **Parameter Optimization**: Finds optimal weights and thresholds
5. **Evaluation**: Evaluates model performance on test data

## Customization

### Adjusting Similarity Threshold

The default similarity threshold is determined by model training, but you can override it:

```python
detector = OptimizedPlagiarismDetector(model_dir="trained_model", similarity_threshold=0.8)
```

### Adding New Detection Techniques

The modular design allows for adding new detection techniques by extending the appropriate classes.

## Directory Structure

```
plagiarism_detector/
├── app.py                  # Flask web application
├── detector.py             # Core detection logic
├── dataset_generator.py    # Training dataset generator
├── model_trainer.py        # Model training and optimization
├── run_training.py         # Complete training pipeline
├── apply_model.py          # CLI for applying trained models
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── compare.html
│   └── results.html
├── static/                 # Static assets
│   ├── css/
│   └── js/
└── results/                # Output directory for results
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to all contributors who have helped with the development of this tool. 