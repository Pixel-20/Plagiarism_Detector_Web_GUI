@echo off
echo Plagiarism Detection System
echo ===========================

if "%1"=="train" (
    echo Running training pipeline...
    python plagiarism_detector/run_training.py %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

if "%1"=="compare" (
    echo Comparing files...
    python plagiarism_detector/apply_model.py compare %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

if "%1"=="analyze" (
    echo Analyzing directory...
    python plagiarism_detector/apply_model.py analyze %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

if "%1"=="compare-dirs" (
    echo Comparing directories...
    python plagiarism_detector/apply_model.py compare-dirs %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

echo Starting web interface...
python run.py

:end 