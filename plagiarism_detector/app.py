import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import logging
import json
import uuid
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plagiarism_detector.log'),
        logging.StreamHandler()
    ]
)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Add template context
from . import add_template_context
add_template_context(app)

# Import core plagiarism detection logic
try:
    from .detector import EnhancedPlagiarismDetector, ASTAnalyzer, TokenProcessor, ScalableSimilarityAnalyzer
except ImportError:
    from detector import EnhancedPlagiarismDetector, ASTAnalyzer, TokenProcessor, ScalableSimilarityAnalyzer

# Initialize the detector
detector = EnhancedPlagiarismDetector(
    output_dir='results',
    similarity_threshold=0.7,
    max_workers=os.cpu_count()
)

# Add a basename filter for templates
@app.template_filter('basename')
def get_basename(path):
    return os.path.basename(path)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'c', 'cpp', 'h', 'hpp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_files():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'file1' not in request.files or 'file2' not in request.files:
            flash('Both files are required')
            return redirect(request.url)
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Check if filenames are empty
        if file1.filename == '' or file2.filename == '':
            flash('Both files are required')
            return redirect(request.url)
        
        # Check if files are allowed
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            flash('Only C/C++ files (.c, .cpp, .h, .hpp) are allowed')
            return redirect(request.url)
        
        # Create unique session ID for this comparison
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save files
        file1_path = os.path.join(session_dir, secure_filename(file1.filename))
        file2_path = os.path.join(session_dir, secure_filename(file2.filename))
        
        file1.save(file1_path)
        file2.save(file2_path)
        
        try:
            # Reset detector for clean comparison
            detector.submissions = {}
            
            # Analyze files
            detector.analyze_files([file1_path, file2_path])
            
            # Check if files were analyzed successfully
            if file1_path not in detector.submissions or file2_path not in detector.submissions:
                flash('Error: Failed to analyze one or both files')
                return redirect(request.url)
            
            # Calculate similarity
            similarity = detector.lsh_analyzer.compute_weighted_similarity(
                detector.submissions[file1_path],
                detector.submissions[file2_path]
            )
            
            # Store results in session
            session['comparison_result'] = {
                'file1': os.path.basename(file1_path),
                'file2': os.path.basename(file2_path),
                'similarity': similarity,
                'timestamp': datetime.now().isoformat()
            }
            
            return redirect(url_for('comparison_result'))
            
        except Exception as e:
            flash(f'Error: {str(e)}')
            logging.error(f"Error in file comparison: {str(e)}", exc_info=True)
            return redirect(request.url)
    
    return render_template('compare.html')

@app.route('/result')
def comparison_result():
    if 'comparison_result' not in session:
        return redirect(url_for('compare_files'))
    
    result = session['comparison_result']
    return render_template('result.html', result=result)

@app.route('/analyze_directory', methods=['GET', 'POST'])
def analyze_directory():
    if request.method == 'POST':
        # Check if directory was uploaded
        if 'directory' not in request.files:
            flash('Please upload a zip file containing your code files')
            return redirect(request.url)
        
        zip_file = request.files['directory']
        
        # Check if filename is empty
        if zip_file.filename == '':
            flash('Please select a zip file')
            return redirect(request.url)
        
        # Check if file is a zip
        if not zip_file.filename.endswith('.zip'):
            flash('Only zip files are allowed')
            return redirect(request.url)
        
        # Get threshold from form
        threshold = float(request.form.get('threshold', 0.7))
        
        # Create unique session ID for this analysis
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save zip file
        zip_path = os.path.join(session_dir, secure_filename(zip_file.filename))
        zip_file.save(zip_path)
        
        # Extract zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(session_dir)
        
        # Remove zip file after extraction
        os.remove(zip_path)
        
        try:
            # Reset detector
            detector.submissions = {}
            detector.comparison_results = []
            detector.similarity_threshold = threshold
            
            # Get all C/C++ files
            cpp_files = []
            for root, _, files in os.walk(session_dir):
                for file in files:
                    if file.endswith(('.cpp', '.c', '.h', '.hpp')):
                        cpp_files.append(os.path.join(root, file))
            
            if not cpp_files:
                flash('No C/C++ files found in the uploaded directory')
                return redirect(request.url)
            
            # Analyze files
            detector.analyze_files(cpp_files)
            
            # Find similarities
            detector.find_similarities()
            
            # Sort results by similarity
            sorted_results = sorted(
                detector.comparison_results,
                key=lambda x: x['similarity_metrics']['overall_similarity'],
                reverse=True
            )
            
            # Store results in session
            session['directory_results'] = {
                'results': sorted_results,
                'file_count': len(cpp_files),
                'timestamp': datetime.now().isoformat()
            }
            
            return redirect(url_for('directory_result'))
            
        except Exception as e:
            flash(f'Error: {str(e)}')
            logging.error(f"Error in directory analysis: {str(e)}", exc_info=True)
            return redirect(request.url)
    
    return render_template('analyze_directory.html')

@app.route('/directory_result')
def directory_result():
    if 'directory_results' not in session:
        return redirect(url_for('analyze_directory'))
    
    results = session['directory_results']
    return render_template('directory_result.html', results=results)

@app.route('/api/compare', methods=['POST'])
def api_compare_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both files are required'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Only C/C++ files (.c, .cpp, .h, .hpp) are allowed'}), 400
    
    # Create temporary directory
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save files
        file1_path = os.path.join(temp_dir, secure_filename(file1.filename))
        file2_path = os.path.join(temp_dir, secure_filename(file2.filename))
        
        file1.save(file1_path)
        file2.save(file2_path)
        
        # Reset detector for clean comparison
        detector.submissions = {}
        
        # Analyze files
        detector.analyze_files([file1_path, file2_path])
        
        # Calculate similarity
        similarity = detector.lsh_analyzer.compute_weighted_similarity(
            detector.submissions[file1_path],
            detector.submissions[file2_path]
        )
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'file1': file1.filename,
            'file2': file2.filename,
            'similarity': similarity
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logging.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 