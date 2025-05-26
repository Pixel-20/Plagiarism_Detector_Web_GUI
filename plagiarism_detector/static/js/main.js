/**
 * Main JavaScript for Plagiarism Detection System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltips.length > 0) {
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip);
        });
    }
    
    // File upload visual feedback
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileUploadDiv = this.closest('.file-upload');
            const fileNameDiv = fileUploadDiv.querySelector('.file-name');
            
            if (this.files.length > 0) {
                fileNameDiv.textContent = this.files[0].name;
                fileUploadDiv.style.borderColor = '#28a745';
            } else {
                fileNameDiv.textContent = '';
                fileUploadDiv.style.borderColor = '#ccc';
            }
        });
    });
    
    // Threshold slider visual feedback
    const thresholdSliders = document.querySelectorAll('input[type="range"]');
    thresholdSliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + 'Value');
        if (valueDisplay) {
            slider.addEventListener('input', function() {
                valueDisplay.textContent = Math.round(this.value * 100) + '%';
            });
        }
    });
    
    // Form submission loading indicator
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const loadingIndicator = document.getElementById('loadingIndicator');
            const submitButton = this.querySelector('button[type="submit"]');
            
            if (loadingIndicator && submitButton) {
                // Check if form is valid before showing loading indicator
                if (this.checkValidity()) {
                    submitButton.disabled = true;
                    loadingIndicator.style.display = 'block';
                }
            }
        });
    });
    
    // Animate similarity meters on page load
    const similarityValues = document.querySelectorAll('.similarity-value');
    if (similarityValues.length > 0) {
        setTimeout(function() {
            similarityValues.forEach(value => {
                // Extract percentage from text content
                const percentage = parseFloat(value.textContent);
                if (!isNaN(percentage)) {
                    value.style.width = percentage + '%';
                }
            });
        }, 100);
    }
    
    // Modal detail view for similarity results
    const detailButtons = document.querySelectorAll('.view-details-btn');
    detailButtons.forEach(button => {
        button.addEventListener('click', function() {
            const file1 = this.getAttribute('data-file1');
            const file2 = this.getAttribute('data-file2');
            const similarityData = JSON.parse(this.getAttribute('data-similarity'));
            
            const modalFile1 = document.getElementById('modalFile1');
            const modalFile2 = document.getElementById('modalFile2');
            const metricsContainer = document.getElementById('metricsContainer');
            
            if (modalFile1 && modalFile2 && metricsContainer) {
                modalFile1.textContent = file1;
                modalFile2.textContent = file2;
                
                // Clear previous metrics
                metricsContainer.innerHTML = '';
                
                // Add metrics cards
                for (const [key, value] of Object.entries(similarityData)) {
                    if (key !== 'timestamp' && typeof value === 'number') {
                        addMetricCard(metricsContainer, formatMetricName(key), (value * 100).toFixed(2) + '%', getMetricDescription(key));
                    }
                }
            }
        });
    });
    
    // Helper function to format metric names
    function formatMetricName(key) {
        // Convert snake_case or camelCase to Title Case with spaces
        return key
            .replace(/_/g, ' ')
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, str => str.toUpperCase())
            .replace(/similarity/i, 'Similarity');
    }
    
    // Helper function to get metric descriptions
    function getMetricDescription(key) {
        const descriptions = {
            'overall_similarity': 'Weighted combination of all metrics',
            'token_similarity': 'Based on normalized code tokens',
            'sequence_similarity': 'Based on token sequences',
            'tree_edit_similarity': 'Based on code structure',
            'subtree_hash_similarity': 'Based on code subtrees',
            'structure_similarity': 'Based on control flow',
            'complexity_similarity': 'Based on algorithmic complexity',
            'confidence': 'Consistency across different metrics'
        };
        
        return descriptions[key] || 'Similarity metric';
    }
    
    // Helper function to add metric card
    function addMetricCard(container, title, value, description) {
        const col = document.createElement('div');
        col.className = 'col-md-6 mb-3';
        
        const card = document.createElement('div');
        card.className = 'card h-100';
        
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body text-center';
        
        const cardTitle = document.createElement('h5');
        cardTitle.className = 'card-title';
        cardTitle.textContent = title;
        
        const cardValue = document.createElement('div');
        cardValue.className = 'display-6';
        cardValue.textContent = value;
        
        const cardText = document.createElement('p');
        cardText.className = 'card-text text-muted small';
        cardText.textContent = description;
        
        cardBody.appendChild(cardTitle);
        cardBody.appendChild(cardValue);
        cardBody.appendChild(cardText);
        card.appendChild(cardBody);
        col.appendChild(card);
        
        container.appendChild(col);
    }
}); 