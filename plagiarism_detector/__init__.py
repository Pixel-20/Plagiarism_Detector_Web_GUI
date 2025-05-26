# Plagiarism Detection System
# Version: 1.0.0

from datetime import datetime

# Add current date to the context of all templates
from flask import Flask, current_app

def add_template_context(app):
    @app.context_processor
    def inject_now():
        return {'now': datetime.now()} 