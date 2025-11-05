"""
Python History Tour - Interactive Web Application
A visual journey through Python's evolution from 2.0.1 to 3.14
"""

from flask import Flask, render_template, jsonify
from python_versions import get_all_versions, get_code_examples, get_feature_examples

app = Flask(__name__)


@app.route('/')
def index():
    """Render the main page with Python history timeline"""
    return render_template('index.html')


@app.route('/api/versions')
def versions():
    """API endpoint to get all Python version data"""
    return jsonify(get_all_versions())


@app.route('/api/code-examples')
def code_examples():
    """API endpoint to get code comparison examples"""
    return jsonify(get_code_examples())


@app.route('/api/feature-examples')
def feature_examples():
    """API endpoint to get feature code examples"""
    return jsonify(get_feature_examples())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
