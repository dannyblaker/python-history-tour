# Python History Tour

An interactive visual journey through Python's evolution from version 2.0.1 (2001) to 3.14 (2025).

[![A Danny Blaker project badge](https://github.com/dannyblaker/dannyblaker.github.io/blob/main/danny_blaker_project_badge.svg)](https://github.com/dannyblaker/)

## Features

- 📅 **Interactive Timeline**: Explore all major Python releases with detailed feature highlights
- 💡 **Clickable Feature Examples**: Click any feature in the timeline to see a code example in a popup
- ⚡ **Code Comparison**: See the dramatic improvements between Python 2.0.1 and 3.14 side-by-side
- 📊 **Statistics Dashboard**: Visualize Python's growth over 24 years
- 🎨 **A polished UI**: Modern, responsive design with smooth animations

## Quick Start

Run the entire project with a single command:

```bash
docker compose up
```

Then open your browser to: **http://localhost:5000**

## Screenshot

![Screenshot](readme_assets/screenshot.png)

## What's Included

### Timeline View
- All major Python versions from 2.0.1 to 3.14
- Release dates and key features for each version
- **Click any feature** to see a code example in a popup modal
- Filter by Python 2.x or 3.x versions
- Visual timeline with animations
- Over 50 interactive feature examples with code snippets

### Code Comparison
- Real-world example comparing Python 2.0.1 and 3.14
- Same functionality, showing 24 years of improvements
- Highlights modern features like:
  - F-strings
  - Type hints
  - Dataclasses
  - Pattern matching
  - Context managers
  - And much more!

## Project Structure

```
history_of_python/
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── app.py                # Flask application
├── python_versions.py    # Version data and code examples
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── app.js        # Frontend JavaScript
```

## Technologies Used

- **Backend**: Python 3.14, Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Container**: Docker, Docker Compose

## Development

To run without Docker:

```bash
pip install -r requirements.txt
python app.py
```