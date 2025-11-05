// Python History Tour - Main JavaScript

let versionsData = [];
let codeExamplesData = {};
let featureExamplesData = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    loadData();
    setupModal();
});

// Setup navigation between sections
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSection = button.getAttribute('data-section');

            // Update active states
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            sections.forEach(section => {
                section.classList.remove('active');
            });

            document.getElementById(`${targetSection}-section`).classList.add('active');
        });
    });
}

// Load data from API
async function loadData() {
    try {
        // Load versions data
        const versionsResponse = await fetch('/api/versions');
        const versionsJson = await versionsResponse.json();
        versionsData = versionsJson.versions;

        // Load code examples
        const examplesResponse = await fetch('/api/code-examples');
        codeExamplesData = await examplesResponse.json();

        // Load feature examples
        const featureResponse = await fetch('/api/feature-examples');
        featureExamplesData = await featureResponse.json();

        // Render all sections
        renderTimeline(versionsData);
        renderCodeComparison();
        renderStatistics();
        setupTimelineFilters();
        setupFeatureClicks();

    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Render timeline section
function renderTimeline(versions) {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';

    versions.forEach((version, index) => {
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.setAttribute('data-era', version.era);

        const isPython2 = version.version.startsWith('2');
        const eraClass = isPython2 ? 'python2' : 'python3';

        item.innerHTML = `
            <div class="timeline-content ${eraClass}">
                <div class="version-header">
                    <div class="version-number">Python ${version.version}</div>
                    <div class="release-date">${version.release_date}</div>
                </div>
                <div class="version-description">${version.description}</div>
                <div class="highlights">
                    <h4>✨ Key Features</h4>
                    <ul>
                        ${version.highlights.map(h => `<li>${h}</li>`).join('')}
                    </ul>
                </div>
                ${version.major_peps ? `
                    <div class="peps">
                        <small style="color: var(--text-muted);">
                            Major PEPs: ${version.major_peps.join(', ')}
                        </small>
                    </div>
                ` : ''}
            </div>
            <div class="timeline-marker"></div>
        `;

        // Add staggered animation delay
        item.style.animationDelay = `${index * 0.05}s`;

        timeline.appendChild(item);
    });
}

// Setup timeline filters
function setupTimelineFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            const era = button.getAttribute('data-era');

            // Update active state
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Filter timeline items
            filterTimeline(era);
        });
    });
}

// Filter timeline by era
function filterTimeline(era) {
    const items = document.querySelectorAll('.timeline-item');

    items.forEach(item => {
        if (era === 'all') {
            item.style.display = 'block';
        } else {
            const itemEra = item.getAttribute('data-era');
            if (itemEra && itemEra.includes(era)) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        }
    });
}

// Render code comparison section
function renderCodeComparison() {
    // Python 2.0.1 code
    const python2Code = document.getElementById('python2-code');
    python2Code.textContent = codeExamplesData.python_2_0_1.code;

    const python2Issues = document.getElementById('python2-issues');
    python2Issues.innerHTML = codeExamplesData.python_2_0_1.issues
        .map(issue => `<li>${issue}</li>`)
        .join('');

    // Python 3.14 code
    const python3Code = document.getElementById('python3-code');
    python3Code.textContent = codeExamplesData.python_3_14.code;

    const python3Improvements = document.getElementById('python3-improvements');
    python3Improvements.innerHTML = codeExamplesData.python_3_14.improvements
        .map(imp => `<li>${imp}</li>`)
        .join('');

    // Comparison summary
    const improvementsGrid = document.getElementById('improvements-grid');
    improvementsGrid.innerHTML = codeExamplesData.comparison.points
        .map(point => `
            <div class="improvement-card">
                <h4>${point.category}</h4>
                <p>${point.improvement}</p>
            </div>
        `)
        .join('');
}

// Render statistics section
function renderStatistics() {
    // Calculate statistics
    const totalYears = new Date().getFullYear() - 2001;
    const totalVersions = versionsData.length;
    const totalFeatures = versionsData.reduce((sum, v) => sum + v.highlights.length, 0);

    // Update stat cards
    document.getElementById('stat-years').textContent = totalYears;
    document.getElementById('stat-versions').textContent = totalVersions;
    document.getElementById('stat-features').textContent = `${totalFeatures}+`;

    // Animate numbers
    animateNumbers();
}

// Animate number counters
function animateNumbers() {
    const statValues = document.querySelectorAll('.stat-value');

    statValues.forEach(stat => {
        const text = stat.textContent;
        const match = text.match(/\d+/);

        if (match) {
            const target = parseInt(match[0]);
            const suffix = text.replace(/\d+/, '');

            animateValue(stat, 0, target, 1500, suffix);
        }
    });
}

// Animate a number from start to end
function animateValue(element, start, end, duration, suffix = '') {
    const startTimestamp = Date.now();

    const step = () => {
        const now = Date.now();
        const progress = Math.min((now - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);

        element.textContent = current + suffix;

        if (progress < 1) {
            requestAnimationFrame(step);
        }
    };

    requestAnimationFrame(step);
}

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all timeline items
setTimeout(() => {
    document.querySelectorAll('.timeline-item').forEach(item => {
        observer.observe(item);
    });
}, 100);

// Setup modal functionality
function setupModal() {
    const modal = document.getElementById('feature-modal');
    const closeBtn = document.getElementById('close-modal');

    // Close modal when clicking close button
    closeBtn.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            modal.classList.remove('active');
        }
    });
}

// Setup feature click handlers
function setupFeatureClicks() {
    // Use event delegation for dynamically added content
    document.getElementById('timeline').addEventListener('click', (e) => {
        const listItem = e.target.closest('.highlights li');
        if (listItem) {
            const featureName = listItem.textContent.trim();
            showFeatureExample(featureName);
        }
    });
}

// Show feature example in modal
function showFeatureExample(featureName) {
    const featureData = featureExamplesData[featureName];

    if (!featureData) {
        // Feature example not found, show generic message
        showGenericFeatureInfo(featureName);
        return;
    }

    const modal = document.getElementById('feature-modal');
    const title = document.getElementById('feature-title');
    const description = document.getElementById('feature-description');
    const code = document.getElementById('feature-code');

    title.textContent = featureName;
    description.textContent = featureData.description;
    code.textContent = featureData.code;

    modal.classList.add('active');
}

// Show generic info when specific example not available
function showGenericFeatureInfo(featureName) {
    const modal = document.getElementById('feature-modal');
    const title = document.getElementById('feature-title');
    const description = document.getElementById('feature-description');
    const code = document.getElementById('feature-code');

    title.textContent = featureName;
    description.textContent = `${featureName} was an important addition to Python. Click on other features to see code examples!`;
    code.textContent = '# Code example coming soon!\n# This feature significantly improved Python.';

    modal.classList.add('active');
}
