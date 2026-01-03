from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import re
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2
import docx
import json
from datetime import datetime, timedelta
import hashlib

app = Flask(__name__)
app.secret_key = 'ai-resume-analyzer-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx', 'txt'}

# Ensure nltk data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Sample job descriptions database
JOB_DESCRIPTIONS = {
    'software_engineer': {
        'title': 'Software Engineer',
        'description': '''
            We are seeking a skilled Software Engineer with:
            - 3+ years experience in Python and JavaScript
            - Strong knowledge of web frameworks (Flask, Django, React)
            - Experience with databases (SQL, MongoDB, PostgreSQL)
            - Familiarity with cloud platforms (AWS, Azure)
            - Understanding of AI/ML concepts and libraries
            - Excellent problem-solving and communication skills
            - Bachelor's degree in Computer Science or related field
            - Experience with Git, Docker, CI/CD pipelines
        ''',
        'keywords': ['python', 'javascript', 'flask', 'django', 'react', 'sql', 
                    'mongodb', 'aws', 'azure', 'git', 'docker', 'machine learning',
                    'api', 'rest', 'agile', 'scrum']
    },
    'data_scientist': {
        'title': 'Data Scientist',
        'description': '''
            Looking for a Data Scientist with:
            - Strong background in statistics and machine learning
            - Experience with Python, R, SQL
            - Knowledge of ML libraries (scikit-learn, TensorFlow, PyTorch)
            - Experience with data visualization tools
            - Big data technologies (Hadoop, Spark)
            - Excellent analytical and problem-solving skills
            - Master's or PhD in relevant field
        ''',
        'keywords': ['python', 'r', 'sql', 'machine learning', 'tensorflow', 'pytorch',
                    'scikit-learn', 'statistics', 'data analysis', 'spark', 'hadoop',
                    'pandas', 'numpy', 'data visualization']
    },
    'web_developer': {
        'title': 'Web Developer',
        'description': '''
            Web Developer position requires:
            - Proficiency in HTML5, CSS3, JavaScript
            - Experience with frontend frameworks (React, Angular, Vue.js)
            - Backend development experience (Node.js, Express)
            - Responsive design principles
            - Version control with Git
            - RESTful API development
            - Performance optimization techniques
        ''',
        'keywords': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node.js',
                    'express', 'responsive design', 'git', 'rest api', 'webpack']
    }
}

# Mock data for Learning Paths
LEARNING_PATHS_DATA = {
    'courses': [
        {
            'id': 1,
            'category': 'DEVELOPMENT',
            'badge': 'FREE',
            'title': 'Full Stack Web Development',
            'description': 'Master frontend and backend development with modern frameworks and best practices.',
            'rating': 4.8,
            'reviews': 1234,
            'duration': '8 weeks',
            'enrolled': 5678,
            'skills': ['HTML/CSS', 'JavaScript', 'React', 'Node.js', 'MongoDB', 'Docker'],
            'price': 0,
            'progress': 65,
            'modules_completed': 12,
            'total_modules': 20,
            'last_accessed': '2 days ago',
            'instructor': 'John Doe',
            'difficulty': 'Intermediate'
        },
        {
            'id': 2,
            'category': 'CS FUNDAMENTALS',
            'badge': 'FREE',
            'title': 'Data Structures & Algorithms',
            'description': 'Master essential algorithms and data structures for technical interviews and coding challenges.',
            'rating': 4.9,
            'reviews': 2341,
            'duration': '6 weeks',
            'enrolled': 8912,
            'skills': ['Arrays', 'Linked Lists', 'Trees', 'Graphs', 'Dynamic Programming', 'Big O'],
            'price': 0,
            'progress': 42,
            'modules_completed': 8,
            'total_modules': 19,
            'last_accessed': '1 week ago',
            'instructor': 'Jane Smith',
            'difficulty': 'Advanced'
        },
        {
            'id': 3,
            'category': 'DATA SCIENCE',
            'badge': 'PREMIUM',
            'title': 'Machine Learning & AI',
            'description': 'Learn machine learning algorithms, neural networks, and AI applications with Python.',
            'rating': 4.7,
            'reviews': 1987,
            'duration': '10 weeks',
            'enrolled': 3456,
            'skills': ['Python', 'TensorFlow', 'PyTorch', 'NLP', 'Computer Vision'],
            'price': 99,
            'progress': 0,
            'modules_completed': 0,
            'total_modules': 24,
            'last_accessed': None,
            'instructor': 'Dr. Alan Turing',
            'difficulty': 'Advanced'
        },
        {
            'id': 4,
            'category': 'DEVOPS',
            'badge': 'FREE',
            'title': 'Cloud & DevOps Engineering',
            'description': 'Master AWS, Docker, Kubernetes, and CI/CD pipelines for modern infrastructure.',
            'rating': 4.6,
            'reviews': 1543,
            'duration': '7 weeks',
            'enrolled': 4321,
            'skills': ['AWS', 'Docker', 'Kubernetes', 'Terraform', 'Jenkins'],
            'price': 0,
            'progress': 0,
            'modules_completed': 0,
            'total_modules': 18,
            'last_accessed': None,
            'instructor': 'Mike Cloud',
            'difficulty': 'Intermediate'
        }
    ],
    'categories': ['All Courses', 'Web Development', 'Data Science', 'Mobile Dev', 'Cloud & DevOps', 'Data Structures']
}

# Mock data for Mock Interviews
MOCK_INTERVIEWS_DATA = {
    'interviews': [
        {
            'id': 1,
            'title': 'Software Engineer - FAANG',
            'company': 'Google',
            'level': 'Intermediate',
            'duration': '45 mins',
            'questions': 15,
            'rating': 4.7,
            'completed': True,
            'score': 82,
            'date': '2024-11-20',
            'type': 'Technical',
            'status': 'completed',
            'feedback': 'Excellent problem-solving skills'
        },
        {
            'id': 2,
            'title': 'Data Scientist',
            'company': 'Meta',
            'level': 'Advanced',
            'duration': '60 mins',
            'questions': 20,
            'rating': 4.8,
            'completed': False,
            'score': None,
            'date': None,
            'type': 'Technical + Behavioral',
            'status': 'available',
            'feedback': None
        },
        {
            'id': 3,
            'title': 'Frontend Developer',
            'company': 'Microsoft',
            'level': 'Beginner',
            'duration': '30 mins',
            'questions': 10,
            'rating': 4.5,
            'completed': False,
            'score': None,
            'date': None,
            'type': 'Technical',
            'status': 'available',
            'feedback': None
        },
        {
            'id': 4,
            'title': 'DevOps Engineer',
            'company': 'Amazon',
            'level': 'Intermediate',
            'duration': '50 mins',
            'questions': 18,
            'rating': 4.6,
            'completed': True,
            'score': 75,
            'date': '2024-11-15',
            'type': 'Technical',
            'status': 'completed',
            'feedback': 'Good knowledge of cloud services'
        }
    ]
}

# Mock data for Certifications
CERTIFICATIONS_DATA = {
    'certifications': [
        {
            'id': 1,
            'title': 'AWS Certified Solutions Architect',
            'provider': 'Amazon Web Services',
            'level': 'Associate',
            'duration': '3 months',
            'validity': '3 years',
            'exam_fee': 150,
            'pass_rate': '72%',
            'popularity': 95,
            'enrolled': True,
            'progress': 30,
            'exam_date': '2024-12-15',
            'modules': 12,
            'completed_modules': 4,
            'skills': ['AWS', 'Cloud Architecture', 'Security', 'Networking']
        },
        {
            'id': 2,
            'title': 'Google Cloud Professional Data Engineer',
            'provider': 'Google Cloud',
            'level': 'Professional',
            'duration': '4 months',
            'validity': '2 years',
            'exam_fee': 200,
            'pass_rate': '65%',
            'popularity': 88,
            'enrolled': False,
            'progress': 0,
            'exam_date': None,
            'modules': 15,
            'completed_modules': 0,
            'skills': ['BigQuery', 'Dataflow', 'Pub/Sub', 'ML Engine']
        },
        {
            'id': 3,
            'title': 'Microsoft Azure Fundamentals',
            'provider': 'Microsoft',
            'level': 'Fundamentals',
            'duration': '2 months',
            'validity': 'Lifetime',
            'exam_fee': 99,
            'pass_rate': '85%',
            'popularity': 92,
            'enrolled': True,
            'progress': 60,
            'exam_date': '2024-11-30',
            'modules': 8,
            'completed_modules': 5,
            'skills': ['Azure Basics', 'Cloud Concepts', 'Core Services']
        },
        {
            'id': 4,
            'title': 'Kubernetes Administrator (CKA)',
            'provider': 'Linux Foundation',
            'level': 'Professional',
            'duration': '3 months',
            'validity': '3 years',
            'exam_fee': 300,
            'pass_rate': '68%',
            'popularity': 90,
            'enrolled': False,
            'progress': 0,
            'exam_date': None,
            'modules': 14,
            'completed_modules': 0,
            'skills': ['Kubernetes', 'Docker', 'Networking', 'Security']
        }
    ]
}

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(filepath):
    """Extract text from PDF files"""
    try:
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(filepath):
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(filepath)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

def extract_text_from_file(filepath, extension):
    """Extract text based on file extension"""
    if extension == 'pdf':
        return extract_text_from_pdf(filepath)
    elif extension in ['doc', 'docx']:
        return extract_text_from_docx(filepath)
    elif extension == 'txt':
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except:
            with open(filepath, 'r', encoding='latin-1') as file:
                return file.read().strip()
    return ""

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_ats_score(resume_text, job_description):
    """Calculate ATS (Applicant Tracking System) Score"""
    score = 0
    max_score = 100
    factors = []
    detailed_breakdown = []
    
    resume_lower = resume_text.lower()
    job_desc_lower = job_description.lower()
    
    # 1. Keyword Matching (35 points)
    keywords = re.findall(r'\b[a-z]+\b', job_desc_lower)
    keywords = [word for word in keywords if len(word) > 3 and word not in stopwords.words('english')]
    keywords = list(set(keywords))[:50]  # Top 50 unique keywords
    
    found_keywords = []
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', resume_lower):
            found_keywords.append(keyword)
    
    keyword_score = min(len(found_keywords) * 0.7, 35)
    score += keyword_score
    
    factors.append(f"Keywords matched: {len(found_keywords)}/{len(keywords)}")
    detailed_breakdown.append({
        'category': 'Keyword Matching',
        'score': round(keyword_score, 1),
        'max_score': 35,
        'details': f"Found {len(found_keywords)} relevant keywords"
    })
    
    # 2. Resume Structure & Sections (25 points)
    sections = {
        'contact': ['email', 'phone', 'address', 'linkedin', 'github'],
        'summary': ['summary', 'objective', 'profile'],
        'experience': ['experience', 'work history', 'employment'],
        'education': ['education', 'degree', 'university', 'college'],
        'skills': ['skills', 'technical skills', 'competencies'],
        'projects': ['projects', 'portfolio'],
        'certifications': ['certifications', 'certificate']
    }
    
    sections_found = 0
    section_details = []
    
    for section_name, section_keywords in sections.items():
        found = any(keyword in resume_lower for keyword in section_keywords)
        if found:
            sections_found += 1
            section_details.append(section_name)
    
    structure_score = min((sections_found / 7) * 25, 25)
    score += structure_score
    
    factors.append(f"Sections found: {sections_found}/7")
    detailed_breakdown.append({
        'category': 'Resume Structure',
        'score': round(structure_score, 1),
        'max_score': 25,
        'details': f"Found sections: {', '.join(section_details)}"
    })
    
    # 3. Length & Readability (20 points)
    word_count = len(resume_text.split())
    sentence_count = len(re.split(r'[.!?]+', resume_text))
    
    if 400 <= word_count <= 800:
        length_score = 20
        factors.append("Ideal resume length ✓")
    elif 300 <= word_count < 400 or 800 < word_count <= 1000:
        length_score = 15
        factors.append(f"Resume length ({word_count} words) - acceptable")
    else:
        length_score = 10
        factors.append(f"Resume length ({word_count} words) - needs optimization")
    
    score += length_score
    
    # Calculate readability (average sentence length)
    avg_sentence_length = word_count / max(sentence_count, 1)
    if 15 <= avg_sentence_length <= 25:
        factors.append("Good readability ✓")
    else:
        factors.append(f"Consider shorter sentences (avg: {avg_sentence_length:.1f} words)")
    
    detailed_breakdown.append({
        'category': 'Length & Readability',
        'score': length_score,
        'max_score': 20,
        'details': f"{word_count} words, {sentence_count} sentences"
    })
    
    # 4. Action Verbs & Quantifiable Achievements (20 points)
    action_verbs = [
        'achieved', 'improved', 'increased', 'decreased', 'developed',
        'implemented', 'managed', 'led', 'created', 'designed',
        'optimized', 'reduced', 'saved', 'generated', 'built'
    ]
    
    verbs_found = sum(1 for verb in action_verbs if verb in resume_lower)
    
    # Check for quantifiable results (numbers, percentages, $ amounts)
    quant_patterns = [
        r'\d+%', r'\$\d+', r'\d+\s*(?:x|times)', 
        r'increased by \d+', r'reduced by \d+', r'saved \$\d+'
    ]
    
    quant_achievements = 0
    for pattern in quant_patterns:
        quant_achievements += len(re.findall(pattern, resume_lower, re.IGNORECASE))
    
    action_score = min((verbs_found * 0.5) + (quant_achievements * 1), 20)
    score += action_score
    
    factors.append(f"Action verbs: {verbs_found}, Quantifiable results: {quant_achievements}")
    detailed_breakdown.append({
        'category': 'Achievements & Impact',
        'score': round(action_score, 1),
        'max_score': 20,
        'details': f"{verbs_found} action verbs, {quant_achievements} quantifiable achievements"
    })
    
    # Ensure score doesn't exceed 100
    final_score = min(round(score), 100)
    
    return final_score, factors, detailed_breakdown, found_keywords

def calculate_similarity_score(resume_text, job_description):
    """Calculate text similarity using TF-IDF and Cosine Similarity"""
    try:
        if not resume_text.strip() or not job_description.strip():
            return 0
        
        # Preprocess texts
        resume_clean = preprocess_text(resume_text)
        job_clean = preprocess_text(job_description)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage
        similarity_percentage = round(similarity * 100, 2)
        
        return similarity_percentage
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0

def analyze_resume_content(resume_text):
    """Analyze resume content and provide detailed insights"""
    insights = []
    metrics = {}
    
    # Basic metrics
    words = resume_text.split()
    word_count = len(words)
    char_count = len(resume_text)
    sentence_count = len(re.split(r'[.!?]+', resume_text))
    
    metrics['word_count'] = word_count
    metrics['char_count'] = char_count
    metrics['sentence_count'] = sentence_count
    metrics['avg_word_length'] = char_count / max(word_count, 1)
    metrics['avg_sentence_length'] = word_count / max(sentence_count, 1)
    
    insights.append(f"📊 Document Statistics: {word_count} words, {sentence_count} sentences")
    
    # Technical Skills Detection
    tech_categories = {
        'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift', 'kotlin'],
        'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd'],
        'Data Science': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'r', 'spark'],
        'Tools & Platforms': ['git', 'jira', 'confluence', 'slack', 'trello', 'figma']
    }
    
    found_skills = {}
    for category, skills in tech_categories.items():
        found = [skill for skill in skills if skill in resume_text.lower()]
        if found:
            found_skills[category] = found
            insights.append(f"🔧 {category}: {', '.join(found[:5])}")
    
    # Experience Detection
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience:\s*\d+',
        r'\d+\s*-\s*\d+\s*years'
    ]
    
    years_experience = 0
    for pattern in experience_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            try:
                years_experience = max(years_experience, int(match))
            except:
                pass
    
    if years_experience > 0:
        insights.append(f"⏳ Years of Experience: {years_experience}+ years")
        metrics['experience_years'] = years_experience
    
    # Education Detection
    degree_keywords = ['bachelor', 'master', 'phd', 'b\.?s\.?', 'b\.?a\.?', 'm\.?s\.?', 'm\.?a\.?', 'mba']
    education_found = any(re.search(r'\b' + deg + r'\b', resume_text.lower()) for deg in degree_keywords)
    
    if education_found:
        insights.append("🎓 Higher education degree detected")
    
    # Contact Information Check
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\+?1?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
    
    has_email = bool(re.search(email_pattern, resume_text))
    has_phone = bool(re.search(phone_pattern, resume_text))
    
    if has_email and has_phone:
        insights.append("📞 Contact information: Complete ✓")
    else:
        missing = []
        if not has_email: missing.append("email")
        if not has_phone: missing.append("phone")
        insights.append(f"⚠️ Missing contact info: {', '.join(missing)}")
    
    return insights, metrics, found_skills

def generate_improvement_suggestions(ats_score, factors, found_keywords, resume_text, job_title):
    """Generate personalized improvement suggestions"""
    suggestions = []
    
    # Overall assessment
    if ats_score >= 80:
        suggestions.append("🎉 Excellent resume! Very ATS-friendly.")
        suggestions.append("💡 Minor improvements: Add more quantifiable achievements and industry-specific keywords.")
    elif ats_score >= 60:
        suggestions.append("👍 Good resume structure with room for improvement.")
        suggestions.append("🎯 Focus on: Increasing keyword density and adding specific metrics.")
    else:
        suggestions.append("⚠️ Resume needs significant optimization for ATS systems.")
        suggestions.append("🔧 Priority fixes: Add missing sections and relevant keywords.")
    
    # Specific suggestions based on factors
    for factor in factors:
        if 'Keywords matched' in factor:
            match = re.search(r'(\d+)/(\d+)', factor)
            if match:
                found, total = map(int, match.groups())
                if found < total * 0.6:
                    suggestions.append(f"🔑 Add {total - found} more job-specific keywords")
        
        if 'Sections found' in factor:
            match = re.search(r'(\d+)/7', factor)
            if match and int(match.group(1)) < 5:
                missing_sections = 7 - int(match.group(1))
                suggestions.append(f"📑 Add {missing_sections} missing resume sections")
        
        if 'Resume length' in factor and 'needs optimization' in factor:
            suggestions.append("📏 Optimize length to 400-800 words for best results")
        
        if 'Action verbs' in factor:
            suggestions.append("⚡ Use more action verbs to describe achievements")
    
    # Check for quantifiable results
    if 'Quantifiable results' in str(factors):
        if '0' in str(factors):
            suggestions.append("📈 Add numbers, percentages, and metrics to show impact")
    
    # Industry-specific suggestions
    if 'software' in job_title.lower() or 'engineer' in job_title.lower():
        suggestions.append("💻 Include specific technologies, frameworks, and project links")
    elif 'data' in job_title.lower():
        suggestions.append("📊 Highlight specific algorithms, tools, and analysis methods")
    elif 'manager' in job_title.lower():
        suggestions.append("👥 Emphasize leadership, team size, and business impact")
    
    return suggestions

def calculate_overall_score(ats_score, similarity_score, content_metrics):
    """Calculate weighted overall score"""
    # Base weights
    weights = {
        'ats': 0.5,
        'similarity': 0.3,
        'content': 0.2
    }
    
    # Calculate content quality score (based on metrics)
    content_score = 0
    
    # Word count score (0-10)
    word_count = content_metrics.get('word_count', 0)
    if 400 <= word_count <= 800:
        content_score += 10
    elif 300 <= word_count < 400 or 800 < word_count <= 1000:
        content_score += 7
    else:
        content_score += 4
    
    # Structure score (0-10)
    sections = ['experience', 'education', 'skills', 'projects']
    found_sections = sum(1 for section in sections if section in content_metrics.get('found_sections', []))
    content_score += (found_sections / 4) * 10
    
    # Calculate weighted average
    overall = (
        ats_score * weights['ats'] +
        similarity_score * weights['similarity'] +
        content_score * weights['content']
    )
    
    return round(overall, 1)

# ==================== MAIN APPLICATION ROUTES ====================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple authentication for demo
        if username and password:
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Please enter username and password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # In a real app, save to database
        if username and email and password:
            session['user'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Calculate user stats
    user_stats = {
        'resumes_analyzed': 5,
        'average_score': 78,
        'improvement': '+12%',
        'interviews_completed': 2,
        'courses_enrolled': 2,
        'certifications_in_progress': 2
    }
    
    return render_template('dashboard.html', 
                         username=session['user'],
                         job_roles=list(JOB_DESCRIPTIONS.keys()),
                         stats=user_stats)

@app.route('/analyze-resume', methods=['GET', 'POST'])
def analyze_resume():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check file upload
        if 'resume' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['resume']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Get job role
        job_role = request.form.get('job_role', 'software_engineer')
        custom_job_desc = request.form.get('custom_job_desc', '')
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get file extension
            extension = filename.rsplit('.', 1)[1].lower()
            
            # Extract text
            resume_text = extract_text_from_file(filepath, extension)
            
            if not resume_text.strip():
                flash('Could not extract text from the file. Try a different format.', 'error')
                return redirect(request.url)
            
            # Get job description
            if custom_job_desc:
                job_description = custom_job_desc
                job_title = "Custom Job Description"
            else:
                job_data = JOB_DESCRIPTIONS.get(job_role, JOB_DESCRIPTIONS['software_engineer'])
                job_description = job_data['description']
                job_title = job_data['title']
            
            # Calculate scores
            ats_score, ats_factors, ats_breakdown, found_keywords = calculate_ats_score(resume_text, job_description)
            similarity_score = calculate_similarity_score(resume_text, job_description)
            
            # Analyze content
            content_insights, content_metrics, found_skills = analyze_resume_content(resume_text)
            
            # Calculate overall score
            overall_score = calculate_overall_score(ats_score, similarity_score, content_metrics)
            
            # Generate suggestions
            suggestions = generate_improvement_suggestions(ats_score, ats_factors, found_keywords, resume_text, job_title)
            
            # Save analysis to session for results page
            analysis_data = {
                'filename': filename,
                'job_title': job_title,
                'overall_score': overall_score,
                'ats_score': ats_score,
                'similarity_score': similarity_score,
                'content_insights': content_insights,
                'suggestions': suggestions,
                'found_keywords': found_keywords[:10],  # Top 10
                'found_skills': found_skills,
                'ats_breakdown': ats_breakdown,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            session['last_analysis'] = analysis_data
            
            return redirect(url_for('resume_results'))
    
    return render_template('analyze_resume.html', job_roles=JOB_DESCRIPTIONS.keys())

@app.route('/results')
def resume_results():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    analysis_data = session.get('last_analysis', {})
    
    if not analysis_data:
        flash('No analysis found. Please upload a resume first.', 'error')
        return redirect(url_for('analyze_resume'))
    
    return render_template('resume_results.html', **analysis_data)

# ==================== LEARNING PATHS ROUTES ====================
@app.route('/learning-paths')
def learning_paths():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Filter enrolled vs available courses
    enrolled_courses = [c for c in LEARNING_PATHS_DATA['courses'] if c['progress'] > 0]
    available_courses = [c for c in LEARNING_PATHS_DATA['courses'] if c['progress'] == 0]
    
    return render_template('learning_paths.html', 
                         username=session['user'],
                         enrolled_courses=enrolled_courses,
                         available_courses=available_courses,
                         all_courses=LEARNING_PATHS_DATA['courses'],
                         categories=LEARNING_PATHS_DATA['categories'])

@app.route('/enroll-course/<int:course_id>')
def enroll_course(course_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find course
    course = next((c for c in LEARNING_PATHS_DATA['courses'] if c['id'] == course_id), None)
    
    if course:
        # In real app, save enrollment to database
        flash(f'Successfully enrolled in "{course["title"]}"!', 'success')
        
        # Update course progress
        for c in LEARNING_PATHS_DATA['courses']:
            if c['id'] == course_id:
                c['progress'] = 5  # Start with 5% progress
                c['modules_completed'] = 1
                c['last_accessed'] = 'Just now'
                break
    else:
        flash('Course not found!', 'error')
    
    return redirect(url_for('learning_paths'))

@app.route('/course-content/<int:course_id>')
def course_content(course_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find course
    course = next((c for c in LEARNING_PATHS_DATA['courses'] if c['id'] == course_id), None)
    
    if not course:
        flash('Course not found!', 'error')
        return redirect(url_for('learning_paths'))
    
    # Mock course content
    course_content_data = {
        'course_id': course_id,
        'title': course['title'],
        'description': course['description'],
        'instructor': course.get('instructor', 'Unknown'),
        'difficulty': course.get('difficulty', 'Intermediate'),
        'progress': course['progress'],
        'modules': [
            {'id': 1, 'title': 'Introduction to Course', 'duration': '1 hour', 'completed': course['progress'] > 0},
            {'id': 2, 'title': 'Getting Started', 'duration': '2 hours', 'completed': course['progress'] > 10},
            {'id': 3, 'title': 'Core Concepts', 'duration': '3 hours', 'completed': course['progress'] > 20},
            {'id': 4, 'title': 'Hands-on Practice', 'duration': '4 hours', 'completed': course['progress'] > 40},
            {'id': 5, 'title': 'Advanced Topics', 'duration': '5 hours', 'completed': course['progress'] > 60},
            {'id': 6, 'title': 'Final Project', 'duration': '6 hours', 'completed': course['progress'] > 80},
        ],
        'quizzes': [
            {'id': 1, 'title': 'Module 1 Quiz', 'score': 85 if course['progress'] > 10 else None, 'max_score': 100, 'completed': course['progress'] > 10},
            {'id': 2, 'title': 'Module 2 Quiz', 'score': 78 if course['progress'] > 20 else None, 'max_score': 100, 'completed': course['progress'] > 20},
        ],
        'assignments': [
            {'id': 1, 'title': 'Assignment 1', 'due_date': '2024-12-15', 'submitted': course['progress'] > 20},
            {'id': 2, 'title': 'Assignment 2', 'due_date': '2024-12-30', 'submitted': course['progress'] > 50},
        ],
        'resources': [
            {'id': 1, 'title': 'Course Slides', 'type': 'PDF', 'size': '2.5 MB'},
            {'id': 2, 'title': 'Reference Materials', 'type': 'Link', 'size': 'Online'},
            {'id': 3, 'title': 'Code Examples', 'type': 'GitHub', 'size': 'Repository'},
        ]
    }
    
    return render_template('course_content.html', 
                         username=session['user'],
                         course=course_content_data)

# ==================== MOCK INTERVIEWS ROUTES ====================
@app.route('/mock-interviews')
def mock_interviews():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Filter interviews
    upcoming_interviews = [i for i in MOCK_INTERVIEWS_DATA['interviews'] if not i['completed']]
    past_interviews = [i for i in MOCK_INTERVIEWS_DATA['interviews'] if i['completed']]
    
    return render_template('mock_interviews.html',
                         username=session['user'],
                         upcoming_interviews=upcoming_interviews,
                         past_interviews=past_interviews,
                         all_interviews=MOCK_INTERVIEWS_DATA['interviews'])

@app.route('/start-interview/<int:interview_id>')
def start_interview(interview_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find interview
    interview = next((i for i in MOCK_INTERVIEWS_DATA['interviews'] if i['id'] == interview_id), None)
    
    if interview:
        # In real app, start interview session
        flash(f'Starting "{interview["title"]}" interview. Good luck!', 'info')
        # Redirect to interview interface
        return redirect(url_for('interview_session', interview_id=interview_id))
    else:
        flash('Interview not found!', 'error')
        return redirect(url_for('mock_interviews'))

@app.route('/interview-session/<int:interview_id>')
def interview_session(interview_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Mock interview questions
    interview_questions = {
        'interview_id': interview_id,
        'title': 'Software Engineer - FAANG',
        'company': 'Google',
        'duration': '45 mins',
        'questions': [
            {
                'id': 1,
                'text': 'Tell me about yourself and your experience.',
                'type': 'Behavioral',
                'time_limit': '3 minutes',
                'hint': 'Focus on relevant experience and achievements'
            },
            {
                'id': 2,
                'text': 'Explain the time complexity of binary search.',
                'type': 'Technical',
                'time_limit': '5 minutes',
                'hint': 'Consider worst-case scenario'
            },
            {
                'id': 3,
                'text': 'Design a URL shortening service like bit.ly',
                'type': 'System Design',
                'time_limit': '15 minutes',
                'hint': 'Consider scalability, databases, and caching'
            },
            {
                'id': 4,
                'text': 'How would you handle a conflict with a team member?',
                'type': 'Behavioral',
                'time_limit': '3 minutes',
                'hint': 'Show communication and problem-solving skills'
            }
        ],
        'current_question': 1,
        'total_questions': 4,
        'time_remaining': 2700  # 45 minutes in seconds
    }
    
    return render_template('interview_session.html',
                         username=session['user'],
                         interview=interview_questions)

@app.route('/submit-interview/<int:interview_id>', methods=['POST'])
def submit_interview(interview_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # In real app, save interview responses
    flash('Interview submitted successfully! Results will be available shortly.', 'success')
    
    # Update interview status
    for interview in MOCK_INTERVIEWS_DATA['interviews']:
        if interview['id'] == interview_id:
            interview['completed'] = True
            interview['score'] = 82  # Mock score
            interview['date'] = datetime.now().strftime('%Y-%m-%d')
            break
    
    return redirect(url_for('interview_results', interview_id=interview_id))

@app.route('/interview-results/<int:interview_id>')
def interview_results(interview_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find interview
    interview = next((i for i in MOCK_INTERVIEWS_DATA['interviews'] if i['id'] == interview_id), None)
    
    if not interview or not interview['completed']:
        flash('Interview results not available!', 'error')
        return redirect(url_for('mock_interviews'))
    
    # Mock interview results
    results = {
        'interview_id': interview_id,
        'title': interview['title'],
        'company': interview['company'],
        'date': interview['date'],
        'overall_score': interview['score'],
        'time_taken': '38 minutes',
        'details': {
            'technical_skills': 85,
            'problem_solving': 80,
            'communication': 78,
            'culture_fit': 82
        },
        'feedback': [
            'Excellent problem-solving approach',
            'Good communication during technical explanations',
            'Consider practicing more system design questions',
            'Work on time management during coding challenges'
        ],
        'questions': [
            {'question': 'Tell me about yourself', 'your_answer': 'Good structure', 'score': 9},
            {'question': 'Binary search complexity', 'your_answer': 'Correct: O(log n)', 'score': 10},
            {'question': 'URL shortener design', 'your_answer': 'Good approach, missing cache details', 'score': 8},
            {'question': 'Conflict resolution', 'your_answer': 'Well explained with example', 'score': 9}
        ]
    }
    
    return render_template('interview_results.html',
                         username=session['user'],
                         results=results)

# ==================== CERTIFICATIONS ROUTES ====================
@app.route('/certifications')
def certifications():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Filter certifications
    enrolled_certs = [c for c in CERTIFICATIONS_DATA['certifications'] if c['enrolled']]
    available_certs = [c for c in CERTIFICATIONS_DATA['certifications'] if not c['enrolled']]
    
    return render_template('certifications.html',
                         username=session['user'],
                         enrolled_certifications=enrolled_certs,
                         available_certifications=available_certs,
                         all_certifications=CERTIFICATIONS_DATA['certifications'])

@app.route('/enroll-certification/<int:cert_id>')
def enroll_certification(cert_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find certification
    cert = next((c for c in CERTIFICATIONS_DATA['certifications'] if c['id'] == cert_id), None)
    
    if cert:
        # In real app, save enrollment to database
        flash(f'Enrolled in "{cert["title"]}" certification program!', 'success')
        
        # Update enrollment status
        for c in CERTIFICATIONS_DATA['certifications']:
            if c['id'] == cert_id:
                c['enrolled'] = True
                c['progress'] = 5  # Start with 5% progress
                break
    else:
        flash('Certification not found!', 'error')
    
    return redirect(url_for('certifications'))

@app.route('/certification-content/<int:cert_id>')
def certification_content(cert_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find certification
    cert = next((c for c in CERTIFICATIONS_DATA['certifications'] if c['id'] == cert_id), None)
    
    if not cert:
        flash('Certification not found!', 'error')
        return redirect(url_for('certifications'))
    
    # Mock certification content
    cert_content = {
        'cert_id': cert_id,
        'title': cert['title'],
        'provider': cert['provider'],
        'level': cert['level'],
        'validity': cert['validity'],
        'exam_fee': cert['exam_fee'],
        'progress': cert['progress'],
        'exam_date': cert['exam_date'],
        'syllabus': [
            {'id': 1, 'title': 'Domain 1: Fundamentals', 'completed': cert['progress'] > 0},
            {'id': 2, 'title': 'Domain 2: Core Services', 'completed': cert['progress'] > 20},
            {'id': 3, 'title': 'Domain 3: Security & Compliance', 'completed': cert['progress'] > 40},
            {'id': 4, 'title': 'Domain 4: Architecture', 'completed': cert['progress'] > 60},
            {'id': 5, 'title': 'Practice Exams', 'completed': cert['progress'] > 80},
        ],
        'practice_exams': [
            {'id': 1, 'title': 'Practice Test 1', 'score': 65 if cert['progress'] > 30 else None, 'date': '2024-11-10'},
            {'id': 2, 'title': 'Practice Test 2', 'score': 72 if cert['progress'] > 50 else None, 'date': '2024-11-15'},
        ],
        'resources': [
            {'id': 1, 'title': 'Official Study Guide', 'type': 'PDF', 'size': '5.2 MB'},
            {'id': 2, 'title': 'Video Tutorials', 'type': 'Video', 'size': '12 hours'},
            {'id': 3, 'title': 'Practice Questions', 'type': 'Quiz', 'size': '200 questions'},
        ]
    }
    
    return render_template('certification_content.html',
                         username=session['user'],
                         certification=cert_content)

# ==================== PROFILE & SETTINGS ROUTES ====================
@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Mock user profile data
    user_profile = {
        'username': session['user'],
        'email': f'{session["user"]}@example.com',
        'full_name': 'John Doe',
        'title': 'Software Engineer',
        'location': 'San Francisco, CA',
        'bio': 'Passionate developer looking to enhance skills and career.',
        'skills': ['Python', 'JavaScript', 'React', 'Node.js', 'AWS'],
        'experience': [
            {'company': 'Tech Corp', 'role': 'Senior Developer', 'duration': '2020-Present'},
            {'company': 'Startup Inc', 'role': 'Full Stack Developer', 'duration': '2018-2020'}
        ],
        'education': [
            {'degree': 'B.S. Computer Science', 'school': 'Stanford University', 'year': '2018'}
        ],
        'joined': '2024-01-15',
        'last_login': datetime.now().strftime('%Y-%m-%d')
    }
    
    return render_template('profile.html', 
                         username=session['user'],
                         profile=user_profile)

@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Update profile logic here
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', username=session['user'])

@app.route('/settings')
def settings():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return render_template('settings.html', username=session['user'])

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

# ==================== API ENDPOINTS ====================
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    resume_text = data.get('resume_text', '')
    job_description = data.get('job_description', '')
    
    if not resume_text or not job_description:
        return jsonify({'error': 'Missing resume text or job description'}), 400
    
    # Calculate scores
    ats_score, ats_factors, ats_breakdown, found_keywords = calculate_ats_score(resume_text, job_description)
    similarity_score = calculate_similarity_score(resume_text, job_description)
    
    return jsonify({
        'ats_score': ats_score,
        'similarity_score': similarity_score,
        'found_keywords': found_keywords,
        'factors': ats_factors
    })

# ==================== MAIN APPLICATION RUN ====================
if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    print("=" * 60)
    print("🤖 AI Resume Analyzer + Career Platform")
    print("=" * 60)
    print(f"📁 Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"📄 Templates folder: {os.path.abspath('templates')}")
    print("🚀 Available Features:")
    print("   • Resume Analysis & Scoring")
    print("   • Learning Paths & Courses")
    print("   • Mock Interviews")
    print("   • Certification Programs")
    print("   • User Profile & Dashboard")
    print(f"👤 Default user: demo (use any credentials)")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)