from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify  # ADDED jsonify

import os
from werkzeug.security import generate_password_hash, check_password_hash
import PyPDF2
from db import get_connection
import docx
from resume_analyzer import ResumeAnalyzer
import traceback  

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a strong secret key
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
resume_analyzer = ResumeAnalyzer()  

# Database initialization - KEEP ONLY THIS ONE
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['education'] = user['education']
            session['role'] = user['role']
            session['skills'] = user['skills']
            session['experience'] = user['experience']
            session['location'] = user['location']
            session['profile_pic'] = user['profile_pic']
            session['progress'] = user['progress']

            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')



# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============ ORIGINAL ROUTES ============

@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'email' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Get user data from session
    return render_template('dashboard.html',
        name=session.get('name', 'Guest'),
        email=session.get('email', ''),
        education=session.get('education', 'Not specified'),
        role=session.get('role', 'student'),
        skills=session.get('skills', ''),
        experience=session.get('experience', ''),
        location=session.get('location', ''),
        profile_pic=session.get('profile_pic', ''),
        progress=session.get('progress', 0)
    )

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        education = request.form['education']
        role = request.form['role']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # 🔹 Password validation
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return render_template('signup.html')

        # 🔹 Hash password
        hashed_password = generate_password_hash(password)

        # 🔹 👉 THIS IS WHERE YOUR CODE GOES 👇
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users (name, email, education, role, password, progress)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (name, email, education, role, hashed_password, 10))

            conn.commit()
            conn.close()

            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash('Email already exists or database error', 'error')
            return render_template('signup.html')

    # 🔹 GET request
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

# ============ PROFILE EDIT ROUTE ============

@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        education = request.form.get('education')
        role = request.form.get('role')
        skills = request.form.get('skills', '')
        experience = request.form.get('experience', '')
        location = request.form.get('location', '')
        email = session['email']  # Keep original email
        
        # Handle profile picture upload
        profile_pic = session.get('profile_pic', '')
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '':
                # Create uploads directory if it doesn't exist
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                # Generate unique filename
                filename = f"{session['user_id']}_{file.filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                profile_pic = filename
        
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE users
            SET name=%s, education=%s, role=%s,
            skills=%s, experience=%s, location=%s,
            profile_pic=%s
            WHERE email=%s
            """, (name, education, role, skills, experience, location, profile_pic, email))

        conn.commit()
        conn.close()

        # Update session data
        session['name'] = name
        session['education'] = education
        session['role'] = role
        session['skills'] = skills
        session['experience'] = experience
        session['location'] = location
        session['profile_pic'] = profile_pic

        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard'))

            
        
    # GET request - populate form with current data
    return render_template('edit_profile.html',
        name=session.get('name', ''),
        email=session.get('email', ''),
        education=session.get('education', ''),
        role=session.get('role', 'student'),
        skills=session.get('skills', ''),
        experience=session.get('experience', ''),
        location=session.get('location', ''),
        profile_pic=session.get('profile_pic', '')
    )

# ============ HELPER ROUTES ============

@app.route('/test-resume', methods=['GET'])
def test_resume():
    return jsonify({
        "status": "ok",
        "message": "Resume analyzer routes are accessible"
    })

@app.route('/profile/view')
def view_profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    # Fetch fresh data from database
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email=%s", (session['email'],))
    user = cursor.fetchone()

    conn.close()

    
    if user:
        return render_template(
        'view_profile.html',
        name=user['name'],
        email=user['email'],
        education=user['education'],
        role=user['role'],
        skills=user['skills'],
        experience=user['experience'],
        location=user['location'],
        profile_pic=user['profile_pic'],
        progress=user['progress']
    )

    
    flash('Profile not found', 'error')
    return redirect(url_for('dashboard'))

# ============ RESUME ANALYSIS ROUTES ============

@app.route('/resume/analyze', methods=['GET', 'POST'])
def analyze_resume():
    """Route for analyzing resume - KEEP ONLY THIS ONE"""
    print("DEBUG: analyze_resume route called")  # ADD THIS
    
    if 'email' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        print("DEBUG: POST request received")  # ADD THIS
        try:
            resume_text = ""
            job_title = request.form.get('job_title', '')
            print(f"DEBUG: Job title from form: {job_title}")  # ADD THIS
            
            # Check if file uploaded
            if 'resume_file' in request.files:
                file = request.files['resume_file']
                print(f"DEBUG: File uploaded: {file.filename}")  # ADD THIS
                if file and file.filename:
                    # Read PDF
                    if file.filename.endswith('.pdf'):
                        print("DEBUG: Processing PDF file")  # ADD THIS
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            resume_text += page.extract_text()
                    # Read DOCX
                    elif file.filename.endswith('.docx'):
                        print("DEBUG: Processing DOCX file")  # ADD THIS
                        doc = docx.Document(file)
                        for para in doc.paragraphs:
                            resume_text += para.text + '\n'
                    # Read TXT
                    elif file.filename.endswith('.txt'):
                        print("DEBUG: Processing TXT file")  # ADD THIS
                        resume_text = file.read().decode('utf-8')
                    else:
                        flash('Unsupported file format. Please use PDF, DOCX, or TXT.', 'error')
                        return redirect(url_for('analyze_resume'))
            
            # Check if text pasted
            elif 'resume_text' in request.form:
                resume_text = request.form['resume_text']
                print(f"DEBUG: Text pasted, length: {len(resume_text)}")  # ADD THIS
            
            if not resume_text.strip():
                flash('Please upload a file or paste resume text', 'error')
                return redirect(url_for('analyze_resume'))
            
            print(f"DEBUG: Resume text length before analysis: {len(resume_text)}")  # ADD THIS
            
            # Analyze with AI
            print("DEBUG: Calling resume_analyzer.analyze()")  # ADD THIS
            analysis = resume_analyzer.analyze(resume_text, job_title)
            print(f"DEBUG: Analysis completed: {analysis}")  # ADD THIS
            
            # Save to database
            
            # Store in session for results page
            session['last_analysis'] = analysis
            
            flash('Resume analyzed successfully!', 'success')
            return redirect(url_for('resume_results'))
            
        except Exception as e:
            print(f"ERROR in analyze_resume: {str(e)}")  # ADD THIS
            import traceback
            traceback.print_exc()  # ADD THIS to see full error
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('analyze_resume'))
    
    return render_template('analyze_resume.html')
@app.route('/test-analyzer')
def test_analyzer():
    """Test if analyzer works"""
    try:
        test_text = """
        John Doe
        Software Engineer
        Email: john@example.com
        Phone: 123-456-7890
        
        EDUCATION
        Bachelor of Computer Science, University of Technology, 2020
        
        EXPERIENCE
        Software Developer at Tech Corp (2021-Present)
        - Developed web applications using Python and JavaScript
        - Improved system performance by 30%
        - Managed a team of 3 developers
        
        SKILLS
        Python, JavaScript, SQL, Git, AWS
        """
        
        result = resume_analyzer.analyze(test_text, "Software Engineer")
        return jsonify({
            "status": "success",
            "analyzer_working": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "analyzer_working": False,
            "error": str(e)
        })

@app.route('/resume/results')
def resume_results():
    """Show analysis results"""
    if 'email' not in session:
        return redirect(url_for('login'))
    
    if 'last_analysis' not in session:
        flash('Please analyze a resume first', 'error')
        return redirect(url_for('analyze_resume'))
    
    return render_template('resume_results.html', 
                         analysis=session['last_analysis'])

@app.route('/resume/history')
def resume_history():
    if 'email' not in session:
        return redirect(url_for('login'))

    return render_template('resume_history.html', history=[])

# ============ DEBUG ROUTES ============

@app.route('/debug-info')
def debug_info():
    """Debug information page"""
    import sys
    import os
    
    info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
        "resume_analyzer_exists": os.path.exists('resume_analyzer.py'),
        "templates_exists": os.path.exists('templates'),
        "app_running": True
    }
    
    # Try to import resume_analyzer
    try:
        from resume_analyzer import ResumeAnalyzer
        info["resume_analyzer_import"] = "SUCCESS"
        info["analyzer_class"] = str(ResumeAnalyzer)
    except Exception as e:
        info["resume_analyzer_import"] = f"FAILED: {str(e)}"
    
    return jsonify(info)

@app.route('/test-direct')
def test_direct():
    """Direct test of the analyzer"""
    test_text = "John Doe\nSoftware Engineer\nPython, Java, SQL"
    
    try:
        result = resume_analyzer.analyze(test_text)
        return jsonify({
            "status": "success",
            "message": "Analyzer is working!",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
@app.route('/learning-paths')
def learning_paths():
    if 'email' not in session:
        return redirect(url_for('login'))

    courses = [
        {
            "id": 1,
            "title": "Java Full Stack Developer",
            "category": "Web Development",
            "duration": "4 Months",
            "level": "Beginner → Advanced",
            "rating": 4.8
        },
        {
            "id": 2,
            "title": "Data Science with Python",
            "category": "Data Science",
            "duration": "3 Months",
            "level": "Beginner",
            "rating": 4.7
        },
        {
            "id": 3,
            "title": "Cloud & DevOps",
            "category": "Cloud",
            "duration": "2.5 Months",
            "level": "Intermediate",
            "rating": 4.6
        },
        {
            "id": 4,
            "title": "DSA Mastery",
            "category": "Data Structures",
            "duration": "2 Months",
            "level": "Beginner → Intermediate",
            "rating": 4.9
        }
    ]

    return render_template('learning_paths.html', courses=courses)

@app.route('/course/<int:course_id>')
def course_details(course_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    courses = {
        1: {
            "title": "Java Full Stack Developer",
            "description": "Learn Core Java, Spring Boot, MySQL, REST APIs, and React with real projects.",
            "start_date": "15 Feb 2026",
            "time": "7:00 PM – 9:00 PM",
            "duration": "4 Months"
        },
        2: {
            "title": "Data Science with Python",
            "description": "Python, Pandas, NumPy, Machine Learning, and hands-on projects.",
            "start_date": "20 Feb 2026",
            "time": "6:00 PM – 8:00 PM",
            "duration": "3 Months"
        },
        3: {
            "title": "Cloud & DevOps",
            "description": "AWS, Docker, Kubernetes, CI/CD pipelines.",
            "start_date": "25 Feb 2026",
            "time": "8:00 PM – 9:30 PM",
            "duration": "2.5 Months"
        },
        4: {
            "title": "DSA Mastery",
            "description": "Data Structures and Algorithms from basics to advanced.",
            "start_date": "10 Feb 2026",
            "time": "5:00 PM – 6:30 PM",
            "duration": "2 Months"
        }
    }

    course = courses.get(course_id)
    if not course:
        flash("Course not found", "error")
        return redirect(url_for('learning_paths'))

    return render_template('course_details.html', course=course)

@app.route('/mock-interviews')
def mock_interviews():
    if 'email' not in session:
        return redirect(url_for('login'))

    roles = ["Java Developer", "Frontend Developer", "Data Analyst"]
    return render_template('mock_interviews.html', roles=roles)

@app.route('/certifications')
def certifications():
    if 'email' not in session:
        return redirect(url_for('login'))

    courses = [
        {"name": "Google Data Analytics", "platform": "Coursera"},
        {"name": "Java Spring Boot", "platform": "Udemy"},
        {"name": "AWS Cloud Practitioner", "platform": "AWS"}
    ]

    return render_template('certifications.html', courses=courses)

@app.route('/mentors')
def mentors():
    if 'email' not in session:
        return redirect(url_for('login'))

    mentors = [
        {"name": "Rahul Sharma", "expertise": "Java", "experience": "8 yrs"},
        {"name": "Anita Verma", "expertise": "Data Science", "experience": "6 yrs"}
    ]

    return render_template('mentors.html', mentors=mentors)

# ============ END OF ROUTES ============

# ============ END OF ROUTES ============

if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    print("=" * 50)
    print("Starting Career Connect Application")
    print("=" * 50)
    
    # Test the analyzer on startup
    try:
        test_result = resume_analyzer.analyze("Test resume")
        print(f"✓ Resume analyzer test passed. Overall score: {test_result['scores']['overall']}")
    except Exception as e:
        print(f"✗ Resume analyzer test failed: {e}")
    
    app.run(debug=True)