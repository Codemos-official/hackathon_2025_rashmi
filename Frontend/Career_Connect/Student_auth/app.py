from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a strong secret key

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            education TEXT,
            role TEXT,
            password TEXT NOT NULL,
            skills TEXT,
            experience TEXT,
            location TEXT,
            profile_pic TEXT,
            progress INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user:
            # Check password
            if check_password_hash(user[5], password):  # password is at index 5
                # Set session variables
                session['user_id'] = user[0]
                session['name'] = user[1]
                session['email'] = user[2]
                session['education'] = user[3] or ''
                session['role'] = user[4] or 'student'
                session['skills'] = user[6] or ''
                session['experience'] = user[7] or ''
                session['location'] = user[8] or ''
                session['profile_pic'] = user[9] or ''
                session['progress'] = user[10] or 0
                
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password', 'error')
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

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
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return render_template('signup.html')
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO users (name, email, education, role, password, progress)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, email, education, role, hashed_password, 10))
            conn.commit()
            conn.close()
            
            # Auto-login after signup
            session['name'] = name
            session['email'] = email
            session['education'] = education
            session['role'] = role
            session['progress'] = 10
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except sqlite3.IntegrityError:
            flash('Email already exists', 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Update profile logic here
        pass
    
    return render_template('edit_profile.html')

if __name__ == '__main__':
    app.run(debug=True)