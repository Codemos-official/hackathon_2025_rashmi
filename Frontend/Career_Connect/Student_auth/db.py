import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'career_connect'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '1234')
        )
        
        if connection.is_connected():
            return connection
            
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def create_tables():
    """Create all necessary tables if they don't exist"""
    conn = get_connection()
    if conn is None:
        return False
    
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            education VARCHAR(200),
            role VARCHAR(50),
            password VARCHAR(255) NOT NULL,
            skills TEXT,
            experience TEXT,
            location VARCHAR(100),
            profile_pic VARCHAR(255),
            progress INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create courses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            rating DECIMAL(3,1),
            reviews VARCHAR(50),
            duration VARCHAR(50),
            enrolled VARCHAR(50),
            price DECIMAL(10,2),
            category VARCHAR(100),
            difficulty VARCHAR(50),
            skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create user_progress table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            course_id INT NOT NULL,
            progress INT DEFAULT 0,
            modules_completed INT DEFAULT 0,
            total_modules INT DEFAULT 10,
            last_accessed DATE,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            UNIQUE KEY unique_user_course (user_id, course_id)
        )
    """)
    
    # Create resume_analysis table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            job_title VARCHAR(255),
            resume_text TEXT,
            overall_score DECIMAL(5,2),
            ats_score DECIMAL(5,2),
            skill_score DECIMAL(5,2),
            experience_score DECIMAL(5,2),
            education_score DECIMAL(5,2),
            match_score DECIMAL(5,2),
            strengths TEXT,
            weaknesses TEXT,
            recommendations TEXT,
            skills_to_learn TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return True

# Initialize tables when module is imported
if __name__ != "__main__":
    create_tables()