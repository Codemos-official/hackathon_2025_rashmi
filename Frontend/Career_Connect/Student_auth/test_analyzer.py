from resume_analyzer import ResumeAnalyzer

analyzer = ResumeAnalyzer()

test_resume = """
John Doe - Software Engineer
Skills: Python, JavaScript, SQL
Experience: 5 years as developer
Education: Bachelor's in Computer Science
"""

result = analyzer.analyze_resume(test_resume)
print("Test successful!")
print(f"Overall score: {result['scores']['overall']}")