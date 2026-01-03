import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textstat
from collections import Counter

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Industry-specific keywords
        self.keyword_categories = {
            'technical': [
                'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node',
                'aws', 'docker', 'kubernetes', 'machine learning', 'ai', 'data analysis',
                'git', 'github', 'rest api', 'mongodb', 'mysql', 'postgresql'
            ],
            'soft_skills': [
                'communication', 'leadership', 'teamwork', 'problem solving',
                'time management', 'creativity', 'adaptability', 'critical thinking'
            ],
            'education': [
                'bachelor', 'master', 'phd', 'degree', 'certification', 'course',
                'training', 'workshop', 'seminar'
            ],
            'experience': [
                'experience', 'worked', 'responsible', 'managed', 'developed',
                'implemented', 'achieved', 'increased', 'reduced', 'created'
            ]
        }
    
    def analyze_resume(self, resume_text, job_title=None):
        """Analyze resume text and return scores"""
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(resume_text)
        
        # Calculate scores
        scores = {
            'overall': self._calculate_overall_score(cleaned_text),
            'ats': self._calculate_ats_score(cleaned_text),
            'content': self._calculate_content_score(cleaned_text),
            'format': self._calculate_format_score(cleaned_text),
            'keyword': self._calculate_keyword_score(cleaned_text, job_title)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(cleaned_text, scores)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(cleaned_text)
        weaknesses = self._identify_weaknesses(cleaned_text)
        
        return {
            'scores': scores,
            'recommendations': recommendations,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'keyword_analysis': self._analyze_keywords(cleaned_text),
            'readability_score': self._calculate_readability(cleaned_text)
        }
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _calculate_overall_score(self, text):
        """Calculate overall resume score (0-100)"""
        word_count = len(text.split())
        
        if word_count < 100:
            length_score = 40
        elif word_count < 300:
            length_score = 70
        elif word_count < 800:
            length_score = 90
        else:
            length_score = 60
        
        sections = self._identify_sections(text)
        section_score = min(len(sections) * 20, 100)
        
        contact_score = 100 if self._has_contact_info(text) else 60
        
        return int((length_score + section_score + contact_score) / 3)
    
    def _calculate_ats_score(self, text):
        """Calculate ATS compatibility score"""
        score = 70
        
        if self._has_contact_info(text):
            score += 10
        
        sections = self._identify_sections(text)
        if len(sections) >= 3:
            score += 10
        
        keywords_found = self._count_keywords(text)
        if keywords_found > 5:
            score += 10
        
        if not self._has_complex_formatting(text):
            score += 10
        
        return min(score, 100)
    
    def _calculate_content_score(self, text):
        """Calculate content quality score"""
        score = 60
        
        action_verbs = ['managed', 'developed', 'created', 'implemented', 'led',
                       'increased', 'reduced', 'improved', 'achieved', 'built']
        verb_count = sum(1 for verb in action_verbs if verb in text)
        score += min(verb_count * 5, 30)
        
        quant_pattern = r'\d+%|\$\d+|\d+\s*(?:years?|months?)'
        quant_matches = len(re.findall(quant_pattern, text))
        score += min(quant_matches * 5, 20)
        
        return min(score, 100)
    
    def _calculate_format_score(self, text):
        """Calculate formatting score"""
        score = 70
        
        bullet_count = text.count('•') + text.count('*') + text.count('-')
        if bullet_count > 5:
            score += 15
        
        section_headers = len(re.findall(r'\b(?:education|experience|skills|projects|summary)\b', text))
        if section_headers >= 3:
            score += 15
        
        return min(score, 100)
    
    def _calculate_keyword_score(self, text, target_job=None):
        """Calculate keyword relevance score"""
        if not target_job:
            keywords_found = self._count_keywords(text)
            return min(keywords_found * 5, 100)
        
        job_keywords = self._get_job_keywords(target_job)
        matched_keywords = [kw for kw in job_keywords if kw in text]
        
        if job_keywords:
            match_percentage = (len(matched_keywords) / len(job_keywords)) * 100
            return int(min(match_percentage, 100))
        
        return 50
    
    def _generate_recommendations(self, text, scores):
        """Generate personalized recommendations"""
        recommendations = []
        
        if scores['ats'] < 70:
            recommendations.append("Improve ATS compatibility by using standard section headers")
        
        if scores['content'] < 70:
            recommendations.append("Add more quantifiable achievements and action verbs")
        
        if scores['format'] < 70:
            recommendations.append("Use bullet points and clear section organization")
        
        if scores['keyword'] < 70:
            recommendations.append("Include more relevant keywords for your target industry")
        
        if len(text.split()) < 200:
            recommendations.append("Consider adding more details to your resume")
        
        if len(text.split()) > 800:
            recommendations.append("Consider making your resume more concise")
        
        if not self._has_contact_info(text):
            recommendations.append("Add contact information (email, phone)")
        
        if not self._has_education_section(text):
            recommendations.append("Add an education section")
        
        if not self._has_experience_section(text):
            recommendations.append("Add work experience details")
        
        return recommendations
    
    def _identify_strengths(self, text):
        """Identify resume strengths"""
        strengths = []
        
        if len(text.split()) >= 300:
            strengths.append("Good length and detail")
        
        if self._count_keywords(text) >= 10:
            strengths.append("Strong keyword usage")
        
        if self._has_quantifiable_achievements(text):
            strengths.append("Includes quantifiable achievements")
        
        if self._has_action_verbs(text):
            strengths.append("Uses action-oriented language")
        
        sections = self._identify_sections(text)
        if len(sections) >= 4:
            strengths.append("Well-organized with clear sections")
        
        return strengths
    
    def _identify_weaknesses(self, text):
        """Identify resume weaknesses"""
        weaknesses = []
        
        if len(text.split()) < 200:
            weaknesses.append("Too brief - needs more detail")
        
        if len(text.split()) > 800:
            weaknesses.append("Too long - consider being more concise")
        
        if not self._has_contact_info(text):
            weaknesses.append("Missing contact information")
        
        if not self._has_quantifiable_achievements(text):
            weaknesses.append("Lacks quantifiable achievements")
        
        if self._count_keywords(text) < 5:
            weaknesses.append("Could use more industry-specific keywords")
        
        return weaknesses
    
    # Helper methods
    def _identify_sections(self, text):
        sections = []
        common_sections = ['summary', 'experience', 'education', 'skills', 
                          'projects', 'certifications', 'awards']
        
        for section in common_sections:
            if section in text:
                sections.append(section)
        
        return sections
    
    def _has_contact_info(self, text):
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _has_complex_formatting(self, text):
        complex_patterns = [r'\|\|', r'---+', r'===+']
        for pattern in complex_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _count_keywords(self, text):
        count = 0
        for category, keywords in self.keyword_categories.items():
            for keyword in keywords:
                if keyword in text:
                    count += 1
        return count
    
    def _has_quantifiable_achievements(self, text):
        patterns = [
            r'\d+%',
            r'\$\d+',
            r'\d+\s*(?:people|team members|clients)',
            r'increased by',
            r'reduced by',
            r'saved\s+\$?\d+'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _has_action_verbs(self, text):
        action_verbs = ['managed', 'developed', 'created', 'implemented', 'led',
                       'increased', 'reduced', 'improved', 'achieved', 'built',
                       'designed', 'launched', 'optimized', 'solved', 'trained']
        
        for verb in action_verbs:
            if verb in text:
                return True
        return False
    
    def _has_education_section(self, text):
        education_indicators = ['education', 'university', 'college', 'degree',
                               'bachelor', 'master', 'phd', 'graduated']
        
        for indicator in education_indicators:
            if indicator in text:
                return True
        return False
    
    def _has_experience_section(self, text):
        experience_indicators = ['experience', 'worked', 'employed', 'job',
                                'position', 'role', 'responsibilities']
        
        for indicator in experience_indicators:
            if indicator in text:
                return True
        return False
    
    def _analyze_keywords(self, text):
        """Analyze keyword distribution"""
        keyword_counts = {}
        
        for category, keywords in self.keyword_categories.items():
            count = 0
            for keyword in keywords:
                if keyword in text:
                    count += 1
            keyword_counts[category] = count
        
        return keyword_counts
    
    def _calculate_readability(self, text):
        """Calculate readability scores"""
        try:
            flesch = textstat.flesch_reading_ease(text)
            
            if flesch > 60:
                return 80
            elif flesch > 50:
                return 70
            elif flesch > 30:
                return 60
            else:
                return 40
        except:
            return 60
    
    def _get_job_keywords(self, job_title):
        """Get relevant keywords for a job title"""
        job_keyword_map = {
            'software engineer': ['python', 'java', 'javascript', 'sql', 'git', 'aws'],
            'data scientist': ['python', 'machine learning', 'sql', 'statistics', 'pandas'],
            'web developer': ['html', 'css', 'javascript', 'react', 'node', 'api'],
            'project manager': ['project management', 'agile', 'scrum', 'leadership'],
            'marketing': ['marketing', 'seo', 'social media', 'campaign', 'analytics']
        }
        
        job_title_lower = job_title.lower()
        for job, keywords in job_keyword_map.items():
            if job in job_title_lower:
                return keywords
        
        return []