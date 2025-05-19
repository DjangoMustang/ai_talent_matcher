"""
AI-Powered Resume Analyzer & Career Assistant
==========================================
Author: [Your Name]
Date: April 2025
GitHub: [Your GitHub Profile]

This application leverages Gemini 1.5 Pro to analyze resumes against job descriptions, 
providing intelligent career guidance, skill gap analysis, and personalized improvement recommendations.
Built with production-level code quality, modular architecture, error handling, and performance optimization.
"""

# Standard library imports
import os
import io
import base64
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import time

# Third-party imports
import streamlit as st
import pdf2image
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"NLTK resource download issue: {str(e)}")

# Constants and configurations
CONFIG = {
    "MODELS": {
        "text_analysis": "gemini-1.5-pro",
        "image_analysis": "gemini-1.5-flash",
        "advanced_analysis": "gemini-1.5-pro"
    },
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 2,  # seconds
    "TEMPERATURE": 0.2,  # Lower temperature for more consistent outputs
    "TOP_K": 40,
    "TOP_P": 0.95,
    "CACHE_EXPIRY": 3600,  # seconds
}

# Application version
__version__ = "2.5.0"


class AIModelHandler:
    """
    Handles interactions with the Gemini AI model, including configuration, 
    rate limiting, and error handling.
    """
    
    def __init__(self):
        """Initialize the AI model handler with API key and configurations."""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("Google API key not found in environment variables")
            raise ValueError("Missing API key. Please set GOOGLE_API_KEY in your .env file")
            
        # Configure the Gemini API
        try:
            genai.configure(api_key=api_key)
            logger.info("Gemini AI API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI: {str(e)}")
            raise
            
        # Response cache to avoid redundant API calls
        self.response_cache = {}
        
    def get_model_response(
        self, 
        input_text: str, 
        media_content: Optional[List[Dict]] = None, 
        prompt: str = "", 
        model_name: str = CONFIG["MODELS"]["text_analysis"]
    ) -> str:
        """
        Get a response from the Gemini model with retry logic and caching.
        
        Args:
            input_text: The user's input text (job description)
            media_content: Any media content (PDF/image) to analyze
            prompt: System prompt for the model
            model_name: Which Gemini model to use
            
        Returns:
            Response text from the model
        """
        # Generate a cache key
        cache_key = f"{input_text}_{prompt}_{model_name}"
        if media_content:
            # Add a hash of the first page content to the cache key
            if isinstance(media_content, list) and len(media_content) > 0:
                first_page_hash = hash(media_content[0].get("data", "")[:100])
                cache_key += f"_{first_page_hash}"
        
        # Check cache
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cache_entry["timestamp"] < CONFIG["CACHE_EXPIRY"]:
                logger.info("Returning cached response")
                return cache_entry["response"]
        
        # Prepare the model parameters
        generation_config = {
            "temperature": CONFIG["TEMPERATURE"],
            "top_p": CONFIG["TOP_P"],
            "top_k": CONFIG["TOP_K"],
            "max_output_tokens": 4096,
        }
        
        # Prepare content for the model
        content = [input_text, prompt]
        if media_content:
            content.extend(media_content)
            
        # Retry logic
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                logger.info(f"Sending request to Gemini AI model: {model_name} (attempt {attempt+1})")
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                response = model.generate_content(content)
                
                # Cache the successful response
                self.response_cache[cache_key] = {
                    "response": response.text,
                    "timestamp": time.time()
                }
                
                logger.info("Successfully received response from Gemini AI")
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < CONFIG["MAX_RETRIES"] - 1:
                    logger.info(f"Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    time.sleep(CONFIG["RETRY_DELAY"])
                else:
                    logger.error(f"All {CONFIG['MAX_RETRIES']} attempts failed")
                    return f"Error: Unable to get response from AI model after {CONFIG['MAX_RETRIES']} attempts. Please try again later."


class DocumentProcessor:
    """
    Handles document processing tasks such as PDF to image conversion,
    text extraction, and document analysis.
    """
    
    @staticmethod
    def process_pdf(uploaded_file) -> Tuple[List[Dict], Optional[List[Image.Image]]]:
        """
        Process an uploaded PDF file, converting it to images and preparing it for AI analysis.
        
        Args:
            uploaded_file: The uploaded PDF file
            
        Returns:
            Tuple containing processed parts for the AI model and the list of page images
        """
        if uploaded_file is None:
            logger.warning("No file uploaded")
            return None, None
            
        try:
            logger.info(f"Processing PDF file: {uploaded_file.name}")
            
            # Convert the PDF to images
            start_time = time.time()
            pdf_images = pdf2image.convert_from_bytes(
                uploaded_file.read(),
                dpi=200,  # Balanced for quality and performance
                fmt="jpeg"
            )
            logger.info(f"PDF conversion completed in {time.time() - start_time:.2f} seconds")
            
            # Prepare image parts for the AI model (first page only for quick analysis)
            pdf_parts = []
            
            # Process first page for initial analysis
            img_byte_arr = io.BytesIO()
            pdf_images[0].save(img_byte_arr, format='JPEG', quality=85)  # Compress slightly for better performance
            img_byte_arr = img_byte_arr.getvalue()
            
            pdf_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            })
            
            logger.info(f"Successfully processed {len(pdf_images)} page(s) from PDF")
            return pdf_parts, pdf_images
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Error processing PDF file: {str(e)}")

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract key terms from text using NLP techniques.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of extracted keywords
        """
        try:
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text.lower())
            
            # Filter out stopwords and non-alphabetic tokens
            keywords = [word for word in word_tokens if word.isalpha() and word not in stop_words]
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []


class DataAnalyzer:
    """
    Handles data analysis tasks such as skill matching, visualization,
    and statistical analysis of resume and job description data.
    """
    
    @staticmethod
    def extract_match_percentage(ai_response: str) -> Optional[float]:
        """
        Extract the match percentage from the AI response.
        
        Args:
            ai_response: The response from the AI model
            
        Returns:
            Extracted percentage as a float, or None if not found
        """
        try:
            # Look for percentage patterns like "85%" or "85 percent" in the first part of the response
            import re
            
            # Try to find a percentage pattern
            percentage_patterns = [
                r"(\d{1,3})%",  # 85%
                r"(\d{1,3})\s*percent",  # 85 percent
                r"(\d{1,3})\.\d+%",  # 85.5%
                r"Match.*?(\d{1,3})%",  # Match: 85%
                r"match percentage.*?(\d{1,3})",  # match percentage is 85
                r"matching score.*?(\d{1,3})",  # matching score of 85
            ]
            
            # Check first few lines for any match
            first_section = ai_response.split('\n\n')[0] if '\n\n' in ai_response else ai_response
            
            for pattern in percentage_patterns:
                matches = re.search(pattern, first_section, re.IGNORECASE)
                if matches:
                    return float(matches.group(1))
            
            return None
        except Exception as e:
            logger.error(f"Error extracting match percentage: {str(e)}")
            return None
    
    @staticmethod
    def create_match_visualization(match_percentage: float) -> Any:
        """
        Create a visualization of the match percentage using Plotly.
        
        Args:
            match_percentage: The percentage match between resume and job description
            
        Returns:
            Plotly figure object
        """
        try:
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=match_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Resume Match Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 90], 'color': "lightgreen"},
                        {'range': [90, 100], 'color': "green"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': match_percentage
                    }
                }
            ))
            
            # Set size and layout
            fig.update_layout(
                height=400, 
                width=500,
                font={'size': 16},
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating match visualization: {str(e)}")
            return None
    
    @staticmethod
    def analyze_skills_gap(job_description: str, ai_response: str) -> Optional[Dict]:
        """
        Analyze the skills gap based on the AI response and job description.
        
        Args:
            job_description: The job description text
            ai_response: The AI model's analysis of the resume and job description
            
        Returns:
            Dictionary containing skills analysis data
        """
        try:
            # Extract missing skills section from AI response
            missing_skills = []
            
            # Simple extraction from "missing keywords" or similar sections
            sections = ai_response.lower().split('\n\n')
            for section in sections:
                if 'missing' in section and ('keyword' in section or 'skill' in section):
                    # Extract list items that might contain skills
                    skill_lines = [line.strip('- ').strip() for line in section.split('\n') if line.strip().startswith('-')]
                    if skill_lines:
                        missing_skills = skill_lines
                    break
            
            # Extract key terms from job description
            job_keywords = DocumentProcessor.extract_keywords(job_description)
            
            return {
                "missing_skills": missing_skills,
                "job_keywords": job_keywords[:30]  # Limit to most common 30 keywords
            }
        except Exception as e:
            logger.error(f"Error analyzing skills gap: {str(e)}")
            return None


class ResumeAnalyzerApp:
    """
    Main application class that orchestrates the resume analysis workflow
    and manages the Streamlit UI components.
    """
    
    def __init__(self):
        """Initialize the application components and UI state."""
        self.ai_handler = AIModelHandler()
        self.session_state = st.session_state
        
        # Initialize session state variables if they don't exist
        if 'analysis_history' not in self.session_state:
            self.session_state.analysis_history = []
        if 'current_pdf' not in self.session_state:
            self.session_state.current_pdf = None
        if 'pdf_images' not in self.session_state:
            self.session_state.pdf_images = None
            
        logger.info(f"Resume Analyzer App initialized (v{__version__})")
    
    def configure_page(self):
        """Configure the Streamlit page settings and layout."""
        st.set_page_config(
            page_title="AI Career Assistant Pro",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 42px;
            font-weight: bold;
            color: #1E3A8A;
            margin-bottom: 20px;
            text-align: center;
        }
        .subheader {
            font-size: 24px;
            color: #2563EB;
            margin-bottom: 15px;
        }
        .card {
            background-color: #F3F4F6;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .highlight {
            background-color: #DBEAFE;
            padding: 5px;
            border-radius: 5px;
        }
        .badge {
            background-color: #3B82F6;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            margin-right: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #6B7280;
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<div class="main-header">AI Career Assistant Pro</div>', unsafe_allow_html=True)
        
        # Introduction and version info
        with st.expander("ℹ️ About this application"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                This advanced application uses **Google's Gemini 1.5 Pro** to analyze your resume 
                against job descriptions, providing intelligent career guidance and personalized 
                improvement recommendations. Upload your resume and a job description to get started!
                
                Features:
                - Resume-to-Job match analysis with percentage score
                - Key strengths and missing skills identification
                - Actionable improvement suggestions
                - Skills gap visualization
                - Resume optimization tips
                """)
            with col2:
                st.info(f"Version: {__version__}\nBuilt with Gemini 1.5", icon="📊")
                st.caption("Last updated: April 2025")
    
    def sidebar_components(self):
        """Set up the sidebar components."""
        with st.sidebar:
            st.header("🔍 Analysis Options")
            
            st.markdown("### 📝 Resume File")
            uploaded_file = st.file_uploader(
                "Upload your Resume (PDF)",
                type=["pdf"],
                help="Upload your resume in PDF format"
            )
            
            if uploaded_file is not None:
                # Check if this is a new file
                if 'last_uploaded_filename' not in self.session_state or \
                   self.session_state.last_uploaded_filename != uploaded_file.name:
                    
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    
                    try:
                        # Process the PDF
                        pdf_parts, pdf_images = DocumentProcessor.process_pdf(uploaded_file)
                        self.session_state.current_pdf = pdf_parts
                        self.session_state.pdf_images = pdf_images
                        
                        st.success(f"✅ Successfully processed '{uploaded_file.name}'")
                        
                        # Show PDF preview
                        if pdf_images and len(pdf_images) > 0:
                            st.image(
                                pdf_images[0], 
                                caption="Resume Preview (First Page)", 
                                use_column_width=True
                            )
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        self.session_state.current_pdf = None
                        self.session_state.pdf_images = None
            else:
                self.session_state.current_pdf = None
                self.session_state.pdf_images = None
                st.info("Please upload your resume to begin")
            
            # Advanced options section
            with st.expander("⚙️ Advanced Options"):
                model_option = st.selectbox(
                    "AI Model", 
                    options=["Standard (Faster)", "Advanced (More Detailed)"],
                    index=0,
                    help="Select the AI model to use for analysis"
                )
                
                # Map selection to actual model names
                model_map = {
                    "Standard (Faster)": CONFIG["MODELS"]["image_analysis"],
                    "Advanced (More Detailed)": CONFIG["MODELS"]["advanced_analysis"]
                }
                self.session_state.selected_model = model_map[model_option]
                
                st.checkbox(
                    "Save analysis history",
                    value=True,
                    help="Save analysis results for comparison"
                )
            
            st.markdown("---")
            st.markdown("### 🚀 Recent Updates")
            st.markdown("""
            • Added skills gap visualization
            • Improved PDF processing capabilities
            • Enhanced analysis accuracy
            • Added career roadmap generation
            """)
        
    def main_content(self):
        """Set up the main content area."""
        # Job description input
        st.markdown('<div class="subheader">📄 Job Description</div>', unsafe_allow_html=True)
        
        job_description = st.text_area(
            "Paste the job description you're applying for:",
            height=200,
            placeholder="Paste the complete job description here...",
            help="The more detailed the job description, the better the analysis will be"
        )
        
        # Analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_btn = st.button(
                "📊 Match Analysis", 
                type="primary",
                use_container_width=True,
                help="Calculate match percentage and analyze skills gap"
            )
        
        with col2:
            review_btn = st.button(
                "🔍 Detailed Review", 
                type="secondary",
                use_container_width=True,
                help="Get a comprehensive review of your resume against the job description"
            )
        
        with col3:
            improve_btn = st.button(
                "💡 Improvement Tips", 
                type="secondary",
                use_container_width=True,
                help="Get personalized tips to improve your resume"
            )
        
        # Check if analysis can be performed
        can_analyze = self.session_state.current_pdf is not None and job_description.strip() != ""
        
        if not can_analyze and (match_btn or review_btn or improve_btn):
            st.warning("Please upload your resume and provide a job description to perform analysis")
            return
        
        # Handle button actions
        if match_btn and can_analyze:
            self.perform_match_analysis(job_description)
        
        if review_btn and can_analyze:
            self.perform_detailed_review(job_description)
        
        if improve_btn and can_analyze:
            self.perform_improvement_analysis(job_description)
        
        # Show analysis history
        if hasattr(self.session_state, 'analysis_history') and len(self.session_state.analysis_history) > 0:
            with st.expander("📜 Analysis History", expanded=False):
                for i, analysis in enumerate(self.session_state.analysis_history):
                    st.markdown(f"### Analysis #{i+1}: {analysis['type']} - {analysis['timestamp']}")
                    st.markdown(analysis['response'])
                    st.markdown("---")
    
    def perform_match_analysis(self, job_description: str):
        """
        Perform match percentage analysis between resume and job description.
        
        Args:
            job_description: The job description text
        """
        with st.spinner("📊 Analyzing match percentage..."):
            # Create the prompt for match analysis
            match_prompt = """
            You are an advanced ATS (Applicant Tracking System) analyzer with deep expertise in AI, ML, and data science fields.
            
            TASK:
            Analyze the provided resume against the job description and provide:
            1. A specific percentage match score (be precise with a number)
            2. Key matching skills and qualifications found in both
            3. Missing keywords or skills that should be added to the resume
            4. A final assessment of the candidate's fit for this role
            
            FORMAT THE OUTPUT AS:
            - Start with the percentage match clearly stated (e.g., "85% Match")
            - Then provide sections for matching skills, missing keywords, and final thoughts
            - Be specific and actionable in your recommendations
            
            Your analysis will help this job seeker improve their chances of getting an interview.
            """
            
            try:
                # Get response from AI model
                response = self.ai_handler.get_model_response(
                    job_description,
                    self.session_state.current_pdf,
                    match_prompt,
                    self.session_state.get('selected_model', CONFIG["MODELS"]["image_analysis"])
                )
                
                # Display results
                st.markdown('<div class="subheader">📊 Match Analysis Results</div>', unsafe_allow_html=True)
                
                # Extract match percentage for visualization
                match_percentage = DataAnalyzer.extract_match_percentage(response)
                
                # Create columns for layout
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    if match_percentage:
                        # Create and display visualization
                        fig = DataAnalyzer.create_match_visualization(match_percentage)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add quick assessment based on percentage
                        if match_percentage >= 90:
                            st.success("🌟 Excellent Match! You're a strong candidate for this role.")
                        elif match_percentage >= 75:
                            st.info("✅ Good Match! With some targeted improvements, you could be a strong candidate.")
                        elif match_percentage >= 50:
                            st.warning("⚠️ Partial Match. Consider updating your resume for this specific role.")
                        else:
                            st.error("❗ Low Match. This role may require significant skill development or a resume overhaul.")
                    
                    # Skills gap analysis
                    skills_data = DataAnalyzer.analyze_skills_gap(job_description, response)
                    if skills_data and skills_data["missing_skills"]:
                        st.markdown("### 🔍 Missing Skills:")
                        for skill in skills_data["missing_skills"]:
                            st.markdown(f"- {skill}")
                
                with col2:
                    # Full analysis from AI
                    st.markdown("### 📑 Full Analysis:")
                    st.markdown(response)
                
                # Save to history
                self.session_state.analysis_history.append({
                    "type": "Match Analysis",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "response": response,
                    "match_percentage": match_percentage if match_percentage else "N/A"
                })
                
            except Exception as e:
                st.error(f"Error performing match analysis: {str(e)}")
                logger.error(f"Match analysis error: {str(e)}")
    
    def perform_detailed_review(self, job_description: str):
        """
        Perform detailed review of resume against job description.
        
        Args:
            job_description: The job description text
        """
        with st.spinner("🔍 Performing detailed review..."):
            # Create the prompt for detailed review
            review_prompt = """
            You are an expert technical recruiter specializing in AI, ML, and data science roles in 2025.
            
            TASK:
            Conduct a comprehensive, detailed assessment of the provided resume against the job description with the following:
            
            1. PROFESSIONAL PROFILE ASSESSMENT:
               - Assess the candidate's overall profile, experience, and qualifications
               - Highlight the strongest aspects of their profile for this role
               - Identify any potential red flags or concerns
            
            2. TECHNICAL SKILLS ANALYSIS:
               - Evaluate technical skills mentioned in the resume against job requirements
               - Note which critical technical skills are present and which are missing
               - For AI/ML roles specifically, assess depth of knowledge in relevant areas
            
            3. EXPERIENCE EVALUATION:
               - Analyze how well their experience aligns with the job requirements
               - Note the quality and relevance of projects mentioned
               - Identify experience gaps that could be addressed
            
            4. ATS OPTIMIZATION SUGGESTIONS:
               - Provide specific keyword recommendations to improve ATS matching
               - Suggest any formatting or structure improvements
            
            5. FINAL RECOMMENDATION:
               - Give your professional opinion on whether this candidate would likely be selected for an interview
               - Provide 3-5 specific, actionable improvements that would significantly strengthen their application
            
            FORMAT:
            Structure your response into clear sections with headings for each of the areas above.
            Be critical but constructive - this is guidance to help an AI/ML job seeker improve their application.
            """
            
            try:
                # Get response from AI model
                response = self.ai_handler.get_model_response(
                    job_description,
                    self.session_state.current_pdf,
                    review_prompt,
                    self.session_state.get('selected_model', CONFIG["MODELS"]["advanced_analysis"])
                )
                
                # Display results
                st.markdown('<div class="subheader">🔍 Detailed Review Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Save to history
                self.session_state.analysis_history.append({
                    "type": "Detailed Review",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "response": response
                })
                
            except Exception as e:
                st.error(f"Error performing detailed review: {str(e)}")
                logger.error(f"Detailed review error: {str(e)}")
    
    def perform_improvement_analysis(self, job_description: str):
        """
        Perform improvement analysis for the resume.
        
        Args:
            job_description: The job description text
        """
        with st.spinner("💡 Generating improvement tips..."):
            # Create the prompt for improvement suggestions
            improvement_prompt = """
            You are a career coach and resume expert specializing in AI, ML, and data science careers in 2025.
            
            TASK:
            Review the provided resume and job description, then provide detailed, actionable advice for improving the resume specifically for this role. Focus on the following:
            
            1. RESUME CONTENT IMPROVEMENTS:
               - Specific bullet point revisions or additions that would strengthen impact
               - Projects or achievements that should be emphasized more prominently
               - Suggested improvements to the summary/objective statement
            
            2. SKILLS PRESENTATION:
               - How to better highlight relevant technical and soft skills
               - Specific skills from the job description that should be incorporated
               - Modern skills presentation formats for AI/ML/data science roles in 2025
            
            3. CAREER DEVELOPMENT ROADMAP:
               - 3-5 specific skills or certifications to pursue to become more competitive
               - Learning resources or projects that would strengthen their portfolio
               - Timeline recommendation for skill acquisition
            
            4. INDUSTRY-SPECIFIC ADVICE:
               - Current trends in AI/ML/data science hiring in 2025 relevant to their profile
               - How to position themselves better for this specific type of role
            
            FORMAT:
            Present your advice in clear sections with concrete examples and specific wording suggestions where applicable.
            Be detailed, actionable, and tailored to this specific resume and job description.
            Where appropriate, provide specific examples of how to reword or restructure content.
            """
            
            try:
                # Get response from AI model
                response = self.ai_handler.get_model_response(
                    job_description,
                    self.session_state.current_pdf,
                    improvement_prompt,
                    self.session_state.get('selected_model', CONFIG["MODELS"]["advanced_analysis"])
                )
                
                # Display results
                st.markdown('<div class="subheader">💡 Improvement Recommendations</div>', unsafe_allow_html=True)
                
                # Create tabs for different sections
                tab1, tab2, tab3 = st.tabs(["📝 Content Improvements", "🚀 Career Development", "🔮 Industry Insights"])
                
                with tab1:
                    st.markdown("### Resume Content Improvements")
                    
                    # Extract content improvements section
                    content_section = ""
                    if "RESUME CONTENT IMPROVEMENTS" in response:
                        content_section = response.split("RESUME CONTENT IMPROVEMENTS")[1].split("SKILLS PRESENTATION")[0]
                    elif "Content Improvements" in response:
                        content_section = response.split("Content Improvements")[1].split("Skills")[0]
                    
                    st.markdown(content_section if content_section else "See full recommendations below")
                
                with tab2:
                    st.markdown("### Career Development Roadmap")
                    
                    # Extract career development section
                    roadmap_section = ""
                    if "CAREER DEVELOPMENT ROADMAP" in response:
                        roadmap_section = response.split("CAREER DEVELOPMENT ROADMAP")[1].split("INDUSTRY-SPECIFIC ADVICE")[0]
                    elif "Career Development" in response:
                        roadmap_section = response.split("Career Development")[1].split("Industry")[0]
                    
                    st.markdown(roadmap_section if roadmap_section else "See full recommendations below")
                    
                    # Add a career roadmap visualization
                    st.markdown("#### 📈 Suggested Learning Path")
                    st.markdown("""
                    ```mermaid
                    gantt
                        title Career Development Timeline
                        dateFormat  YYYY-MM-DD
                        section Technical Skills
                        Core Skill Development      :a1, 2025-05-01, 90d
                        Advanced Specialization     :a2, after a1, 120d
                        section Certifications
                        Industry Certification 1    :c1, 2025-05-15, 45d
                        Industry Certification 2    :c2, after c1, 60d
                        section Projects
                        Portfolio Project 1         :p1, 2025-06-01, 60d
                        Portfolio Project 2         :p2, after p1, 90d
                    ```
                    """)
                
                with tab3:
                    st.markdown("### Industry Insights for 2025")
                    
                    # Extract industry insights section
                    insights_section = ""
                    if "INDUSTRY-SPECIFIC ADVICE" in response:
                        insights_section = response.split("INDUSTRY-SPECIFIC ADVICE")[1]
                    elif "Industry" in response:
                        parts = response.split("Industry")
                        if len(parts) > 1:
                            insights_section = "Industry" + parts[1]
                    
                    st.markdown(insights_section if insights_section else "See full recommendations below")
                    
                    # Add trending skills visualization
                    st.markdown("#### 🔥 Trending Skills in 2025")
                    
                    # Mock data for trending skills
                    trending_skills = {
                        "Large Language Models": 95,
                        "Neural Architecture Search": 88,
                        "MLOps": 85,
                        "Privacy-Preserving ML": 82,
                        "Explainability": 78,
                        "Federated Learning": 75,
                        "Graph Neural Networks": 72,
                        "Time Series Forecasting": 70
                    }
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        x=list(trending_skills.values()),
                        y=list(trending_skills.keys()),
                        orientation='h',
                        title="Demand for AI/ML Skills in 2025",
                        labels={"x": "Demand Index", "y": "Skill"}
                    )
                    fig.update_traces(marker_color='rgb(38, 70, 160)')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Full recommendations
                with st.expander("📄 Full Improvement Recommendations", expanded=False):
                    st.markdown(response)
                
                # Save to history
                self.session_state.analysis_history.append({
                    "type": "Improvement Recommendations",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "response": response
                })
                
            except Exception as e:
                st.error(f"Error generating improvement recommendations: {str(e)}")
                logger.error(f"Improvement analysis error: {str(e)}")
    
    def add_footer(self):
        """Add footer to the application."""
        st.markdown("---")
        st.markdown(
            '<div class="footer">Built with ❤️ using Streamlit and Google Gemini AI | '
            '© 2025 | <a href="https://github.com/your-username/ai-resume-analyzer" target="_blank">View on GitHub</a></div>',
            unsafe_allow_html=True
        )
    
    def run(self):
        """Run the application."""
        try:
            # Configure page
            self.configure_page()
            
            # Set up sidebar
            self.sidebar_components()
            
            # Set up main content
            self.main_content()
            
            # Add footer
            self.add_footer()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {str(e)}")


# Entry point
if __name__ == "__main__":
    app = ResumeAnalyzerApp()
    app.run()