# AI-Powered Resume Analyzer & Career Assistant

**Version:** 2.5.0  
**Author:** Siddhi Shrivastava
**Last Updated:** May 2025

## Overview

AI Career Assistant Pro is a powerful, AI-driven web application that leverages Google's Gemini 1.5 Pro to analyze resumes against job descriptions. It provides intelligent insights such as match scoring, skill gap analysis, and personalized career development recommendations. The tool is ideal for job seekers aiming to optimize their resumes for specific roles in AI, machine learning, data science, and related fields.

## Features

- Upload resume in PDF format for analysis
- Match percentage calculation between resume and job description
- Identification of key strengths and missing skills
- Actionable improvement suggestions
- Interactive visualizations with Plotly
- Personalized career development roadmap
- Fully modular and production-ready code

## Tech Stack

- **Frontend:** Streamlit  
- **AI Backend:** Google Gemini 1.5 Pro & Flash  
- **NLP:** NLTK  
- **Data Visualization:** Plotly, Matplotlib, Seaborn  
- **PDF/Image Processing:** pdf2image, Pillow (PIL)  
- **Environment Management:** python-dotenv  
- **Logging:** Python `logging` module

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
````

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

Create a `.env` file in the root directory with the following content:

```
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Run the application:**

```bash
streamlit run app.py
```

## File Structure

```
.
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## How It Works

1. Upload your resume in PDF format.
2. Paste the job description for the target role.
3. Choose between standard and advanced AI models.
4. Select an action:

   * Match Analysis
   * Detailed Review
   * Improvement Tips
5. View analysis results, match score, skill gaps, and personalized tips.

## Acknowledgements

* [Google Generative AI](https://ai.google/discover/gemini/)
* [Streamlit](https://streamlit.io/)
* [NLTK](https://www.nltk.org/)
* [Plotly](https://plotly.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


