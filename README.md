# 🎯 Smart ATS Resume Analyzer

An intelligent resume analysis web application that combines **AI-assisted career preparation, Applicant Tracking System (ATS) analysis, and a Data Engineering pipeline**. It analyzes resumes against job descriptions, identifies skill gaps, generates improvement suggestions, and transforms unstructured resume PDFs into structured, validated candidate data.

## 🚀 Live Demo

👉 **[Open Smart ATS Resume Analyzer](https://smartaianalyzer.streamlit.app/)**

## ✨ Key Features

### 🧠 1. Resume and Job Description Analysis
- Upload resume PDFs and provide job descriptions.
- Extract and clean resume and job description text.
- Measure textual relevance using TF-IDF and cosine similarity.
- Evaluate how closely a resume aligns with a target role.

### 📊 2. ATS Fit Score
- Combines resume similarity and skill matching.
- Uses a weighted scoring formula:

  **Fit Score = 0.6 × Similarity + 0.4 × Skill Match**

- Displays analysis metrics to help users understand resume-job alignment.

### 🛠️ 3. Intelligent Skill Matching
- Identifies matched and missing skills.
- Highlights skill gaps against job requirements.
- Provides insights to guide targeted resume improvements.

### 📈 4. Keyword and Section Analysis
- Analyzes keyword frequency.
- Highlights matched and missing keywords.
- Evaluates resume sections such as skills, projects, experience, and education.
- Presents visual summaries of analysis results.

### 🛡️ 5. ATS Resume Checker
Evaluates important resume quality indicators, including:
- Resume length
- Contact information
- Quantified achievements
- Formatting and content-related checks

### ✨ 6. Resume Improvement Suggestions
Provides actionable suggestions to improve resume relevance, content quality, and alignment with a target job description.

### 🌍 7. Job Location Insights
- Extracts job locations from job descriptions.
- Normalizes duplicate location names.
- Displays geographic insights using interactive maps, markers, and location frequency visualizations.
- Supports fallback coordinates for selected locations when geocoding is unavailable.

### ⚙️ 8. Data Engineering Pipeline

The project includes a modular pipeline for processing resume PDFs in batches.

**Pipeline architecture:**

```text
Resume PDFs
     ↓
Data Ingestion
     ↓
Text Extraction
     ↓
Data Cleaning
     ↓
Data Transformation
     ↓
Data Validation
     ↓
Structured Candidate Data
     ↓
CSV Output and ATS Analysis
```

**Pipeline capabilities:**
- Batch ingestion of PDF resumes.
- Text extraction using `pdfplumber`.
- Text normalization and cleaning.
- Structured extraction of candidate information.
- Validation of names, email addresses, phone numbers, and resume sections.
- Duplicate email detection within the processed batch.
- Data quality scoring.
- Classification of records as `VALID` or `REVIEW`.
- Generation of processed data and validation reports in CSV format.

### 📋 9. Structured Resume Data

The pipeline extracts structured fields such as:
- Candidate name
- Email address
- Phone number
- Skills
- Education
- Experience
- Projects
- Certifications
- Achievements

### 📄 10. Export and Reports
- Downloadable PDF analysis reports.
- Structured resume data in CSV format.
- Validation reports containing data quality scores, validation flags, duplicate indicators, and processing status.

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Data Processing | Pandas, NumPy |
| NLP and Similarity | Scikit-learn, TF-IDF, Cosine Similarity |
| PDF Text Extraction | pdfplumber, PyMuPDF where used |
| Data Visualization | Matplotlib |
| Geographic Insights | Folium, Geopy |
| PDF Report Generation | ReportLab |
| Data Storage and Export | CSV |
| Version Control | Git, GitHub |

## 🏗️ Project Structure

```text
resume-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data_pipeline/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── extraction.py
│   ├── validation.py
│   ├── output.py
│   └── pipeline.py
│
├── input_resumes/       # Local test PDFs; excluded from Git
└── output/              # Generated CSV reports; excluded from Git
```

## ⚙️ Installation and Local Setup

### Prerequisites
- Python 3.10 or a compatible version supported by the project's dependencies.
- Git.
- pip.

### 1. Clone the repository

```bash
git clone https://github.com/TechGracia/Smart-ATS-Resume-Analyzer.git
cd Smart-ATS-Resume-Analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Data Engineering pipeline

Place your test resume PDFs in the local `input_resumes/` directory, then execute:

```bash
python -m data_pipeline.pipeline
```

The pipeline generates:

- `output/processed_resumes.csv`
- `output/validation_report.csv`

### 5. Launch the web application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 🎯 Project Objectives

This project demonstrates how resume analysis and data processing can be combined in a practical application. Its objectives are to:

- Automate the extraction of information from unstructured resume documents.
- Transform extracted text into structured candidate records.
- Apply data quality checks and validation rules.
- Generate reusable CSV outputs for downstream processing.
- Analyze resume-job relevance and identify skill gaps.
- Present results through an interactive web interface.

## 🔒 Data Handling

- Local test resume PDFs are excluded from version control.
- Generated pipeline CSV files are excluded from version control.
- Environment files and local secrets are excluded using `.gitignore`.
- Duplicate email detection currently operates within the processed batch.

**Note:** Avoid uploading real candidates' personal information, API keys, or credentials to a public repository.

## 🔮 Future Enhancements

Potential improvements include:
- Database integration for validated candidate records.
- More advanced NLP-based entity extraction.
- Improved resume classification and job recommendations.
- More comprehensive data quality monitoring.
- Automated testing and continuous integration.
- Cloud deployment and scalable batch processing.

## 🎓 Academic Context

Developed as an academic project to demonstrate concepts from **Data Engineering, Natural Language Processing, data validation, machine learning, data visualization, and Python-based web application development**.

## 👩‍💻 Author

**Gracia Sharon Jopson**

GitHub: [TechGracia](https://github.com/TechGracia)

---

⭐ If you find this project useful, consider giving the repository a star.
