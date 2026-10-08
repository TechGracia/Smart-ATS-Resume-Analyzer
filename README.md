# 🎯 Smart ATS Resume Analyzer

An AI-powered web application that analyzes resumes against job descriptions and provides actionable insights to improve job matching.

---

## 🚀 Live Demo

👉 **[Open App](https://smartaianalyzer.streamlit.app/)**

---

## 📌 Features

### 🧠 Resume Analysis

* Extracts and cleans resume + job description text
* Uses **TF-IDF + Cosine Similarity** to measure relevance

---

### 📊 ATS Fit Score

* Calculates final score using:

  **Fit Score = 0.6 × Similarity + 0.4 × Skill Match**

---

### 🛠️ Skill Matching

* Identifies:

  * ✅ Matched Skills
  * ❌ Missing Skills
* Provides skill gap insights

---

### 📈 Data Visualization

* Matched vs Missing Skills (Bar Chart)
* Keyword Frequency Graph
* Score Breakdown

---

### 🔍 Keyword Highlighting

* Highlights:

  * 🟢 Matched keywords
  * 🔴 Missing keywords

---

### 📂 Section-wise Analysis

* Evaluates:

  * Skills
  * Projects
  * Experience
  * Education

---

### 🛡️ ATS Checker

* Checks resume quality:

  * Length
  * Contact info
  * Quantified achievements
  * Formatting

---

### ✨ Resume Improvement Suggestions

* Generates actionable tips to improve resume

---

### 🌍 Job Location Insights

* Extracts job locations from JD
* Normalizes duplicate locations
* Displays:

  * 📍 Map with markers
  * 🔥 Heatmap
  * 📊 Location frequency

---

### 📄 PDF Report

* Download complete analysis report

---

### 🎨 Premium UI

* Glassmorphism design
* Neon effects
* Animated particle background

---

## 🧰 Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Matplotlib**
* **Pandas & NumPy**
* **Folium (Maps)**
* **Geopy (Location)**
* **ReportLab (PDF)**

---

## 📁 Project Structure

```bash
resume-analyzer/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/YOUR_USERNAME/smart-ats-resume-analyzer.git
cd smart-ats-resume-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Use Case

This project simulates a real-world **Applicant Tracking System (ATS)** used by companies to filter resumes.

It helps users:

* Understand resume-job match
* Identify missing skills
* Improve resume quality
* Visualize job insights

---

## 📌 Future Improvements

* AI-based resume rewriting
* Better NLP (spaCy / transformers)
* Job recommendation system
* Multi-role comparison

---

## 🙌 Acknowledgment

Developed as part of an academic project to demonstrate:

* NLP concepts
* Data visualization
* Full-stack development using Python

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
