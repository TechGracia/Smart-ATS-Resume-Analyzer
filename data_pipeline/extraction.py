import re


# ---------------------------------------------------------
# EMAIL EXTRACTION
# ---------------------------------------------------------

def extract_email(text):
    """Extract email address from resume text."""

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return ""


# ---------------------------------------------------------
# PHONE EXTRACTION
# ---------------------------------------------------------

def extract_phone(text):
    """Extract phone number from resume text."""

    patterns = [
        r"\+91[\s-]?\d{5}[\s-]?\d{5}",
        r"\+91[\s-]?\d{10}",
        r"\b\d{10}\b"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return ""


# ---------------------------------------------------------
# NAME EXTRACTION
# ---------------------------------------------------------

def extract_name(text):
    """Extract candidate name from the beginning of the resume."""

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        lower_line = line.lower()

        # Ignore common headings
        if lower_line in [
            "resume",
            "curriculum vitae",
            "cv",
            "curriculum vitae resume"
        ]:
            continue

        # Ignore contact information
        if "@" in line:
            continue

        if re.search(r"\d{7,}", line):
            continue

        # Ignore obvious headings
        if lower_line in [
            "professional summary",
            "summary",
            "objective",
            "profile"
        ]:
            continue

        return line.title()

    return ""


# ---------------------------------------------------------
# SECTION DETECTION
# ---------------------------------------------------------

SECTION_ALIASES = {

    "skills": [
        "skills",
        "technical skills",
        "technical skill",
        "core skills",
        "key skills",
        "professional skills"
    ],

    "education": [
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
        "academic qualification",
        "educational qualifications"
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history"
    ],

    "projects": [
        "projects",
        "project",
        "key projects",
        "academic projects",
        "personal projects",
        "major projects"
    ],

    "certifications": [
        "certifications",
        "certification",
        "certificates",
        "certifications and achievements"
    ],

    "achievements": [
        "achievements",
        "achievement",
        "key achievements",
        "accomplishments"
    ]
}

# ---------------------------------------------------------
# FIND SECTION
# ---------------------------------------------------------

def find_section_start(lines, section_names):
    """Find the starting line of a resume section."""

    for i, line in enumerate(lines):

        cleaned_line = line.strip().lower()

        # Remove common punctuation
        cleaned_line = cleaned_line.replace(":", "")
        cleaned_line = cleaned_line.replace("-", " ")

        if cleaned_line in section_names:
            return i + 1

    return None


# ---------------------------------------------------------
# EXTRACT SECTION
# ---------------------------------------------------------

def extract_section(text, section_type):
    """
    Extract content belonging to a specific resume section.
    """

    lines = text.split("\n")

    section_names = SECTION_ALIASES.get(section_type, [])

    start_index = find_section_start(
        lines,
        section_names
    )

    if start_index is None:
        return ""

    # All known section headings
    all_headings = []

    for aliases in SECTION_ALIASES.values():
        all_headings.extend(aliases)

    content = []

    for line in lines[start_index:]:

        cleaned_line = line.strip().lower()

        cleaned_line = cleaned_line.replace(":", "")
        cleaned_line = cleaned_line.replace("-", " ")

        # Stop when another section begins
        if cleaned_line in all_headings:
            break

        if line.strip():
            content.append(line.strip())

    return " ".join(content)


# ---------------------------------------------------------
# SKILLS EXTRACTION
# ---------------------------------------------------------

def extract_skills(text):
    """Find known technical skills in the resume."""

    known_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c",
        "sql",
        "mysql",
        "mongodb",
        "postgresql",
        "html",
        "html5",
        "css",
        "css3",
        "react",
        "react.js",
        "node.js",
        "express.js",
        "django",
        "flask",
        "fastapi",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "git",
        "github",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "scikit-learn",
        "power bi",
        "tableau",
        "excel",
        "rest api",
        "firebase",
        "langchain",
        "openai",
        "bert",
        "nlp",
        "redis",
        "php",
        "flutter",
        "dart",
        "verilog",
        "systemverilog"
    ]

    text_lower = text.lower()

    found_skills = []

    for skill in known_skills:

        if skill.lower() in text_lower:

            if skill not in found_skills:
                found_skills.append(skill)

    return ", ".join(found_skills)


# ---------------------------------------------------------
# REMOVE SUSPICIOUS CONTENT
# ---------------------------------------------------------

def remove_suspicious_content(text):
    """
    Remove obvious instruction-like content that should not
    be treated as candidate information.
    """

    suspicious_patterns = [
        r"ignore all previous instructions.*?(?=\n|$)",
        r"ignore previous instructions.*?(?=\n|$)",
        r"shortlist this candidate.*?(?=\n|$)",
        r"do not reject this resume.*?(?=\n|$)"
    ]

    cleaned_text = text

    for pattern in suspicious_patterns:

        cleaned_text = re.sub(
            pattern,
            "",
            cleaned_text,
            flags=re.IGNORECASE
        )

    return cleaned_text


# ---------------------------------------------------------
# COMPLETE TRANSFORMATION
# ---------------------------------------------------------

def transform_resume(resume):
    """
    Convert cleaned resume text into structured candidate data.
    """

    text = resume.get("cleaned_text", "")

    # Remove suspicious instruction-like text
    text = remove_suspicious_content(text)

    name = extract_name(text)

    email = extract_email(text)

    phone = extract_phone(text)

    skills = extract_skills(text)

    education = extract_section(
        text,
        "education"
    )

    experience = extract_section(
        text,
        "experience"
    )

    projects = extract_section(
        text,
        "projects"
    )

    certifications = extract_section(
        text,
        "certifications"
    )
    achievements = extract_section(
    text,
    "achievements"
)

    return {

        "filename": resume.get("filename", ""),

        "name": name,

        "email": email,

        "phone": phone,

        "skills": skills,

        "education": education,

        "experience": experience,

        "projects": projects,

        "certifications": certifications,
        
        "achievements": achievements
    }