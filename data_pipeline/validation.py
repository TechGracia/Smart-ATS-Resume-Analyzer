import re


# ---------------------------------------------------------
# EMAIL VALIDATION
# ---------------------------------------------------------

def validate_email(email):
    """Check whether the email has a valid format."""

    if not email:
        return False

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return bool(re.match(pattern, email))


# ---------------------------------------------------------
# PHONE VALIDATION
# ---------------------------------------------------------

def validate_phone(phone):
    """Check whether the phone number has a valid format."""

    if not phone:
        return False

    # Remove spaces and hyphens
    cleaned_phone = re.sub(r"[\s-]", "", phone)

    # Indian phone number with +91
    if re.fullmatch(r"\+91\d{10}", cleaned_phone):
        return True

    # Indian phone number without country code
    if re.fullmatch(r"\d{10}", cleaned_phone):
        return True

    return False


# ---------------------------------------------------------
# NAME VALIDATION
# ---------------------------------------------------------

def validate_name(name):
    """Check whether a candidate name was extracted."""

    if not name:
        return False

    # Name should contain at least two characters
    if len(name.strip()) < 2:
        return False

    return True


# ---------------------------------------------------------
# SECTION VALIDATION
# ---------------------------------------------------------

def validate_sections(resume):
    """
    Check whether important resume sections are present.
    """

    required_sections = [
        "skills",
        "education"
    ]

    missing_sections = []

    for section in required_sections:

        if not resume.get(section, "").strip():
            missing_sections.append(section)

    return missing_sections


# ---------------------------------------------------------
# RESUME LENGTH VALIDATION
# ---------------------------------------------------------

def validate_resume_length(resume):
    """
    Check whether enough resume information was extracted.
    """

    fields = [
        "name",
        "email",
        "phone",
        "skills",
        "education",
        "experience",
        "projects"
    ]

    filled_fields = 0

    for field in fields:

        if resume.get(field, "").strip():
            filled_fields += 1

    # At least 4 important fields should be available
    return filled_fields >= 4


# ---------------------------------------------------------
# DUPLICATE VALIDATION
# ---------------------------------------------------------

def check_duplicate_email(resume, seen_emails):
    """
    Check whether the email already appeared
    in another resume.
    """

    email = resume.get("email", "").strip().lower()

    if not email:
        return False

    if email in seen_emails:
        return True

    seen_emails.add(email)

    return False

# ---------------------------------------------------------
# DATA QUALITY SCORE
# ---------------------------------------------------------

def calculate_data_quality_score(
    email_valid,
    phone_valid,
    name_valid,
    missing_sections,
    resume_length_valid,
    duplicate
):
    """
    Calculate a data quality score out of 100.

    Each basic quality check contributes 20 points.
    Duplicate records are capped at 80 points.
    """

    score = 0

    # Email check
    if email_valid:
        score += 20

    # Phone check
    if phone_valid:
        score += 20

    # Name check
    if name_valid:
        score += 20

    # Required sections check
    if len(missing_sections) == 0:
        score += 20

    # Resume completeness check
    if resume_length_valid:
        score += 20

    # Duplicate records require review
    if duplicate:
        score = min(score, 80)

    return score
# ---------------------------------------------------------
# OVERALL VALIDATION
# ---------------------------------------------------------

def validate_resume(resume, seen_emails):
    """
    Perform all data quality checks on one resume.
    """

    email_valid = validate_email(
        resume.get("email", "")
    )

    phone_valid = validate_phone(
        resume.get("phone", "")
    )

    name_valid = validate_name(
        resume.get("name", "")
    )

    missing_sections = validate_sections(
        resume
    )

    resume_length_valid = validate_resume_length(
        resume
    )

    duplicate = check_duplicate_email(
        resume,
        seen_emails
    )

    # Calculate overall data quality score
    data_quality_score = calculate_data_quality_score(
        email_valid,
        phone_valid,
        name_valid,
        missing_sections,
        resume_length_valid,
        duplicate
    )

    # Overall validation status
    if (
        email_valid
        and phone_valid
        and name_valid
        and len(missing_sections) == 0
        and resume_length_valid
        and not duplicate
    ):
        status = "VALID"

    else:
        status = "REVIEW"

    # Return validation results
    return {
        "filename": resume.get("filename", ""),
        "email_valid": email_valid,
        "phone_valid": phone_valid,
        "name_valid": name_valid,
        "missing_sections": ", ".join(missing_sections),
        "resume_length_valid": resume_length_valid,
        "duplicate": duplicate,
        "data_quality_score": data_quality_score,
        "status": status
    }