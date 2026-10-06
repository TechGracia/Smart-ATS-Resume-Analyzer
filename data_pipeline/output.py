import os
import pandas as pd


# ---------------------------------------------------------
# SAVE PROCESSED RESUME DATA
# ---------------------------------------------------------

def save_processed_resumes(resumes, output_folder="output"):
    """
    Save structured and validated resume data to CSV.
    """

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(
        output_folder,
        "processed_resumes.csv"
    )

    df = pd.DataFrame(resumes)

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nProcessed resume data saved to:")
    print(output_file)

    return output_file


# ---------------------------------------------------------
# SAVE VALIDATION REPORT
# ---------------------------------------------------------

def save_validation_report(resumes, output_folder="output"):
    """
    Save only validation-related information to CSV.
    """

    os.makedirs(output_folder, exist_ok=True)

    validation_data = []

    for resume in resumes:

        validation_data.append({
            "filename": resume.get("filename", ""),
            "name": resume.get("name", ""),
            "email": resume.get("email", ""),
            "phone": resume.get("phone", ""),
            "email_valid": resume.get("email_valid", False),
            "phone_valid": resume.get("phone_valid", False),
            "name_valid": resume.get("name_valid", False),
            "missing_sections": resume.get(
                "missing_sections",
                ""
            ),
            "resume_length_valid": resume.get(
                "resume_length_valid",
                False
            ),
            "duplicate": resume.get(
                "duplicate",
                False
            ),
            "status": resume.get(
                "status",
                "REVIEW"
            )
        })

    df = pd.DataFrame(validation_data)

    output_file = os.path.join(
        output_folder,
        "validation_report.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Validation report saved to:")
    print(output_file)

    return output_file