from data_pipeline.ingestion import ingest_resumes
from data_pipeline.cleaning import clean_text
from data_pipeline.extraction import transform_resume
from data_pipeline.validation import validate_resume
from data_pipeline.output import (
    save_processed_resumes,
    save_validation_report
)


def run_pipeline():

    print("\n==============================")
    print("RESUME DATA PIPELINE STARTED")
    print("==============================")

    # --------------------------------
    # STEP 1: DATA INGESTION
    # --------------------------------
    resumes = ingest_resumes()

    print(f"\nTotal resumes processed: {len(resumes)}")

    # --------------------------------
    # STEP 2: CLEANING
    # --------------------------------
    for resume in resumes:
        resume["cleaned_text"] = clean_text(resume["text"])

    # --------------------------------
    # STEP 3: TRANSFORMATION
    # --------------------------------
    transformed_resumes = []

    for resume in resumes:

        cleaned_resume = {
            "filename": resume["filename"],
            "cleaned_text": resume["cleaned_text"]
        }

        transformed_data = transform_resume(cleaned_resume)

        transformed_resumes.append(transformed_data)

    print("\n==============================")
    print("TRANSFORMATION COMPLETED")
    print("==============================")

    print(f"Total resumes processed: {len(transformed_resumes)}")

    # --------------------------------
    # STEP 4: DATA VALIDATION
    # --------------------------------
    print("\n==============================")
    print("DATA VALIDATION STARTED")
    print("==============================")

    validated_resumes = []

    # Used for duplicate email detection
    seen_emails = set()

    for resume in transformed_resumes:

        validation_result = validate_resume(
            resume,
            seen_emails
        )

        # Combine transformed data + validation results
        resume.update(validation_result)

        validated_resumes.append(resume)

    # Count validation results
    valid_count = sum(
        1 for resume in validated_resumes
        if resume["status"] == "VALID"
    )

    review_count = sum(
        1 for resume in validated_resumes
        if resume["status"] == "REVIEW"
    )

    print("\n==============================")
    print("DATA VALIDATION COMPLETED")
    print("==============================")

    print(f"Valid resumes: {valid_count}")
    print(f"Resumes requiring review: {review_count}")

    # --------------------------------
    # STEP 5: LOAD / SAVE DATA
    # --------------------------------
    print("\n==============================")
    print("LOADING PROCESSED DATA")
    print("==============================")

    save_processed_resumes(validated_resumes)

    save_validation_report(validated_resumes)

    # --------------------------------
    # STEP 6: PIPELINE SUMMARY
    # --------------------------------
    duplicate_count = sum(
        1 for resume in validated_resumes
        if resume["duplicate"]
    )

    print("\n==============================")
    print("PIPELINE SUMMARY")
    print("==============================")

    print(f"Total Resumes       : {len(validated_resumes)}")
    print(f"Successfully Parsed : {len(validated_resumes)}")
    print(f"Valid Records       : {valid_count}")
    print(f"Review Records      : {review_count}")
    print(f"Duplicates Detected : {duplicate_count}")

    print("\nOutput Files:")
    print("✓ processed_resumes.csv")
    print("✓ validation_report.csv")

    print("\n==============================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================")

    # --------------------------------
    # STEP 7: DISPLAY EACH RESUME
    # --------------------------------
    for resume in validated_resumes:

        print("\n------------------------------")
        print(f"RESUME: {resume['filename']}")
        print("------------------------------")

        print("Name:", resume["name"])
        print("Email:", resume["email"])
        print("Phone:", resume["phone"])
        print("Skills:", resume["skills"])
        print("Education:", resume["education"])
        print("Experience:", resume["experience"])
        print("Projects:", resume["projects"])
        print("Certifications:", resume["certifications"])
        print("Achievements:", resume["achievements"])

        print("\n--- VALIDATION ---")
        print("Email Valid:", resume["email_valid"])
        print("Phone Valid:", resume["phone_valid"])
        print("Name Valid:", resume["name_valid"])
        print("Missing Sections:", resume["missing_sections"])
        print("Resume Length Valid:", resume["resume_length_valid"])
        print("Duplicate:", resume["duplicate"])

        # NEW: show quality score
        print(
            "Data Quality Score:",
            f"{resume['data_quality_score']}/100"
        )

        # NEW: show final status
        print("Overall Status:", resume["status"])

    return validated_resumes


# --------------------------------
# RUN PIPELINE
# --------------------------------
if __name__ == "__main__":
    run_pipeline()