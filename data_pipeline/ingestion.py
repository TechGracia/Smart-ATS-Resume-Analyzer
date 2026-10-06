import os
import pdfplumber


def ingest_resumes(input_folder="input_resumes"):
    """
    Read all PDF resumes from the input folder.

    Returns:
        list of dictionaries containing filename and extracted text.
    """

    resumes = []

    if not os.path.exists(input_folder):
        print(f"ERROR: Folder '{input_folder}' does not exist.")
        return resumes

    pdf_files = [
        file for file in os.listdir(input_folder)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF resumes found.")
        return resumes

    for filename in pdf_files:
        file_path = os.path.join(input_folder, filename)

        try:
            text = ""

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

            text = text.strip()

            if not text:
                print(f"WARNING: No text extracted from {filename}")

            resumes.append({
                "filename": filename,
                "text": text
            })

            print(f"Processed: {filename}")

        except Exception as error:
            print(f"ERROR processing {filename}: {error}")

            resumes.append({
                "filename": filename,
                "text": "",
                "error": str(error)
            })

    print(f"\nTotal resumes processed: {len(resumes)}")

    return resumes
if __name__ == "__main__":
    resumes = ingest_resumes()

    for resume in resumes:
        print("\n-----------------------------")
        print("File:", resume["filename"])
        print("Characters extracted:", len(resume["text"]))