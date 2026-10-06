import re


def clean_text(text):
    """
    Clean and normalize extracted resume text.
    """

    if not text:
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Replace multiple spaces and tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)

    # Remove unnecessary special characters
    text = re.sub(r"[^\w\s@.+#&/-]", "", text)

    # Remove spaces at the beginning and end of each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Join everything back together
    cleaned_text = "\n".join(lines)

    return cleaned_text


if __name__ == "__main__":

    # Small test
    sample_text = """
    JOHN SHARMA


    Email : JOHN@GMAIL.COM


    Python     SQL       Java
    """

    print("BEFORE CLEANING:")
    print(sample_text)

    cleaned = clean_text(sample_text)

    print("\nAFTER CLEANING:")
    print(cleaned)