import os
import sys

def check_pdf_requirements():
    # Check font file
    font_path = "src/fonts/DejaVuSans.ttf"
    font_exists = os.path.exists(font_path)
    print(f"Font exists: {font_exists}")
    
    # Check reports directory
    reports_dir = "reports"
    reports_exists = os.path.exists(reports_dir)
    print(f"Reports directory exists: {reports_exists}")
    
    # Check write permissions
    if reports_exists:
        test_file = os.path.join(reports_dir, "test.txt")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("Reports directory is writable")
        except Exception as e:
            print(f"Reports directory not writable: {e}")
    else:
        print("Creating reports directory...")
        try:
            os.makedirs(reports_dir)
            print("Reports directory created")
        except Exception as e:
            print(f"Failed to create reports directory: {e}")

if __name__ == "__main__":
    check_pdf_requirements()
