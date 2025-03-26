from src.reporting.pdf_reporter import PDFReporter

def generate_pdf_report():
    try:
        print("Attempting to generate PDF report...")
        reporter = PDFReporter()
        reporter.generate_report()
        print("PDF generation completed. Check reports/ directory for llm_report.pdf")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    generate_pdf_report()
