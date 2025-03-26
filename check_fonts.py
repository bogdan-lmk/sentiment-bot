from fpdf import FPDF
import sys

def main():
    with open('font_check.txt', 'w') as f:
        # Try different font approaches
        try:
            # Approach 1: Try built-in font substitution
            pdf1 = FPDF()
            pdf1.add_page()
            pdf1.set_font("helvetica", size=12)
            try:
                pdf1.cell(200, 10, text="Тест кириллицы", new_x="LMARGIN", new_y="NEXT")
                f.write("Built-in font substitution works\n")
            except:
                f.write("Built-in fonts don't support Cyrillic\n")
             
            # Approach 2: Try DejaVuSans if available
            try:
                pdf2 = FPDF()
                pdf2.add_page()
                pdf2.add_font('DejaVuSans', '', 'DejaVuSans.ttf', uni=True)
                pdf2.set_font('DejaVuSans', size=12)
                pdf2.cell(200, 10, text="Тест кириллицы", new_x="LMARGIN", new_y="NEXT")
                f.write("DejaVuSans works (font file present)\n")
            except Exception as e:
                f.write(f"DejaVuSans failed: {str(e)}\n")
             
            # Approach 3: Try Arial Unicode MS if available
            try:
                pdf3 = FPDF()
                pdf3.add_page()
                pdf3.add_font('ArialUnicodeMS', '', 'arialuni.ttf', uni=True)
                pdf3.set_font('ArialUnicodeMS', size=12)
                pdf3.cell(200, 10, text="Тест кириллицы", new_x="LMARGIN", new_y="NEXT")
                f.write("ArialUnicodeMS works (font file present)\n")
            except Exception as e:
                f.write(f"ArialUnicodeMS failed: {str(e)}\n")
             
            f.write("\nSolution: Install a Unicode font like DejaVu or Arial Unicode MS\n")
        except Exception as e:
            f.write(f"Critical error: {str(e)}\n")

if __name__ == "__main__":
    main()