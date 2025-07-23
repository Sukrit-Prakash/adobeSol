from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", size=24, style="B")
pdf.cell(200, 10, txt="Understanding AI", ln=True, align="C")

# H1
pdf.set_font("Arial", size=20, style="B")
pdf.cell(200, 10, txt="Introduction", ln=True)

# H2
pdf.set_font("Arial", size=16, style="B")
pdf.cell(200, 10, txt="What is AI?", ln=True)

# H3
pdf.set_font("Arial", size=13, style="B")
pdf.cell(200, 10, txt="History of AI", ln=True)

# Normal Text
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt=(
    "Artificial Intelligence (AI) is the simulation of human intelligence processes "
    "by machines, especially computer systems."
))

pdf.output("sample_input.pdf")
