from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black

def create_sample_pdf(filename="sample_document.pdf", title_text="Sample Document Title"):
    """
    Generates a sample PDF with various heading levels and body text.

    Args:
        filename (str): The name of the output PDF file.
        title_text (str): The main title for the document.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles for headings - MODIFYING existing styles instead of adding new ones
    # H1 Style
    # Check if 'Heading1' exists, if not, add it. Otherwise, modify.
    if 'Heading1' not in styles:
        styles.add(ParagraphStyle(name='Heading1'))
    styles['Heading1'].fontSize = 24
    styles['Heading1'].leading = 28
    styles['Heading1'].alignment = TA_CENTER
    styles['Heading1'].spaceAfter = 14
    styles['Heading1'].fontName = 'Helvetica-Bold'
    styles['Heading1'].textColor = black

    # H2 Style
    if 'Heading2' not in styles:
        styles.add(ParagraphStyle(name='Heading2'))
    styles['Heading2'].fontSize = 18
    styles['Heading2'].leading = 22
    styles['Heading2'].alignment = TA_LEFT
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 8
    styles['Heading2'].fontName = 'Helvetica-Bold'
    styles['Heading2'].textColor = black

    # H3 Style
    if 'Heading3' not in styles:
        styles.add(ParagraphStyle(name='Heading3'))
    styles['Heading3'].fontSize = 14
    styles['Heading3'].leading = 18
    styles['Heading3'].alignment = TA_LEFT
    styles['Heading3'].spaceBefore = 10
    styles['Heading3'].spaceAfter = 6
    styles['Heading3'].fontName = 'Helvetica-Bold'
    styles['Heading3'].textColor = black

    # Body Text Style
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText'))
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 12
    styles['BodyText'].alignment = TA_LEFT
    styles['BodyText'].spaceAfter = 6
    styles['BodyText'].fontName = 'Helvetica'
    styles['BodyText'].textColor = black


    # Add Title
    story.append(Paragraph(title_text, styles['Heading1']))
    story.append(Spacer(1, 0.2 * letter[1])) # Add some space

    # Section 1
    story.append(Paragraph("1. Introduction to the Document", styles['Heading2']))
    story.append(Paragraph("This is the introductory paragraph for the first main section. It provides an overview of the content that will be covered in this part of the document. We will discuss various aspects and concepts here.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Subsection 1.1
    story.append(Paragraph("1.1. Purpose and Scope", styles['Heading3']))
    story.append(Paragraph("The purpose of this subsection is to define the objectives and boundaries of the document. It clarifies what the document aims to achieve and what topics are included or excluded.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Subsection 1.2
    story.append(Paragraph("1.2. Document Structure", styles['Heading3']))
    story.append(Paragraph("This part describes how the document is organized, outlining the major sections and their logical flow. Understanding the structure helps readers navigate the content more effectively.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Section 2
    story.append(Paragraph("2. Core Concepts and Methodology", styles['Heading2']))
    story.append(Paragraph("In this section, we delve into the fundamental concepts and the methodology employed. Detailed explanations and examples are provided to ensure a clear understanding of the subject matter.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Subsection 2.1
    story.append(Paragraph("2.1. Key Principles", styles['Heading3']))
    story.append(Paragraph("Here, we outline the essential principles that underpin the entire framework. Each principle is briefly explained to highlight its significance.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Subsection 2.2
    story.append(Paragraph("2.2. Implementation Details", styles['Heading3']))
    story.append(Paragraph("This subsection covers the practical aspects of implementing the concepts discussed. It includes steps, considerations, and potential challenges.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Section 3
    story.append(Paragraph("3. Conclusion and Future Work", styles['Heading2']))
    story.append(Paragraph("This final section summarizes the main findings and discusses potential avenues for future research or development. It provides a forward-looking perspective.", styles['BodyText']))
    story.append(Spacer(1, 0.1 * letter[1]))

    # Build the PDF
    doc.build(story)
    print(f"Generated {filename}")

if __name__ == "__main__":
    # Generate 5 sample PDFs
    for i in range(1, 6):
        create_sample_pdf(f"sample_document_{i}.pdf", f"Sample Document {i} Title")

    print("\nAll 5 sample PDFs generated in the current directory.")
