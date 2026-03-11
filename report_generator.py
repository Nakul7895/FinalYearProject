from reportlab.pdfgen import canvas

def create_report(tumor_type, tumor_percentage):

    c = canvas.Canvas("tumor_report.pdf")

    c.drawString(100,750,"Brain Tumor AI Diagnosis")

    c.drawString(100,700,f"Tumor Type: {tumor_type}")
    c.drawString(100,670,f"Tumor Size: {tumor_percentage}%")

    c.save()