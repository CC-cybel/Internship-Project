import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem

def register_fonts():
    # Register Chinese font
    # Try multiple common locations for WenQuanYi Zen Hei or similar
    possible_fonts = [
        '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/cjkuni-uming/uming.ttc',
        os.path.expanduser('~/.fonts/wqy-zenhei.ttc'),
        '/data1/yezj/gitlab/leadbench/wqy-zenhei.ttc' # Just in case
    ]
    
    font_name = 'Helvetica' # Default fallback
    
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('ZenHei', font_path))
                font_name = 'ZenHei'
                print(f"Registered font: {font_path}")
                break
            except Exception as e:
                print(f"Failed to register font {font_path}: {e}")
                continue
                
    return font_name

def generate_pdf(md_file_path, output_pdf_path):
    font_name = register_fonts()
    
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    # Define custom styles with Chinese font support
    styles.add(ParagraphStyle(name='ChineseNormal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='ChineseH1', parent=styles['Heading1'], fontName=font_name, fontSize=18, leading=22, spaceAfter=12, spaceBefore=12, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='ChineseH2', parent=styles['Heading2'], fontName=font_name, fontSize=14, leading=18, spaceAfter=10, spaceBefore=10, textColor=colors.black))
    styles.add(ParagraphStyle(name='ChineseH3', parent=styles['Heading3'], fontName=font_name, fontSize=12, leading=16, spaceAfter=8, spaceBefore=8, textColor=colors.black))
    styles.add(ParagraphStyle(name='ChineseBullet', parent=styles['Bullet'], fontName=font_name, fontSize=10, leading=14, leftIndent=20))
    
    story = []
    
    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    base_dir = os.path.dirname(md_file_path)
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        # Headers
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['ChineseH1']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['ChineseH2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['ChineseH3']))
        
        # Images: ![alt](path)
        elif line.startswith('![') and '](' in line:
            match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if match:
                img_rel_path = match.group(2)
                full_img_path = os.path.join(base_dir, img_rel_path)
                if os.path.exists(full_img_path):
                    try:
                        # Create Image flowable
                        img = Image(full_img_path)
                        # Resize logic: max width 450
                        max_width = 450
                        if img.imageWidth > max_width:
                            ratio = max_width / img.imageWidth
                            img.drawWidth = max_width
                            img.drawHeight = img.imageHeight * ratio
                        else:
                            img.drawWidth = img.imageWidth
                            img.drawHeight = img.imageHeight
                            
                        story.append(img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        print(f"Error loading image {full_img_path}: {e}")
                        story.append(Paragraph(f"[Image: {img_rel_path}]", styles['ChineseNormal']))
                else:
                    story.append(Paragraph(f"[Image not found: {img_rel_path}]", styles['ChineseNormal']))
        
        # Tables
        elif line.startswith('|'):
            # Collect table lines
            table_lines = []
            # Look ahead to collect all table rows
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1 # Adjust index as loop will increment
            
            if len(table_lines) >= 2:
                # Parse markdown table
                # Row 0: Headers
                # Row 1: Separator (ignore content, used for alignment if sophisticated)
                # Row 2+: Data
                
                def parse_row(row_str):
                    # Remove leading/trailing pipes if present
                    content = row_str.strip('|')
                    cells = [c.strip() for c in content.split('|')]
                    return cells

                headers = parse_row(table_lines[0])
                data_rows = [parse_row(row) for row in table_lines[2:]]
                
                # Build ReportLab table data
                # All cells must be Flowables (Paragraphs) to support wrapping and Chinese font
                
                table_data = []
                # Header row
                # Clean up headers: replace <br> with <br/>, handle bolding
                header_row_flowables = []
                for h in headers:
                    # Process markdown bolding first
                    h_fmt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', h)
                    # Replace <br> with <br/>
                    h_fmt = h_fmt.replace('<br>', '<br/>')
                    # Wrap in bold if not already (headers usually bold)
                    # But if we just wrap everything in <b>, it's safer
                    header_row_flowables.append(Paragraph(f"<b>{h_fmt}</b>", styles['ChineseNormal']))
                
                table_data.append(header_row_flowables)
                
                for row in data_rows:
                    row_flowables = []
                    # Handle varying column counts
                    for cell in row:
                        # Convert markdown bold to HTML bold tag reportlab understands
                        cell_formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cell)
                        # Replace <br> with <br/> tag or just handle as Paragraph
                        cell_formatted = cell_formatted.replace('<br>', '<br/>')
                        row_flowables.append(Paragraph(cell_formatted, styles['ChineseNormal']))
                    
                    # Pad row if shorter than headers
                    while len(row_flowables) < len(headers):
                        row_flowables.append(Paragraph("", styles['ChineseNormal']))
                        
                    table_data.append(row_flowables)
                
                if table_data:
                    # Create Table
                    # Determine column widths: distribute evenly or based on content?
                    # Let's use auto (None) or fixed. A4 width ~500pts available.
                    col_width = 450 / len(headers) if headers else 100
                    
                    t = Table(table_data, colWidths=[col_width]*len(headers))
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dddddd')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (-1, -1), font_name),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('padding', (0,0), (-1,-1), 6),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 12))

        # Lists
        elif line.startswith('* ') or line.startswith('- '):
            text = line[2:]
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            # Use Paragraph with bullet char, simpler than ListFlowable for mixed content
            # Indent simulation
            p = Paragraph(f'<bullet>&bull;</bullet> {text}', styles['ChineseBullet'])
            story.append(p)
        
        # Horizontal Rule
        elif line.startswith('---'):
            story.append(Spacer(1, 12))
            # Maybe draw a line? 
            # story.append(Flowable...)
            
        # Normal text
        else:
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(text, styles['ChineseNormal']))
            
        i += 1
        
    try:
        doc.build(story)
        print(f"PDF generated successfully at: {output_pdf_path}")
    except Exception as e:
        print(f"Failed to build PDF: {e}")

if __name__ == "__main__":
    md_path = '/data1/yezj/gitlab/leadbench/output/comparison_report/stage2_evaluation_summary.md'
    pdf_path = '/data1/yezj/gitlab/leadbench/output/comparison_report/stage2_evaluation_report.pdf'
    generate_pdf(md_path, pdf_path)
