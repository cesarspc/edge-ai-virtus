"""Render this report from main.tex when the editor's LaTeX compiler is unavailable.

This is a deliberately small renderer for this document's sections, tables and
figure. main.tex remains the only source of the report's text and figures.
"""
from pathlib import Path
import html
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, KeepTogether)

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'main.tex').read_text(encoding='utf-8')
for name, filename in [('Report', 'times.ttf'), ('Report-Bold', 'timesbd.ttf'),
                       ('Report-Italic', 'timesi.ttf'), ('Report-BoldItalic', 'timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, Path('C:/Windows/Fonts') / filename))
pdfmetrics.registerFontFamily('Report', normal='Report', bold='Report-Bold',
                              italic='Report-Italic', boldItalic='Report-BoldItalic')

WIDTH = A4[0] - 4.8 * cm
styles = {
    'body': ParagraphStyle('body', fontName='Report', fontSize=10.2, leading=13.2,
                           spaceAfter=6, splitLongWords=False),
    'title': ParagraphStyle('title', fontName='Report-Bold', fontSize=16, leading=20,
                            alignment=TA_CENTER, spaceAfter=12),
    'subtitle': ParagraphStyle('subtitle', fontName='Report', fontSize=11, leading=14,
                               alignment=TA_CENTER, spaceAfter=13),
    'author': ParagraphStyle('author', fontName='Report', fontSize=10.5, leading=14,
                             alignment=TA_CENTER, spaceAfter=16),
    'h1': ParagraphStyle('h1', fontName='Report-Bold', fontSize=13, leading=16,
                         spaceBefore=12, spaceAfter=7, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='Report-Bold', fontSize=11.5, leading=14,
                         spaceBefore=10, spaceAfter=5, keepWithNext=True),
    'caption': ParagraphStyle('caption', fontName='Report', fontSize=9, leading=12,
                              spaceBefore=5, spaceAfter=7, alignment=TA_CENTER),
    'cell': ParagraphStyle('cell', fontName='Report', fontSize=8.7, leading=11),
    'head': ParagraphStyle('head', fontName='Report-Bold', fontSize=8.7, leading=11),
}


def argument(text, command):
    start = text.index('\\' + command + '{') + len(command) + 2
    depth, end = 1, start
    while depth:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    return text[start:end - 1]


labels = {}
for number, match in enumerate(re.finditer(r'\\begin\{table\}(.*?)\\end\{table\}', SOURCE, re.S), 1):
    label = re.search(r'\\label\{([^}]+)\}', match[1])
    if label:
        labels[label[1]] = number


def inline(text):
    text = re.sub(r'\\ref\{([^}]+)\}', lambda m: str(labels[m[1]]), text)
    text = re.sub(r'\\\\(?:\[[^]]*\])?', '<br/>', text)
    text = text.replace('\\normalsize ', '').replace('\\%', '%').replace('\\,', '\u00a0').replace('\\;', ' ')
    text = text.replace('~', '\u00a0').replace('{,}', ',').replace('---', '-').replace('--', '-')
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    text = text.replace('$10^{-4}$', '10<SUP>-4</SUP>').replace('$', '')
    text = html.escape(text).replace('&lt;br/&gt;', '<br/>').replace('&lt;SUP&gt;', '<super>').replace('&lt;/SUP&gt;', '</super>')
    for command, tag in [('textbf', 'b'), ('emph', 'i'), ('texttt', 'font name="Courier"')]:
        closing = tag.split()[0]
        text = re.sub(r'\\' + command + r'\{([^{}]*)\}', lambda m: f'<{tag}>{m[1]}</{closing}>', text)
    if re.search(r'\\[a-zA-Z]+', text):
        raise ValueError('Unsupported LaTeX command: ' + text)
    return re.sub(r'\s*\n\s*', ' ', text).strip()


def p(text, style='body'):
    return Paragraph(inline(text), styles[style])


title = argument(SOURCE, 'title')
title = title.replace('\\normalsize ', '')
split = title.split('\\\\[0.6em]')
story = [p(split[0], 'title'), p(split[1], 'subtitle'), p(argument(SOURCE, 'author'), 'author')]
body = SOURCE.split('\\begin{document}', 1)[1]
lines = body.splitlines()
i, section, subsection, table_number, figure_number = 0, 0, 0, 0, 0

while i < len(lines):
    line = lines[i].strip()
    i += 1
    if not line or line in ('\\maketitle', '\\end{document}'):
        continue
    if line.startswith('\\section{'):
        section += 1
        subsection = 0
        story.append(p(f'{section}. ' + argument(line, 'section'), 'h1'))
    elif line.startswith('\\subsection{'):
        subsection += 1
        story.append(p(f'{section}.{subsection} ' + argument(line, 'subsection'), 'h2'))
    elif line == '\\begin{abstract}':
        block = []
        while lines[i].strip() != '\\end{abstract}':
            block.append(lines[i]); i += 1
        i += 1
        story.extend([p('Resumen', 'h2'), p('\n'.join(block))])
    elif line.startswith('\\begin{table}') or line.startswith('\\begin{figure}'):
        kind = 'table' if line.startswith('\\begin{table}') else 'figure'
        block = []
        while lines[i].strip() != '\\end{' + kind + '}':
            block.append(lines[i]); i += 1
        i += 1
        block = '\n'.join(block)
        caption = argument(block, 'caption')
        if kind == 'table':
            table_number += 1
            raw = re.search(r'\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}', block, re.S)[1]
            raw = re.sub(r'\\(?:toprule|midrule|bottomrule)', '', raw)
            rows = [[cell.strip() for cell in row.split('&')] for row in raw.split('\\\\') if row.strip()]
            n = len(rows[0])
            if 'tab:recortes' in block:
                widths = [WIDTH * q for q in (.21, .35, .22, .22)]
            elif n == 5:
                widths = [WIDTH * q for q in (.22, .15, .18, .23, .22)]
            elif n == 3:
                widths = [WIDTH * q for q in (.54, .23, .23)]
            elif 'tab:particiones' in block:
                widths = [WIDTH * q for q in (.19, .19, .16, .46)]
            else:
                widths = [WIDTH / n] * n
            table = Table([[p(c, 'head' if j == 0 else 'cell') for c in row] for j, row in enumerate(rows)],
                          colWidths=widths, repeatRows=1, hAlign='CENTER')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf0f3')),
                ('LINEABOVE', (0, 0), (-1, 0), .6, colors.black),
                ('LINEBELOW', (0, 0), (-1, 0), .4, colors.black),
                ('LINEBELOW', (0, -1), (-1, -1), .6, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.extend([KeepTogether([p(f'Tabla {table_number}. ' + caption, 'caption'), table]), Spacer(1, 9)])
        else:
            figure_number += 1
            filename = re.search(r'\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}', block)[1]
            path = ROOT / filename
            with PILImage.open(path) as image:
                w, h = image.size
            fraction = re.search(r'width=([0-9.]+)\\linewidth', block)
            width = WIDTH * (float(fraction[1]) if fraction else 1.0)
            story.append(KeepTogether([Image(str(path), width=width, height=width * h / w),
                                        p(f'Figura {figure_number}. ' + caption, 'caption')]))
    elif line in ('\\begin{itemize}', '\\begin{enumerate}'):
        numbered = 'enumerate' in line
        items, current = [], []
        while not lines[i].strip().startswith('\\end{'):
            part = lines[i].strip(); i += 1
            if part.startswith('\\item '):
                if current: items.append(' '.join(current))
                current = [part[6:]]
            else:
                current.append(part)
        if current: items.append(' '.join(current))
        i += 1
        for number, item in enumerate(items, 1):
            story.append(p((str(number) + '. ' if numbered else '• ') + item))
    else:
        block = [line]
        while (i < len(lines) and lines[i].strip()
               and not re.match(r'\\(?:begin|end|section|subsection|maketitle)\b', lines[i].strip())):
            block.append(lines[i].strip()); i += 1
        story.append(p('\n'.join(block)))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Report', 8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(2.4 * cm, 1.2 * cm, 'Semanas 7 y 8 · FungiTastic v002')
    canvas.drawRightString(A4[0] - 2.4 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


if __name__ == '__main__':
    document = SimpleDocTemplate(str(ROOT / 'main.pdf'), pagesize=A4, leftMargin=2.4 * cm,
                                 rightMargin=2.4 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                                 title='Resultados de las semanas 7 y 8',
                                 author='Daniel Esteban Camacho Ospina; Cesar Augusto Pulido Cuervo')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(ROOT / 'main.pdf')
