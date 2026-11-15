"""Read all 17 year-label centers from the original report's PDF text geometry."""

import hashlib
import json
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF_HASH = "187312710f7684c0f2e3e0becc4e92512c2ec194727dde53e96a5924c0571185"


def main():
    pdf, = Path("C:/Users/Administrator/Downloads").glob("2024-12-28*国盛*中国经济六周期模型*.pdf")
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == PDF_HASH
    with pdfplumber.open(pdf) as report:
        page = report.pages[14]
        characters = sorted((c for c in page.chars if 353 < c["top"] < 355 and c["text"].isdigit()),
                            key=lambda c: c["x0"])
    assert len(characters) == 17 * 4
    positions = []
    for offset in range(0, len(characters), 4):
        group = characters[offset:offset + 4]
        year = int("".join(c["text"] for c in group))
        assert year == 2008 + offset // 4
        left = min(c["x0"] for c in group)
        right = max(c["x1"] for c in group)
        center = (left + right) / 2
        positions.append(dict(year=year, x0_pt=left, x1_pt=right, center_pt=center,
                              crop_x_px=center * 400 / 72 - 270))
    receipt = dict(pdf=str(pdf), pdf_sha256=PDF_HASH, page=15, dpi=400, crop_x=270,
                   method="PDF glyph bounding boxes; group four horizontally adjacent year digits on chart x axis",
                   year_x=positions)
    output = HERE / "phase_2024_calibration.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(positions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
