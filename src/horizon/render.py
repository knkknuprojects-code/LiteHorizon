"""Rendering a report's markdown to the copies people actually read.

In the package rather than in ``cli`` because two other things render the same documents
— the commentary and the annotated report, both built by scripts — and a recipe kept in
two places drifts. It did: the figures in the annotated PDF floated into the wrong
sections and carried a duplicate number for a full day after the report's own PDF had
been fixed, because the script repeated the flags by hand and the fix went into the CLI.

Every option below is load-bearing, and each was found by a build that succeeded and was
wrong:

* ``--resource-path`` — images are written relative to the report and pandoc resolves
  them relative to the working directory. Without it the PDF builds, keeps its page
  count, and contains no images at all;
* the ``Gin``/``alt`` no-op — pandoc 3.x passes an image's alt text to
  ``\\includegraphics`` as ``alt=``, which graphicx does not know: "Package keyval Error:
  alt undefined", and no PDF;
* ``float`` with ``[H]`` — a figure is a float, so LaTeX carries it to wherever it fits;
  measured, the co-occurrence matrix landed three pages into the next section, and a
  figure that has drifted away from the table it illustrates has lost its reason to be
  there;
* ``captionsetup{labelformat=empty}`` — the caption already reads "Діаграма 5. …",
  numbered by the figure's place in the catalogue. LaTeX's own "Рис. N:" is a second,
  different number for the same picture;
* the fonts — Arial Unicode MS carries Cyrillic and the arrow that Helvetica Neue lacks,
  Menlo the box-drawing characters used in diagrams.;
* the body size — one number, two mechanisms. LaTeX takes ``fontsize`` as a class option
  and scales the headings with it; DOCX has no such variable, and its sizes come from a
  reference stylesheet, which is why one is built and rescaled here.

Warnings are surfaced even on a successful build. Discarding them is how a report shipped
with seven figure references and no figures: pandoc had said so.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: Body text size, in points, for both rendered copies.
#:
#: Applied through two different mechanisms because the formats have nothing in common
#: here: LaTeX takes a class option and scales every heading with it, while DOCX has no
#: such knob at all and takes its sizes from a reference document's stylesheet. Kept as
#: one number so the two cannot be changed apart — a PDF and a DOCX of the same report
#: set in different sizes is the kind of difference nobody sees until both are on a desk.
FONT_SIZE_PT = 10

#: LaTeX preamble additions, as one ``header-includes`` value.
HEADER_INCLUDES = (
    r"\makeatletter\define@key{Gin}{alt}{}\makeatother"
    r"\usepackage{float}\floatplacement{figure}{H}"
    r"\usepackage{caption}\captionsetup[figure]{labelformat=empty}"
    # The outline pane, which hyperref alone did not produce: the file came out with no
    # /Outlines at all. A contents list is a page you turn to; the outline is the
    # sidebar, and a document of 145 pages wants both.
    r"\usepackage{bookmark}"
)

#: pandoc ``-V`` settings for the PDF path. DOCX needs none of them: it embeds no fonts
#: and lays nothing out, leaving both to the reader's Word.
PDF_VARIABLES = (
    ("fontsize", f"{FONT_SIZE_PT}pt"),
    ("mainfont", "Arial Unicode MS"),
    ("monofont", "Menlo"),
    ("geometry:margin", "2cm"),
    ("colorlinks", "true"),
    ("header-includes", HEADER_INCLUDES),
)

#: What the table of contents is called, per language. pandoc's default is English, and
#: LaTeX would take it from ``lang`` while the DOCX writer would not — so it is set
#: explicitly and both agree.
TOC_TITLE = {"uk": "Зміст", "en": "Contents"}

#: Headings deep enough to list. Two takes in the subject areas of section 3 and the
#: indicators of section 5 — which is the point of having one, since those are what a
#: reader looks something up by. Three would add every topic label.
TOC_DEPTH = 2


def _scale_sizes(styles_xml: str, size_pt: int) -> str:
    """Rescale every font size in a reference stylesheet to a ``size_pt`` body.

    Sizes are in half-points, and each is scaled against the body size the stylesheet
    itself declares rather than against an assumed default — so headings, the title, the
    abstract and the code face keep their proportions instead of being flattened to one
    size. Rounded to whole points, which is what the reader's Word offers.
    """
    base = re.search(r"<w:docDefaults>.*?<w:sz w:val=\"(\d+)\"", styles_xml, re.S)
    if not base:
        return styles_xml
    body = int(base.group(1))

    def scaled(m: "re.Match[str]") -> str:
        points = max(1, round(int(m.group(2)) * size_pt / body))
        return f'{m.group(1)} w:val="{points * 2}"'

    return re.sub(r'(<w:sz(?:Cs)?) w:val="(\d+)"', scaled, styles_xml)


@lru_cache(maxsize=None)
def _reference_doc(size_pt: int) -> Path | None:
    """pandoc's own reference.docx with its stylesheet rescaled, or ``None``.

    Built from ``--print-default-data-file`` rather than kept in the repo as a binary:
    the file encodes pandoc's whole DOCX stylesheet, and a copy committed here would go
    on being used long after the pandoc that produced it was replaced — silently, since
    a stale reference document renders without complaint.

    Cached in the temporary directory under a name carrying the size and the pandoc
    version, so an upgraded pandoc builds a new one and an unchanged one is built once
    per machine. Keyed on the version rather than on a digest of the file: pandoc stamps
    the current time into the archive it prints, so the bytes differ between two calls a
    second apart and a content-addressed name would rebuild on every render and leave a
    new file behind each time.
    """
    import re as _re
    import shutil
    import subprocess
    import tempfile
    import zipfile

    if not shutil.which("pandoc"):
        return None
    version = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
    if version.returncode != 0:
        return None
    tag = _re.sub(r"[^0-9a-zA-Z.]+", "-", version.stdout.splitlines()[0].strip())
    out = Path(tempfile.gettempdir()) / f"horizon-reference-{size_pt}pt-{tag}.docx"
    if out.exists():
        return out
    done = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True
    )
    if done.returncode != 0 or not done.stdout:
        return None
    source = out.with_suffix(".source.docx")
    source.write_bytes(done.stdout)
    tmp = out.with_suffix(".partial.docx")
    try:
        with (
            zipfile.ZipFile(source) as zin,
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout,
        ):
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/styles.xml":
                    data = _scale_sizes(data.decode("utf-8"), size_pt).encode("utf-8")
                zout.writestr(item, data)
        tmp.replace(out)
    finally:
        source.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return out


def pandoc_command(
    md_path: Path,
    out_path: Path,
    *,
    pdf: bool,
    lang: str = "uk",
    reference_doc: Path | None = None,
) -> list[str]:
    """The full pandoc invocation for one output, as an argv list.

    ``reference_doc`` is the DOCX stylesheet to render against — how the body size is set
    for that format, since it has no variable for one. Ignored on the PDF path, which
    takes its size as a class option instead.
    """
    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(out_path),
        f"--resource-path={md_path.parent}",
        "-V",
        f"lang={lang}",
    ]
    # Both formats get a contents list. In DOCX it is a Word TOC *field*, and pandoc
    # writes the field without a cached result — the file says "build a contents list"
    # and holds none — so Word has to update it to show anything, and updating fields is
    # what raises "the document contains fields that refer to other files" on open. The
    # warning is Word's blanket one about fields; ours is a plain TOC with no external
    # target, confirmed by there being no external relationship in the package. Answering
    # Yes fills it in. A Word-authored TOC ships its rendered result alongside the field,
    # which is why those open silently; pandoc cannot do that, because the page numbers
    # are decided by Word's own layout.
    cmd += [
        "--toc",
        f"--toc-depth={TOC_DEPTH}",
        "-V",
        f"toc-title={TOC_TITLE.get(lang, TOC_TITLE['en'])}",
    ]
    if pdf:
        cmd.append("--pdf-engine=xelatex")
        for key, value in PDF_VARIABLES:
            cmd += ["-V", f"{key}={value}"]
    elif reference_doc is not None:
        cmd.append(f"--reference-doc={reference_doc}")
    return cmd


def render(md_path: Path, out_path: Path, *, pdf: bool, lang: str = "uk") -> str:
    """Render ``md_path`` with pandoc; returns a status line for the caller to print.

    Not a hard dependency: a missing pandoc costs the rendered copies, not the report.
    """
    import shutil
    import subprocess

    if not shutil.which("pandoc"):
        return f"  skipped {out_path.name} (pandoc not installed)"
    reference = None if pdf else _reference_doc(FONT_SIZE_PT)
    done = subprocess.run(
        pandoc_command(
            Path(md_path),
            Path(out_path),
            pdf=pdf,
            lang=lang,
            reference_doc=reference,
        ),
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        return f"  FAILED {out_path.name}: {done.stderr.strip()[:160]}"
    warnings = [w for w in done.stderr.splitlines() if w.strip()]
    if not pdf and reference is None:
        # Said out loud: the DOCX is readable either way, but it would be set in
        # pandoc's default size while the PDF beside it is in ours, and nobody notices
        # that until the two are on a desk together.
        warnings.append(
            f"could not build the {FONT_SIZE_PT}pt stylesheet; "
            "this DOCX uses pandoc's default size"
        )
    line = f"  wrote {out_path}"
    if warnings:
        line += "\n" + "\n".join(f"    {w}" for w in warnings[:5])
        if len(warnings) > 5:
            line += f"\n    ... and {len(warnings) - 5} more"
    return line
