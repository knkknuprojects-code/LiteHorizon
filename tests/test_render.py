"""Tests for the shared pandoc recipe.

Every option in it was found by a build that succeeded and was wrong, so the point of
these is that none of them quietly goes missing — and that one recipe serves the report,
the commentary and the annotated document, which is why it is in the package at all.
"""

from pathlib import Path

from horizon.render import pandoc_command


def test_the_resource_path_is_the_markdown_directory():
    """Images are written relative to the report; pandoc resolves relative to the cwd.
    Without this the PDF builds, keeps its page count, and contains no images."""
    cmd = pandoc_command(
        Path("reports/run/x-ua.md"), Path("reports/run/x-ua.pdf"), pdf=True
    )
    assert "--resource-path=reports/run" in cmd


def test_the_pdf_recipe_carries_every_load_bearing_option():
    cmd = " ".join(pandoc_command(Path("x.md"), Path("x.pdf"), pdf=True))
    assert "--pdf-engine=xelatex" in cmd
    assert "Arial Unicode MS" in cmd and "Menlo" in cmd
    assert r"define@key{Gin}{alt}" in cmd  # or pandoc 3.x produces no PDF at all
    assert (
        r"floatplacement{figure}{H}" in cmd
    )  # or a figure drifts into the next section
    assert "labelformat=empty" in cmd  # or every caption carries two numbers


def test_docx_takes_none_of_the_latex_options():
    """It embeds no fonts and lays nothing out; both are the reader's Word."""
    cmd = " ".join(pandoc_command(Path("x.md"), Path("x.docx"), pdf=False))
    assert "xelatex" not in cmd and "header-includes" not in cmd
    assert "--resource-path" in cmd


def test_both_formats_get_a_contents_list():
    """In DOCX it is a Word field with no cached result, so Word asks to update it on
    open — the price of having one, and worth paying: a 145-page document with no way
    into it is one nobody navigates."""
    for pdf in (True, False):
        cmd = pandoc_command(Path("x.md"), Path("x.out"), pdf=pdf)
        assert "--toc" in cmd
        assert "--toc-depth=2" in cmd


def test_the_contents_is_titled_in_the_documents_language():
    """LaTeX would take it from `lang`; the DOCX writer would not, so it is set."""
    uk = " ".join(pandoc_command(Path("x.md"), Path("x.pdf"), pdf=True, lang="uk"))
    en = " ".join(pandoc_command(Path("x.md"), Path("x.pdf"), pdf=True, lang="en"))
    assert "toc-title=Зміст" in uk and "lang=uk" in uk
    assert "toc-title=Contents" in en and "lang=en" in en


def test_the_depth_stops_above_the_topic_labels():
    """Two takes in the subject areas and the indicators — what a reader looks up by.
    Three would add every topic in every table."""
    from horizon.render import TOC_DEPTH

    assert TOC_DEPTH == 2


def test_the_pdf_carries_an_outline_pane():
    """The contents list is a page you turn to; the outline is the reader's sidebar, and
    a document of 145 pages wants both. hyperref alone left /Outlines out of the file.
    """
    cmd = " ".join(pandoc_command(Path("x.md"), Path("x.pdf"), pdf=True))
    assert "bookmark" in cmd


def test_the_pdf_is_set_in_the_size_the_module_declares():
    """LaTeX takes it as a class option and scales the headings with it."""
    from horizon.render import FONT_SIZE_PT

    cmd = " ".join(pandoc_command(Path("x.md"), Path("x.pdf"), pdf=True))
    assert f"fontsize={FONT_SIZE_PT}pt" in cmd


def test_the_docx_is_set_through_a_reference_stylesheet():
    """DOCX has no size variable at all, so the only way in is the stylesheet."""
    cmd = " ".join(
        pandoc_command(
            Path("x.md"), Path("x.docx"), pdf=False, reference_doc=Path("ref.docx")
        )
    )
    assert "--reference-doc=ref.docx" in cmd


def test_the_pdf_path_ignores_the_reference_document():
    """It is a DOCX stylesheet; pandoc would reject it on a LaTeX build."""
    cmd = " ".join(
        pandoc_command(
            Path("x.md"), Path("x.pdf"), pdf=True, reference_doc=Path("ref.docx")
        )
    )
    assert "--reference-doc" not in cmd


def test_rescaling_a_stylesheet_keeps_the_headings_in_proportion():
    """Scaling only the body would leave a 20pt heading over 10pt text.

    Sizes are half-points, and each is read against the body size the stylesheet itself
    declares — so this holds whatever pandoc's own default happens to be.
    """
    from horizon.render import _scale_sizes

    xml = (
        '<w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="24" />'
        '<w:szCs w:val="24" /></w:rPr></w:rPrDefault></w:docDefaults>'
        '<w:style w:styleId="Heading1"><w:sz w:val="40" /></w:style>'
        '<w:style w:styleId="Title"><w:sz w:val="56" /></w:style>'
    )
    out = _scale_sizes(xml, 10)
    assert 'w:val="20"' in out  # 12pt body -> 10pt
    assert 'w:val="34"' in out  # 20pt heading -> 17pt
    assert 'w:val="46"' in out  # 28pt title -> 23pt


def test_a_stylesheet_with_no_declared_body_size_is_left_alone():
    """Rescaling against a guess would be worse than not rescaling."""
    from horizon.render import _scale_sizes

    xml = '<w:style w:styleId="Heading1"><w:sz w:val="40" /></w:style>'
    assert _scale_sizes(xml, 10) == xml
