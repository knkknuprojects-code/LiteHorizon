"""Tests for the patent pipeline.

Two things are worth testing here and they are not the obvious ones. The parsing is
mostly column plumbing, but two of its decisions were wrong on the first pass and produced
analysis that looked entirely plausible — the list delimiter and the citation join key —
so both are pinned. And the completeness guard is the only genuinely new statistics in the
package, so it is tested against the failure it exists to prevent rather than against a
number it happens to produce.

No workbook is read: :func:`normalize_row` takes the row dict the reader yields, so the
parsing tests build that dict directly and stay fast.
"""

from __future__ import annotations

from datetime import date

import pytest

from horizon.patents.completeness import (
    YEAR_AXES,
    complete_years,
    coverage_at,
    lag_quantiles,
)
from horizon.patents.source import PatentSource, item_text
from horizon.patents.workbooks import (
    MAIN_COLUMNS,
    attach_citation_percentiles,
    attach_satellites,
    normalize_row,
)


def row(**over):
    """A register row as the reader yields it, with everything present by default."""
    base = {
        "patent_number": "UA100101",
        "application_number": "a201200035",
        "filed_at": "2012-01-03",
        "granted_at": "2012-11-12",
        "priority_at": "2012-01-03",
        "priority_year": "2012.0",
        "priority_source": "INID (22)",
        "filing_year": "2012.0",
        "grant_year": "2012.0",
        "lag_years": "0.8597",
        "title": "СПОСІБ РОЗКРИТТЯ РУДНИХ РОДОВИЩ",
        "applicant": "DONETSKSTAL",
        "owner": "DONETSKSTAL",
        "inventors": "Filatov Yurii [UA]; Volovyk Volodymyr [UA]",
        "abstract_lang": "UA",
        "attorney": None,
        "applicant_countries": "UA",
        "origin": "resident",
        "ua_participation": "True",
        "applicant_type": "enterprise",
        "ipc_codes": "E21C41/22; E21C41/26",
        "ipc_subclasses": "E21C",
        "wipo_field_codes": "35",
        "wipo_technologies": "Будівництво",
        "wipo_sectors": "Інші напрями",
        "family_number": "56725442",
        "family_members": "UA100101 C2\nEP4025006 A1",
        "wo_publication": None,
        "is_ip5": None,
        "abstract": "Спосіб розкриття та підземної розробки рудних родовищ.",
    }
    base.update(over)
    return base


class TestParsing:
    def test_the_reader_and_the_normalizer_agree_on_every_column(self):
        """A column added to the map but not read is a field that is silently always
        ``None``; this catches the pair drifting apart."""
        assert set(row()) == set(MAIN_COLUMNS)

    def test_lists_split_on_semicolons_only(self):
        """The bug this pins produced a plausible, wrong analysis: 1 907 records carry a
        WIPO technology whose own name contains a comma, and splitting on comma reported
        the halves as two technologies that co-occur."""
        rec = normalize_row(
            row(wipo_technologies="Електричні машини, апарати та енергетика; Двигуни")
        )
        assert rec["subject_areas"] == [
            "Електричні машини, апарати та енергетика",
            "Двигуни",
        ]

    def test_inventor_names_keep_their_commas(self):
        rec = normalize_row(row(inventors="Ivanov Ivan, Jr. [UA]; Petrov Petro [UA]"))
        assert rec["inventors"] == ["Ivanov Ivan, Jr. [UA]", "Petrov Petro [UA]"]
        assert rec["author_count"] == 2

    def test_family_members_split_on_newlines(self):
        assert normalize_row(row())["family_members"] == ["UA100101 C2", "EP4025006 A1"]

    def test_years_survive_the_export_writing_them_as_floats(self):
        """``int("2012.0")`` raises, so a naive parse loses every year in the file."""
        rec = normalize_row(row())
        assert rec["priority_year"] == 2012 and rec["year"] == 2012

    def test_the_axis_chooses_which_year_lands_on_the_canonical_key(self):
        r = row(priority_year="2012.0", filing_year="2013.0", grant_year="2018.0")
        assert normalize_row(r, axis="priority")["year"] == 2012
        assert normalize_row(r, axis="filing")["year"] == 2013
        assert normalize_row(r, axis="grant")["year"] == 2018

    def test_every_axis_names_a_field_the_record_actually_has(self):
        rec = normalize_row(row())
        assert all(key in rec for key in YEAR_AXES.values())

    def test_a_row_with_neither_title_nor_abstract_is_dropped(self):
        assert normalize_row(row(title=None, abstract=None)) is None

    def test_a_row_missing_only_one_of_them_is_kept_and_flagged(self):
        """0.7% of the register has no title. Those are still grants, and dropping them
        would remove them from every denominator too."""
        assert normalize_row(row(title=None))["flags"] == ["no-title"]
        assert normalize_row(row(abstract=None))["flags"] == ["no-abstract"]

    def test_the_canonical_keys_the_analysis_layer_reads_are_populated(self):
        """The whole design rests on these: the analysis modules read these names and are
        never told they are looking at patents."""
        rec = normalize_row(row())
        assert rec["year"] == 2012
        assert rec["subject_areas"] == ["Будівництво"]
        assert rec["venue"] == "DONETSKSTAL"  # applicant, for the dispersion lens
        assert rec["authors"] and rec["script"] == "cyrillic"

    def test_an_out_of_range_year_is_dropped_rather_than_believed(self):
        rec = normalize_row(row(priority_year="12.0"))
        assert rec["priority_year"] is None
        assert "priority-year-out-of-range" in rec["flags"]


class TestSatelliteJoin:
    def test_wo_joins_on_family_and_ip5_on_patent_number(self):
        """Measured, not assumed: joining the IP5 sheet on family number matches zero
        rows, because it carries no PatBase numbering at all."""
        recs = [normalize_row(row())]
        stats = attach_satellites(
            recs,
            [{"Family Number": "56725442", "Number of Forward Citations": 3}],
            [
                {
                    "Patent Number": "UA100101",
                    "Number of Forward Citations": 4,
                    "Patent Valuation Score Overall": 7.5,
                }
            ],
        )
        assert stats["wo_matched"] == 1 and stats["ip5_matched"] == 1
        # Summed across both sheets: either alone understates the invention's record.
        assert recs[0]["forward_citations"] == 7
        assert recs[0]["valuation"]["overall"] == 7.5

    def test_an_unmatched_satellite_row_is_counted_not_added(self):
        """The satellites enrich records; letting them create rows would replace the
        register with the 2% of it filed internationally."""
        recs = [normalize_row(row())]
        stats = attach_satellites(recs, [{"Family Number": "999"}], [])
        assert len(recs) == 1 and stats["wo_unmatched"] == 1

    def test_offices_come_from_the_family_member_list(self):
        recs = [normalize_row(row())]
        attach_satellites(recs, [], [])
        assert recs[0]["international_offices"] == ["UA", "EP"]


class TestCitationPercentiles:
    def test_records_without_a_count_stay_unknown(self):
        """98% of the register has no international sibling and so no citation count.
        Ranking that as a zero percentile would put the whole register below its own
        international tail — absent means unmeasured, not uncited."""
        recs = [normalize_row(row()) for _ in range(3)]
        recs[0]["forward_citations"] = 5
        attach_citation_percentiles(recs)
        assert recs[0]["citation_percentile"] == 0.5  # alone in its cohort
        assert recs[1]["citation_percentile"] is None

    def test_ranking_is_within_the_cohort_not_across_the_register(self):
        """A 2013 patent has had a decade to be cited and a 2020 one has not."""
        recs = []
        for year, cites in ((2013, 1), (2013, 9), (2020, 1), (2020, 9)):
            rec = normalize_row(row())
            rec["year"], rec["forward_citations"] = year, cites
            recs.append(rec)
        attach_citation_percentiles(recs)
        assert [r["citation_percentile"] for r in recs] == [0.0, 1.0, 0.0, 1.0]


class TestCompleteness:
    """The guard exists because grant lag truncates a *run* of cohorts, not one."""

    @staticmethod
    def register(sizes: dict[int, int], lag_years: float = 3.0):
        """A register whose cohorts are the given sizes, every grant at one lag."""
        out = []
        for year, n in sizes.items():
            for _ in range(n):
                rec = normalize_row(row())
                rec["year"] = rec["priority_year"] = year
                rec["lag_years"] = lag_years
                rec["granted_at"] = f"{year + int(lag_years)}-06-01"
                out.append(rec)
        return out

    def test_cohorts_younger_than_the_lag_are_trimmed(self):
        """With every grant taking 3 years and the register cut in 2026, the 2024 and
        2025 cohorts cannot be complete however many rows they happen to show."""
        recs = self.register({y: 300 for y in range(2015, 2026)})
        keep, est = complete_years(recs, cut=date(2026, 8, 12))
        assert max(keep) == 2023
        assert est[2024] == 0.0 and est[2023] == 1.0

    def test_only_a_trailing_run_is_trimmed(self):
        """A gap in the middle is a fact about the register; removing interior cohorts
        would reshape the series into one no data supports."""
        sizes = {y: 300 for y in range(2015, 2024)}
        sizes[2019] = 1
        keep, _ = complete_years(self.register(sizes), cut=date(2026, 8, 12))
        assert 2019 in keep

    def test_the_reference_distribution_ignores_the_truncated_cohorts(self):
        """The defect being corrected: a recent cohort's observed lags are short *because*
        it is truncated, so pooling them in shortens the yardstick and makes an incomplete
        cohort measure as complete. Here the old cohorts take 5 years; the recent one has
        only had time to show its 1-year grants. Pooled, that pulls the distribution short
        enough to call 2024 finished."""
        recs = self.register({y: 400 for y in range(2012, 2020)}, lag_years=5.0)
        recs += self.register({2024: 400}, lag_years=1.0)
        keep, est = complete_years(recs, cut=date(2026, 8, 12))
        assert est[2024] == 0.0 and 2024 not in keep

    def test_grant_year_needs_no_lag_correction(self):
        """A patent is in the register the moment it is granted, so the only incomplete
        grant cohort is the one the cut date falls inside."""
        recs = self.register({y: 300 for y in range(2015, 2027)})
        for rec in recs:
            rec["grant_year"] = rec["priority_year"]
        _, est = complete_years(recs, axis="grant", cut=date(2026, 7, 1))
        assert est[2025] == 1.0 and est[2026] == pytest.approx(0.5, abs=0.02)

    def test_coverage_at_reads_as_a_cdf(self):
        assert coverage_at([1.0, 2.0, 3.0, 4.0], 2.0) == 0.5
        assert coverage_at([1.0, 2.0], 99.0) == 1.0
        assert coverage_at([], 5.0) == 0.0

    def test_lag_quantiles_do_not_interpolate(self):
        """The register reports lag to several decimals but it means about a month;
        interpolating would dress up a precision the source does not have."""
        assert lag_quantiles([1.0, 2.0, 3.0, 4.0], (0.5,))[0.5] == 2.0


class TestSource:
    def test_item_text_joins_the_title_to_the_abstract(self):
        """A patent title is a formal claim phrase, so it carries the subject and the
        abstract carries the substance."""
        text = item_text({"title": "СПОСІБ ОЧИЩЕННЯ", "abstract": "Винахід належить."})
        assert text == "СПОСІБ ОЧИЩЕННЯ. Винахід належить."

    def test_item_text_does_not_double_the_stop(self):
        joined = item_text(
            {"title": "Спосіб очищення води.", "abstract": "Винахід належить до хімії."}
        )
        assert joined == "Спосіб очищення води. Винахід належить до хімії."

    def test_text_below_the_normalizer_floor_is_skipped(self):
        """``normalize_text`` rejects anything under 20 characters, and the source drops
        the record rather than emitting an item with no text. On this register that costs
        nothing — the shortest abstract is 43 characters — but a stub title with no
        abstract would otherwise reach the embedder as a near-empty string."""
        assert item_text({"title": "Спосіб", "abstract": None}) == ""

    def test_limit_and_sample_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            PatentSource("x", limit=5, sample=5)

    def test_an_unbuilt_dataset_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="patents-build"):
            list(PatentSource(tmp_path).fetch())

    def test_items_carry_the_cohort_label_and_the_attention_key(self, tmp_path):
        import json

        from horizon.ingest.base import attention_of

        shards = tmp_path / "records"
        shards.mkdir()
        rec = normalize_row(row())
        rec["citation_percentile"] = 0.75
        (shards / "2012.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        item = list(PatentSource(tmp_path).fetch())[0]
        assert item.source == "patents-ua-2012"
        assert item.published_at == "2012-01-01"
        assert item.lang == "uk"  # the register writes "UA"
        assert attention_of(item) == 0.75


class TestVocabulary:
    """The overlay exists so a patent register does not render as a corpus of papers."""

    def test_the_overlay_only_names_keys_the_base_already_has(self):
        """A key here that the report does not use is wording nobody will ever see, and
        the usual cause is a typo in a key that *is* used."""
        from horizon.patents.vocabulary import PATENT_LABELS
        from horizon.report import LABELS

        for lang, over in PATENT_LABELS.items():
            assert set(over) <= set(LABELS[lang]), f"{lang}: unknown keys"

    def test_both_languages_override_the_same_keys(self):
        """Overriding a heading in one language only is how a report ends up half in the
        wrong vocabulary, and it is invisible unless both are read side by side."""
        from horizon.patents.vocabulary import PATENT_LABELS

        assert set(PATENT_LABELS["en"]) == set(PATENT_LABELS["uk"])

    def test_an_unknown_language_raises_rather_than_falling_back(self):
        from horizon.patents.vocabulary import labels

        with pytest.raises(KeyError):
            labels("de")

    def test_rendering_with_an_overlay_keeps_the_keys_it_does_not_name(self):
        """The overlay is partial by design: the base completes it, so a string added to
        the report later appears in its bibliographic form rather than vanishing."""
        from horizon.report import LABELS, ReportFacts, render_markdown
        from horizon.patents.vocabulary import labels

        facts = ReportFacts(
            records=10, unclassified=0, years=(2012, 2013), fields=(), pairs=()
        )
        text = render_markdown(facts, lang="uk", labels=labels("uk"))
        assert "Патентів:" in text  # overridden
        assert LABELS["uk"]["span"] in text  # inherited, not dropped

    def test_the_default_render_is_unchanged_by_the_hook(self):
        """``labels=None`` must be the exact path that existed before domains did — the
        bibliographic reports are already published against it."""
        from horizon.report import ReportFacts, render_markdown

        facts = ReportFacts(
            records=10, unclassified=0, years=(2012, 2013), fields=(), pairs=()
        )
        assert render_markdown(facts, lang="uk") == render_markdown(
            facts, lang="uk", labels=None
        )

    def test_the_domain_suppresses_the_script_indicator(self):
        """Structural, not empirical. Filing with the Ukrainian office requires Ukrainian,
        so a patent's script is set by the filing rules rather than by its subject and can
        say nothing about topics. The generic flatness guard would re-enable the indicator
        the moment some slice happened to be mixed — and would then print a ranking of an
        administrative requirement."""
        from horizon.cli import _suppressed

        assert "script-distribution" in _suppressed("patents")
        assert _suppressed("bibliographic") == frozenset()
        assert _suppressed(None) == frozenset()

    def test_every_suppressed_indicator_is_one_the_report_knows(self):
        """A name that matches nothing suppresses nothing, silently."""
        from horizon.lenses import BY_NAME
        from horizon.patents.vocabulary import SUPPRESSED_INDICATORS
        from horizon.report import LABELS

        known = set(BY_NAME) | {
            k[len("ind_") :] for k in LABELS["en"] if k.startswith("ind_")
        }
        assert SUPPRESSED_INDICATORS <= known, SUPPRESSED_INDICATORS - known

    def test_no_patent_label_carries_bibliographic_vocabulary(self):
        """The overlay is partial by design, and that is exactly how it goes wrong.

        The first version overrode the column headings and left the long notes alone, so
        the report printed a table headed "патентів/рік" above a paragraph explaining
        what "Публікації/рік" measures — and the script indicator quoted the *other*
        corpus's numbers, 8.7% Cyrillic against 91.3% Latin, at a register that is 100%
        Cyrillic. Nothing failed; it just read as though written about papers.

        This scan is the guard. A key added to the report in a domain-neutral wording
        stays inherited and silent, but one written in publication terms trips here.
        """
        import re

        from horizon.report import LABELS
        from horizon.patents.vocabulary import PATENT_LABELS

        offenders = {
            "uk": re.compile(r"публікац|Публікац|автор|Автор|предметн|Предметн|видан"),
            "en": re.compile(
                r"publication|Publication|author|Author|subject area|journal|Journal"
            ),
        }
        leaked = {
            lang: [
                key
                for key, text in LABELS[lang].items()
                if key not in PATENT_LABELS[lang] and offenders[lang].search(text)
            ]
            for lang in offenders
        }
        assert not any(leaked.values()), leaked

    def test_a_multi_paragraph_caveat_stays_inside_the_blockquote(self):
        """The patent caveat carries two footnotes. Prefixing only its first line drops
        the rest into the body, where they read as the report's own text rather than as
        a qualification of it."""
        from horizon.report import ReportFacts, render_markdown
        from horizon.patents.vocabulary import labels

        facts = ReportFacts(
            records=10, unclassified=0, years=(2012, 2013), fields=(), pairs=()
        )
        text = render_markdown(facts, lang="uk", labels=labels("uk"))
        block = [ln for ln in text.splitlines() if ln.startswith(">")]
        assert any("†" in ln for ln in block) and any("‡" in ln for ln in block)


class TestStratifiedCoverage:
    """Two populations that grant at different speeds break a single pooled estimate.

    On this register residents reach grant in a median 1.57 years and non-residents in
    5.37, so a pooled distribution describes neither — and the error is worst exactly
    where the answer matters, in the newest cohorts.
    """

    @staticmethod
    def mixed(year: int, fast: int, slow: int):
        """One cohort of ``fast``-granting and ``slow``-granting records."""
        out = []
        for origin, n, lag in (("resident", fast, 1.5), ("non_resident", slow, 5.5)):
            for _ in range(n):
                rec = normalize_row(row())
                rec["year"] = rec["priority_year"] = year
                rec["origin"], rec["lag_years"] = origin, lag
                out.append(rec)
        return out

    def register(self):
        """Old cohorts to learn both lag distributions from, plus one recent cohort."""
        recs = []
        for y in range(2010, 2019):
            recs += self.mixed(y, 300, 300)
        return recs

    def test_a_group_is_extrapolated_on_its_own_lag(self):
        """Three years after priority the fast group is fully granted and the slow one
        has nothing at all. Pooled, the cohort reads as half complete and the guard is
        deciding on a number true of neither group."""
        recs = self.register() + self.mixed(2023, 300, 0)
        _, est = complete_years(recs, cut=date(2026, 8, 12), by="origin")
        assert est[2023] == pytest.approx(1.0)  # only fast records are in it
        _, pooled = complete_years(recs, cut=date(2026, 8, 12))
        assert pooled[2023] < 0.75  # the pooled estimate is wrong about this cohort

    def test_a_cohort_whose_slow_half_is_invisible_is_not_called_complete(self):
        recs = self.register() + self.mixed(2023, 300, 300)
        _, est = complete_years(recs, cut=date(2026, 8, 12), by="origin")
        assert est[2023] < 0.6

    def test_an_unstratified_call_is_unchanged(self):
        recs = self.register()
        assert (
            complete_years(recs, cut=date(2026, 8, 12))[1]
            == complete_years(recs, cut=date(2026, 8, 12), by=None)[1]
        )


class TestLagDrift:
    """The estimator assumes the lag distribution is stationary. Here it is not."""

    def test_the_pooled_median_can_hide_a_drift_that_the_groups_show(self):
        """The defect this guards: pooled, the register's median lag is flat and then
        falls, which reads as an office speeding up, while the resident median rises 62%
        throughout. What moves the pooled number is the changing mix of the two
        populations, so a pooled drift check can be cancelled by composition alone."""
        from horizon.patents.completeness import lag_drift

        recs = []
        # Both groups' lags are constant; only their proportions change.
        for year, fast, slow in ((2012, 100, 400), (2020, 400, 100)):
            for origin, n, lag in (
                ("resident", fast, 2.0),
                ("non_resident", slow, 6.0),
            ):
                for _ in range(n):
                    rec = normalize_row(row())
                    rec["year"] = rec["priority_year"] = year
                    rec["origin"], rec["lag_years"] = origin, lag
                    recs.append(rec)
        pooled = lag_drift(recs)[None]
        assert pooled[2012] > pooled[2020]  # looks like a speed-up
        grouped = lag_drift(recs, by="origin")
        for series in grouped.values():  # but neither group moved at all
            assert series[2012] == series[2020]


class TestFitYears:
    """Showing a partial cohort and deriving statistics from it are different decisions.

    The war cohorts must stay visible — deleting the years a reader came to look at is
    not a fix — but a cohort still filling up corrupts both derived columns, and in two
    different ways.
    """

    @staticmethod
    def tabulation():
        from horizon.fields import tabulate

        recs = []
        for year, slow, fast in (
            (2018, 100, 100),
            (2019, 100, 100),
            (2020, 100, 100),
            (2021, 5, 100),  # only the fast field has been granted yet
        ):
            recs += [{"year": year, "subject_areas": ["slow"]}] * slow
            recs += [{"year": year, "subject_areas": ["fast"]}] * fast
        return tabulate(recs)

    def test_the_counts_keep_every_cohort(self):
        from horizon.fields import field_series

        series = {s.name: s for s in field_series(self.tabulation())}
        assert dict(series["slow"].counts)[2021] == 5

    def test_shares_are_read_at_the_fitted_endpoints(self):
        """The subtle failure: a partial cohort is not a uniform sample of itself. Here
        the slow field is granted later, so the 2021 cohort is 95% fast, and reading the
        end share there reports a collapse in one field and a surge in the other when
        nothing about either has changed."""
        from horizon.fields import field_series

        naive = {s.name: s for s in field_series(self.tabulation())}
        assert naive["slow"].share_last < 0.06  # reads as a collapse
        fitted = {
            s.name: s
            for s in field_series(self.tabulation(), fit_years=[2018, 2019, 2020])
        }
        assert fitted["slow"].share_last == pytest.approx(0.5)

    def test_the_slope_is_fitted_over_the_given_years_only(self):
        from horizon.fields import field_series

        naive = {s.name: s for s in field_series(self.tabulation())}
        fitted = {
            s.name: s
            for s in field_series(self.tabulation(), fit_years=[2018, 2019, 2020])
        }
        assert naive["slow"].slope < -20  # dragged toward the stub
        assert fitted["slow"].slope == pytest.approx(0.0)

    def test_omitting_fit_years_is_the_previous_behaviour(self):
        from horizon.fields import field_series

        tab = self.tabulation()
        assert [s.slope for s in field_series(tab)] == [
            s.slope for s in field_series(tab, fit_years=None)
        ]

    def test_the_facts_carry_the_fitted_range_for_the_figures(self):
        """The figures shade what was not fitted, and they read it from the JSON."""
        from horizon.report import build_facts

        facts = build_facts(self.tabulation(), fit_years=[2018, 2019, 2020])
        assert facts.to_dict()["fit_years"] == [2018, 2019, 2020]
        assert facts.to_dict()["years"][-1] == 2021  # still shown


class TestUnblockedClustering:
    """Dropping the blocking changes what section 3 claims, and the change is the point.

    Blocked, a topic is assigned to one field and its numbers are that field's. Unblocked,
    the clusters are global and each is scored inside every field its members belong to —
    so a topic spanning three technologies appears three times, with three different sets
    of numbers, none of them the register-wide ones.
    """

    def test_a_multi_field_record_reaches_every_field_it_names(self):
        """The blocked path picks one primary field per record and drops the rest. On a
        register where 46% of grants carry more than one WIPO technology that is not a
        simplification, it is a discarded half of the classification."""
        from horizon.corpus import block, primary_field

        record = {"subject_areas": ["Фармацевтика", "Біотехнології"]}
        assert primary_field(record, keys=("subject_areas",)) == "Біотехнології"
        # One block, one field — the second is simply gone.
        grouped = block([(object(), record)], keys=("subject_areas",))
        assert list(grouped) == ["Біотехнології"]

    def test_the_flag_defaults_to_the_previous_behaviour(self):
        """The bibliographic reports are published against the blocked path."""
        from horizon.cli import build_parser

        args = build_parser().parse_args(["report", "--records", "x"])
        assert args.block_by == "field"
