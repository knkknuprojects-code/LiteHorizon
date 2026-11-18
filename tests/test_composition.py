"""Tests for the per-topic composition split."""

from __future__ import annotations

import pytest

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.composition import baselines_for, compose
from horizon.corpus import RecordIndex
from horizon.trends import Trend


def _analysis(h, year):
    return Analysis(
        summary="s",
        signals=("sig",),
        classification=Classification(
            "t", Maturity.EMERGING, Impact.HIGH, TimeHorizon.NEAR
        ),
        entities=Entities(),
        keywords=("k",),
        content_hash=h,
        model_id="m",
        prompt_version="phase2.v4",
        strategy="combined",
        generated_at="2026-01-01T00:00:00+00:00",
        item_date=f"{year}-01-01T00:00:00+00:00",
    )


def _setup(rows):
    """rows: (hash, year, group). Returns (trend, analyses, index)."""
    analyses = {h: _analysis(h, y) for h, y, _ in rows}
    records = [
        {"title": f"t{h}", "abstract": "a" * 40, "year": y, "applicant_type": g}
        for h, y, g in rows
    ]
    # RecordIndex keys on the record hash; build it from the same texts the analyses
    # were made from so the join resolves.
    index = RecordIndex(by_hash={h: r for (h, _, _), r in zip(rows, records)})
    trend = Trend(
        label="t",
        members=tuple(h for h, _, _ in rows),
        size=len(rows),
        first_seen="2012-01-01",
        last_seen="2020-01-01",
        velocity=1.0,
        score=1.0,
        top_keywords=(),
    )
    return trend, analyses, index, records


YEARS = [2012, 2013, 2014, 2015, 2016, 2017]


def test_groups_are_rated_against_their_own_corpus_rate():
    """A population contracting everywhere must not make all of its topics read as
    declining — that would say something about the population, not about any topic."""
    # academic halves corpus-wide; inside this topic it also halves -> velocity 1.0
    rows = [(f"a{i}", 2012, "academic") for i in range(10)]
    rows += [(f"b{i}", 2016, "academic") for i in range(5)]
    trend, analyses, index, _ = _setup(rows)
    base = {"academic": {2012: 100, 2016: 50}}
    comp = compose(
        trend, analyses, index, key="applicant_type", years=YEARS, baselines=base
    )
    g = {x.name: x for x in comp.groups}["academic"]
    assert g.velocity == 1.0  # moving exactly with its own population


def test_a_small_group_gets_no_velocity():
    """A ratio over three records is noise with a decimal point."""
    rows = [(f"a{i}", 2012, "academic") for i in range(10)]
    rows += [(f"e{i}", 2016, "enterprise") for i in range(3)]
    trend, analyses, index, _ = _setup(rows)
    comp = compose(
        trend,
        analyses,
        index,
        key="applicant_type",
        years=YEARS,
        baselines=baselines_for([], key="applicant_type", years=YEARS),
    )
    small = {g.name: g for g in comp.groups}["enterprise"]
    assert small.size == 3 and small.velocity is None and not small.measurable


def test_divergence_needs_two_measurable_groups():
    rows = [(f"a{i}", 2012, "academic") for i in range(10)]
    trend, analyses, index, _ = _setup(rows)
    comp = compose(
        trend,
        analyses,
        index,
        key="applicant_type",
        years=YEARS,
        baselines={"academic": {2012: 10}},
    )
    assert comp.divergence is None  # one group -> nothing to diverge from


def test_divergence_is_the_spread_between_the_extremes():
    """The point of the section: a topic where one population rises while another
    withdraws is a different claim from either, and the total shows neither."""
    # academic 2 early -> 8 late (rising); enterprise 8 early -> 2 late (withdrawing)
    rows = [(f"a{i}", 2012, "academic") for i in range(2)]
    rows += [(f"A{i}", 2017, "academic") for i in range(8)]
    rows += [(f"e{i}", 2012, "enterprise") for i in range(8)]
    rows += [(f"E{i}", 2017, "enterprise") for i in range(2)]
    trend, analyses, index, _ = _setup(rows)
    base = {
        "academic": {2012: 50, 2017: 50},
        "enterprise": {2012: 50, 2017: 50},
    }
    comp = compose(
        trend, analyses, index, key="applicant_type", years=YEARS, baselines=base
    )
    rates = {g.name: g.velocity for g in comp.groups}
    assert rates["academic"] > rates["enterprise"]
    assert comp.divergence == rates["academic"] - rates["enterprise"]


def test_an_empty_prior_window_is_new_rather_than_enormous_growth():
    """The first version computed recent / max(prior, 1), which returns the recent count
    itself when the prior window is empty — so a group of twenty scored 20.0 and a group
    of five scored 5.0, which is size wearing the name of growth. It put 12% of cells
    above 5.0 against a median of 1.12, topping out at 31.4."""
    rows = [(f"a{i}", 2017, "academic") for i in range(20)]
    trend, analyses, index, _ = _setup(rows)
    base = {"academic": {2012: 50, 2017: 50}}
    comp = compose(
        trend, analyses, index, key="applicant_type", years=YEARS, baselines=base
    )
    g = comp.groups[0]
    assert g.velocity is None and g.is_new
    assert g.size == 20  # the count is still reported; only the fake rate is gone


def test_a_thin_prior_window_gets_no_rate_either():
    """With one record in the prior window the denominator carries the answer."""
    rows = [("p0", 2012, "academic")]
    rows += [(f"a{i}", 2017, "academic") for i in range(9)]
    trend, analyses, index, _ = _setup(rows)
    comp = compose(
        trend,
        analyses,
        index,
        key="applicant_type",
        years=YEARS,
        baselines={"academic": {2012: 50, 2017: 50}},
    )
    g = comp.groups[0]
    assert g.velocity is None and not g.is_new  # present early, just too thin to rate


def test_unresolved_members_do_not_reweight_the_shares():
    """A member whose record is missing counts toward size and toward no group; the
    shares must come from what was resolved, or partial coverage silently inflates
    whichever group did resolve."""
    rows = [(f"a{i}", 2012, "academic") for i in range(6)]
    trend, analyses, index, _ = _setup(rows)
    trend = Trend(
        label="t",
        members=trend.members + ("missing",),
        size=7,
        first_seen="2012-01-01",
        last_seen="2016-01-01",
        velocity=1.0,
        score=1.0,
        top_keywords=(),
    )
    comp = compose(
        trend, analyses, index, key="applicant_type", years=YEARS, baselines={}
    )
    assert comp.size == 7 and comp.resolved == 6
    assert comp.groups[0].share == 1.0


def test_baselines_count_only_the_reported_cohorts():
    records = [
        {"year": 2012, "applicant_type": "academic"},
        {"year": 2030, "applicant_type": "academic"},
    ]
    base = baselines_for(records, key="applicant_type", years=YEARS)
    assert base == {"academic": {2012: 1}}


class TestPromptDomain:
    """The prompt tells the model what it is reading, and getting that wrong is the one
    error a reader of the output cannot catch — the prose will be fluent and about the
    wrong kind of document."""

    def test_every_domain_defines_the_same_nouns(self):
        """A noun missing from one domain raises KeyError at format time, mid-run, after
        the model is already loaded."""
        from horizon.interpret import VOCAB

        shapes = {
            (domain, lang): set(words)
            for domain, langs in VOCAB.items()
            for lang, words in langs.items()
        }
        assert len(set(map(frozenset, shapes.values()))) == 1, shapes.keys()

    def test_every_domain_covers_every_language(self):
        from horizon.interpret import VOCAB

        assert {tuple(sorted(v)) for v in VOCAB.values()} == {("en", "uk")}

    def test_the_prompts_name_the_right_corpus(self):
        from horizon.interpret import topic_prompt

        facts = {"records": 100, "years": [2012, 2020]}
        topic = {"label": "T", "size": 5, "velocity": 1.0, "growth": 0.0}
        biblio = topic_prompt(topic, [], facts, lang="uk", domain="bibliographic")
        patents = topic_prompt(topic, [], facts, lang="uk", domain="patents")
        assert "WoS/Scopus" in biblio and "патент" not in biblio
        assert "патентному реєстрі" in patents and "WoS/Scopus" not in patents

    def test_an_unknown_domain_raises_rather_than_falling_back(self):
        from horizon.interpret import vocab

        with pytest.raises(KeyError):
            vocab("uk", "nonsense")


class TestFieldSplits:
    """Whether an aggregate movement is about the subject or about the mix."""

    @staticmethod
    def records(rows):
        """rows: (year, group, field) repeated n times via a count."""
        out = []
        for year, group, field, n in rows:
            out += [
                {"year": year, "applicant_type": group, "subject_areas": [field]}
            ] * n
        return out

    def test_a_field_moving_one_way_in_each_group_does_not_disagree(self):
        from horizon.composition import field_splits

        recs = self.records(
            [
                (2012, "a", "F", 10),
                (2012, "a", "G", 90),
                (2020, "a", "F", 30),
                (2020, "a", "G", 70),
                (2012, "b", "F", 10),
                (2012, "b", "G", 90),
                (2020, "b", "F", 30),
                (2020, "b", "G", 70),
            ]
        )
        f = {
            s.name: s
            for s in field_splits(recs, key="applicant_type", years=[2012, 2020])
        }
        assert not f["F"].disagrees

    def test_opposite_directions_are_flagged(self):
        """The case the whole thing exists for: the union says one thing and every
        population inside it says something else."""
        from horizon.composition import field_splits

        recs = self.records(
            [
                (2012, "a", "F", 10),
                (2012, "a", "G", 90),
                (2020, "a", "F", 40),
                (2020, "a", "G", 60),  # a: F rises
                (2012, "b", "F", 60),
                (2012, "b", "G", 40),
                (2020, "b", "F", 20),
                (2020, "b", "G", 80),  # b: F falls
            ]
        )
        f = {
            s.name: s
            for s in field_splits(recs, key="applicant_type", years=[2012, 2020])
        }
        assert f["F"].disagrees

    def test_shares_are_of_the_group_not_of_the_corpus(self):
        """Dividing by the union would fold the changing mix back into the answer this
        exists to separate."""
        from horizon.composition import field_splits

        recs = self.records(
            [
                (2012, "a", "F", 50),
                (2012, "a", "G", 50),
                (2020, "a", "F", 50),
                (2020, "a", "G", 50),  # a never changes
                (2012, "b", "F", 5),
                (2012, "b", "G", 45),
                (2020, "b", "F", 50),
                (2020, "b", "G", 450),  # b grows tenfold at a constant 10% F
            ]
        )
        f = {
            s.name: s
            for s in field_splits(recs, key="applicant_type", years=[2012, 2020])
        }
        a = next(g for g in f["F"].groups if g[0] == "a")
        assert a[1] == a[2] == 0.5  # unmoved, despite the corpus mix changing hugely

    def test_a_cohort_too_small_for_a_share_is_dropped(self):
        from horizon.composition import field_splits

        recs = self.records(
            [
                (2012, "a", "F", 100),
                (2020, "a", "F", 100),
                (2012, "tiny", "F", 3),
                (2020, "tiny", "F", 3),
            ]
        )
        rows = field_splits(
            recs, key="applicant_type", years=[2012, 2020], min_group=50
        )
        assert rows == []  # only one group survives, and one group is not a split

    def test_the_aggregate_view_is_used_when_there_is_no_split(self):
        """A bibliographic corpus has one population, and must get the prompt it always
        had — including the instruction, which would otherwise ask for a paragraph about
        populations that were never supplied."""
        from horizon.interpret import fields_view

        facts = {
            "fields": [
                {
                    "name": "Physics",
                    "share_first": 0.2,
                    "share_last": 0.1,
                    "share_delta": -0.1,
                }
            ]
        }
        data, task = fields_view(facts, lang="uk")
        return_all = data + task
        assert "найбільше втрачають частку" in data
        assert "Сукупності" not in return_all and "сукупність" not in task

    def test_the_population_view_leads_when_a_split_exists(self):
        """The arrangement that finally worked: populations in the primary position, the
        aggregate surviving only for the fields they agree on. An earlier version put the
        aggregate first and appended a rule saying not to read it as a field changing —
        the model read it that way anyway, because the rule contradicted the output
        specification and the specification won."""
        from horizon.interpret import fields_view

        facts = {
            "populations": {"university": {"total": 100, "first": 50, "last": 50}},
            "field_splits": [
                {
                    "name": "Фармацевтика",
                    "disagrees": True,
                    "groups": [
                        {
                            "name": "university",
                            "share_first": 0.06,
                            "share_last": 0.11,
                        }
                    ],
                }
            ],
        }
        data, task = fields_view(facts, lang="uk")
        assert "Сукупності в цьому корпусі" in data
        assert "Фармацевтика" in data and "6.0%→11.0%" in data
        assert "сукупність" in task

    def test_the_data_and_the_instruction_cannot_disagree(self):
        """They are returned as a pair for exactly this reason."""
        from horizon.interpret import FIELDS_TASK, fields_view

        empty, task = fields_view({}, lang="uk")
        assert task == FIELDS_TASK["aggregate"]["uk"]


class TestLeakedChannelMarker:
    """Gemma sometimes emits a bare ``thought`` line before the answer proper."""

    def test_the_marker_is_stripped(self):
        """It reached the rendered documents as a first word — three blocks in 87 —
        because it is lower-case, carries no markup, and reads as a typo rather than as
        the model's own scaffolding."""
        from horizon.interpret import clean_reading

        assert clean_reading("thought\nТема «X» охоплює.") == "Тема «X» охоплює."

    def test_a_stray_character_before_it_does_not_save_it(self):
        """It arrived both ways; the second form is what made the first look accidental."""
        from horizon.interpret import clean_reading

        assert clean_reading(".thought\nДана область.") == "Дана область."

    def test_a_real_sentence_containing_the_word_survives(self):
        from horizon.interpret import clean_reading

        text = "Ця думка (thought) належить до тексту."
        assert clean_reading(text) == text

    def test_only_the_first_line_is_taken(self):
        """Anchored to the start: a later occurrence is prose, not scaffolding."""
        from horizon.interpret import clean_reading

        text = "Перший абзац.\nthought\nДругий."
        assert clean_reading(text) == text
