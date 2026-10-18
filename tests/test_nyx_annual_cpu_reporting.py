import numpy as np
import pandas as pd
import pytest
from html import escape

from chronos2_hourly.nyx_annual_cpu_reporting import render_country_report, write_country_report
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid


@pytest.mark.parametrize("day,hours", [("2026-09-30",24), ("2026-03-29",23), ("2026-10-25",25)])
def test_report_preserves_all_physical_hours_and_has_no_remote_resources(day, hours):
    index = delivery_grid(day)[1]
    frame = pd.DataFrame({"price_eur_mwh": np.linspace(-10,100,hours),
                          "p_negative": np.linspace(1,0,hours)}, index=index)
    page = render_country_report(frame, country="DE", delivery_day=day, de_performance_exception=True)
    assert page.count("<tr>") == hours+1
    assert "DE : exception de performance" in page
    assert "<script" not in page and "https://" not in page and "http://" not in page
    assert page.count("<svg ") == 2
    for stamp in index:
        assert stamp.strftime("%Y-%m-%d %H:%M")+"Z" in page
    if hours == 25:
        assert "25/10/2026 02:00 <span class=\"offset\">UTC+02:00" in page
        assert "25/10/2026 02:00 <span class=\"offset\">UTC+01:00" in page


def test_report_refuses_partial_grid_invalid_probabilities_and_overwriting(tmp_path):
    day = "2026-09-30"
    index = delivery_grid(day)[1]
    frame = pd.DataFrame({"price_eur_mwh": np.zeros(len(index)), "p_negative": np.full(len(index), .5)}, index=index)
    with pytest.raises(ValueError, match="exact physical"):
        render_country_report(frame.iloc[:-1], country="FR", delivery_day=day)
    bad = frame.copy()
    bad.iloc[0,1] = 1.1
    with pytest.raises(ValueError, match="Finite prices"):
        render_country_report(bad, country="FR", delivery_day=day)
    destination = tmp_path/"report.html"
    write_country_report(frame, destination, country="FR", delivery_day=day)
    assert "exception de performance" not in destination.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_country_report(frame, destination, country="FR", delivery_day=day)


def _forecast_frame(day):
    index = delivery_grid(day)[1]
    return pd.DataFrame({"price_eur_mwh": np.linspace(-10, 100, len(index)),
                         "p_negative": np.linspace(1, 0, len(index))}, index=index)


def test_report_default_keeps_qualified_footer_and_no_status_banner():
    day = "2026-09-30"
    frame = _forecast_frame(day)
    page = render_country_report(frame, country="FR", delivery_day=day)
    assert '<footer>Prévision calculée sur CPU par la chaîne annuelle qualifiée. ' in page
    assert 'Statut du modèle' not in page
    assert 'Modèle non encore qualifié' not in page
    assert page == render_country_report(frame, country="FR", delivery_day=day, report_status=None)


@pytest.mark.parametrize("available,label", [(True, "Oui"), (False, "Non")])
@pytest.mark.parametrize("day,cutoff", [
    ("2026-09-30", "29/09/2026 à 08:00 Europe/Paris (UTC+02:00)"),
    ("2026-03-30", "29/03/2026 à 08:00 Europe/Paris (UTC+02:00)"),
    ("2026-10-26", "25/10/2026 à 08:00 Europe/Paris (UTC+01:00)"),
])
def test_unqualified_report_has_actual_history_status_and_civil_day_cutoff(day, cutoff, available, label):
    frame = _forecast_frame(day)
    page = render_country_report(frame, country="NL", delivery_day=day, report_status={
        "qualification_status": "unqualified", "public_history_before_forecast_cutoff": available,
    })
    assert '<strong>Modèle non encore qualifié</strong>' in page
    assert f'Historique public disponible avant la coupure : <strong>{label}</strong>.' in page
    assert f'Coupure de prévision (J−1 à 08:00) : {cutoff}.' in page
    assert 'chaîne annuelle qualifiée' not in page
    assert '<footer>Prévision calculée sur CPU par le modèle annuel non encore qualifié. ' in page
    # Only the status banner and footer change; charts, price values and risks stay identical.
    banner_start = page.index('<aside class="exception" aria-label="Statut du modèle">')
    banner_end = page.index('</aside>', banner_start) + len('</aside>')
    without_status = page[:banner_start] + page[banner_end:]
    without_status = without_status.replace('le modèle annuel non encore qualifié', 'la chaîne annuelle qualifiée')
    assert without_status == render_country_report(frame, country="NL", delivery_day=day)


def test_write_unqualified_report_escapes_status_note_and_preserves_status(tmp_path):
    day = "2026-09-30"
    note = '<script>alert("test")</script> & contrôle'
    status = {"qualification_status": "unqualified",
              "public_history_before_forecast_cutoff": False, "note": note}
    destination = tmp_path / "unqualified.html"
    result = write_country_report(_forecast_frame(day), destination, country="FR",
                                  delivery_day=day, report_status=status)
    page = result.read_text(encoding="utf-8")
    assert result == destination
    assert escape(note) in page
    assert '<script' not in page
    assert '<strong>Modèle non encore qualifié</strong>' in page
    assert 'chaîne annuelle qualifiée' not in page


@pytest.mark.parametrize("status", [
    {},
    {"qualification_status": "qualified", "public_history_before_forecast_cutoff": True},
    {"qualification_status": '<script>alert("x")</script>', "public_history_before_forecast_cutoff": False},
    {"qualification_status": "unqualified"},
    {"qualification_status": "unqualified", "public_history_before_forecast_cutoff": "False"},
    {"qualification_status": "unqualified", "public_history_before_forecast_cutoff": None},
    {"qualification_status": "unqualified", "public_history_before_forecast_cutoff": False, "note": False},
])
def test_report_refuses_ambiguous_or_invalid_explicit_status(status):
    day = "2026-09-30"
    with pytest.raises(ValueError):
        render_country_report(_forecast_frame(day), country="FR", delivery_day=day, report_status=status)
