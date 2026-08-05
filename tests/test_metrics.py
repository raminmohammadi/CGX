"""Unit tests for the in-process metrics registry + Prometheus exporter."""

from __future__ import annotations

from cgx import metrics as m


def _reset():
    m.reset_for_tests()


def test_counter_render_with_and_without_labels():
    _reset()
    m.inc("cgx_test_total", help="a help string")
    m.inc("cgx_test_total", 2.0)
    m.inc("cgx_test_labeled_total", provider="ollama", model="x")
    out = m.render_prometheus()
    assert "# HELP cgx_test_total a help string" in out
    assert "# TYPE cgx_test_total counter" in out
    assert "cgx_test_total 3" in out
    assert 'cgx_test_labeled_total{model="x",provider="ollama"} 1' in out


def test_gauge_set_overwrites():
    _reset()
    m.set_gauge("cgx_test_gauge", 5.0)
    m.set_gauge("cgx_test_gauge", 9.0)
    out = m.render_prometheus()
    assert "# TYPE cgx_test_gauge gauge" in out
    assert "cgx_test_gauge 9" in out


def test_histogram_buckets_sum_count():
    _reset()
    for v in (10.0, 300.0, 5000.0):
        m.observe("cgx_test_latency_ms", v, model="m")
    out = m.render_prometheus()
    assert "# TYPE cgx_test_latency_ms histogram" in out
    # +Inf bucket equals total count.
    assert 'cgx_test_latency_ms_bucket{model="m",le="+Inf"} 3' in out
    assert 'cgx_test_latency_ms_count{model="m"} 3' in out
    assert 'cgx_test_latency_ms_sum{model="m"} 5310' in out
    # Cumulative: le=250 should capture only the 10ms sample.
    assert 'cgx_test_latency_ms_bucket{model="m",le="250"} 1' in out
    # le=1000 captures the 10ms and 300ms samples.
    assert 'cgx_test_latency_ms_bucket{model="m",le="1000"} 2' in out


def test_label_value_escaping():
    _reset()
    m.inc("cgx_test_esc_total", route='a"b\\c')
    out = m.render_prometheus()
    assert 'route="a\\"b\\\\c"' in out


def test_render_empty_registry():
    _reset()
    assert m.render_prometheus() == ""
