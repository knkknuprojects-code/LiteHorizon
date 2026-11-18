from horizon.scanner import SYSTEM_PROMPT, scan_text


def test_scan_text_uses_system_prompt_and_includes_text(make_client):
    client = make_client("REPORT")
    result = scan_text("  some article body  ", client)

    assert result == "REPORT"
    (messages,) = client.calls
    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    # The (stripped) source text is embedded in the user prompt.
    assert "some article body" in messages[1].content
