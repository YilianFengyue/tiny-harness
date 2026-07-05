from main import parse_args


def test_coordinator_cli_flag_defaults_to_existing_mode():
    args = parse_args(["do something"])

    assert args.coordinator_mode is None


def test_coordinator_cli_flag_can_enable_or_disable_mode():
    enabled = parse_args(["--coordinator", "do something"])
    disabled = parse_args(["--no-coordinator", "do something"])

    assert enabled.coordinator_mode is True
    assert disabled.coordinator_mode is False
