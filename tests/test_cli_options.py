from haymaker.config.cli_options import (
    CustomArgParser,
    get_parser_for_dataloader,
    get_parser_for_other_module,
)


def test_set_option_short():
    parser = CustomArgParser.from_profile("dataloader", ["-s", "key", "value"])
    assert parser.output.get("key") == "value"


def test_set_option_long():
    parser = CustomArgParser.from_profile(
        "dataloader", ["--set_option", "key", "value"]
    )
    assert parser.output.get("key") == "value"


def test_set_option_hyphenated_alias():
    parser = CustomArgParser.from_profile(
        "dataloader", ["--set-option", "key", "value"]
    )
    assert parser.output.get("key") == "value"


def test_set_option_with_multiple_options():
    parser = CustomArgParser.from_profile(
        "dataloader",
        ["-s", "key", "value", "-s", "key1", "value1", "-s", "key2", "value2"],
    )
    output = parser.output
    assert output["key"] == "value"
    assert output["key1"] == "value1"
    assert output["key2"] == "value2"


def test_module_lookup_works():
    parser = CustomArgParser.from_str("--test_option", "my_module.py")
    assert parser.output.get("test_option")


def test_common_options_work_for_non_default_modules():
    parser = CustomArgParser.from_str("--set_option key value", "my_module.py")
    assert parser.output.get("key") == "value"


def test_dataloader_source():
    parser = CustomArgParser.from_profile("dataloader", ["myfile.csv"])
    output = parser.output
    assert output["source"] == "myfile.csv"


def test_live_module_path_not_source():
    parser = CustomArgParser.from_profile(
        "live", ["my_strategy.py", "-r", "-f", "filename.yaml", "-z", "--nuke"]
    )
    output = parser.output
    assert output["module_path"] == "my_strategy.py"
    assert "source" not in output


def test_app_options():
    parser = CustomArgParser.from_profile(
        "live", ["my_strategy.py", "-r", "-f", "filename.yaml", "-z", "--nuke"]
    )
    output = parser.output
    assert output["reset"]
    assert output["zero"]
    assert output["nuke"]
    assert not output["coldstart"]
    assert output["file"] == "filename.yaml"


def test_docs_parser_helpers_return_unparsed_parsers():
    live_parser = get_parser_for_other_module()
    dataloader_parser = get_parser_for_dataloader()

    assert live_parser.parse_args(["strategy.py", "--nuke"]).nuke
    assert dataloader_parser.parse_args(["contracts.csv"]).source == "contracts.csv"
