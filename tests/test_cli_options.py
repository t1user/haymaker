import pytest

from ib_tools.config.cli_options import CustomArgParser


def test_set_option_short():
    parser = CustomArgParser.from_str("-s key value")
    assert parser.output.get("key") == "value"


def test_set_option_long():
    parser = CustomArgParser.from_str("--set_option key value")
    assert parser.output.get("key") == "value"


def test_set_option_with_multiple_options():
    parser = CustomArgParser.from_str("-s key value -s key1 value1 -s key2 value2")
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


def test_source():
    parser = CustomArgParser.from_str("myfile.yaml")
    output = parser.output
    assert output["source"] == "myfile.yaml"


def test_no_source():
    parser = CustomArgParser.from_str("-r -f filename.yaml -z --nuke")
    output = parser.output
    with pytest.raises(KeyError):
        output["source"]


def test_app_options():
    parser = CustomArgParser.from_str("-r -f filename.yaml -z --nuke")
    output = parser.output
    assert output["reset"]
    assert output["zero"]
    assert output["nuke"]
    assert not output["coldstart"]
    assert output["file"] == "filename.yaml"
