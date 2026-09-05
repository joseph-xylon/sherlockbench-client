import pytest
from sherlockbench_anthropic.investigate_verify import list_to_map, normalize_args, make_tools

def test_list_to_map():
    assert list_to_map(['integer', 'integer', 'integer']) == {
        'a': {'type': 'integer'},
        'b': {'type': 'integer'},
        'c': {'type': 'integer'}
    }
    assert list_to_map(["boolean", "boolean"]) == {
        'a': {'type': 'boolean'},
        'b': {'type': 'boolean'}
    }

def test_normalize_args():
    assert normalize_args({'a': 1, 'b': 2, 'c': 3}) == [1, 2, 3]

def test_make_tools():
    """anthropic nests the schema under input_schema, and wants strict + closed properties"""
    assert make_tools(['integer', 'string']) == [
        {
            "name": "mystery_function",
            "description": "Use this tool to test the mystery function.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {'a': {'type': 'integer'},
                               'b': {'type': 'string'}},
                "required": ['a', 'b'],
                "additionalProperties": False
            }
        }
    ]
