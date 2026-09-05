import pytest
from sherlockbench_openai.investigate_verify import list_to_map, normalize_args, make_tools

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
    assert list_to_map(['string', 'string']) == {
        'a': {'type': 'string'},
        'b': {'type': 'string'}
    }
    assert list_to_map(['string', 'integer']) == {
        'a': {'type': 'string'},
        'b': {'type': 'integer'}
    }

def test_normalize_args():
    assert normalize_args({'a': 1, 'b': 2, 'c': 3}) == [1, 2, 3]

def test_make_tools():
    """the responses api wants the function flat, not nested under a "function" key"""
    assert make_tools(['integer', 'string']) == [
        {
            "type": "function",
            "name": "mystery_function",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {'a': {'type': 'integer'},
                               'b': {'type': 'string'}},
                "required": ['a', 'b'],
                "additionalProperties": False
            },
        }
    ]
