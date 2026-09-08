import pytest
from sherlockbench_oai_compat.investigate_verify import list_to_map, normalize_args, make_tools

def test_list_to_map():
    assert list_to_map(['integer', 'integer']) == {'a': {'type': 'integer'},
                                                   'b': {'type': 'integer'}}
    assert list_to_map(['string', 'boolean']) == {'a': {'type': 'string'},
                                                  'b': {'type': 'boolean'}}

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

def test_prompt_mentions_no_parallel_calls():
    """these servers ignore parallel_tool_calls, so the prompt has to ask"""
    from sherlockbench_oai_compat.prompts import make_initial_messages
    developer = make_initial_messages(10)[0]["content"]
    assert "does not support parallel calls" in developer
