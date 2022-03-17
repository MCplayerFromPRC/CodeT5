import sys
import csv
import re
import json, jsonlines
from typing import List
from tree_sitter import Parser, Language

sys.path.append("../")
DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
PARSER_PATH = '../evaluator/CodeBLEU/'
DATA_PATH = '../data/dataset/'
JAVA_LANGUAGE = Language(f'{PARSER_PATH}parser/my-languages.so', 'python')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def tokenize_docstring(docstring: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


def match_from_span(node, blob: str) -> str:
    lines = blob.split('\n')
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]


def get_function_definitions(node):
    for child in node.children:
        if child.type == 'function_definition':
            yield child
        elif child.type == 'decorated_definition':
            for c in child.children:
                if c.type == 'function_definition':
                    yield c


def traverse(node, results: List) -> None:
    if node.type == 'string':
        results.append(node)
        return
    for n in node.children:
        traverse(n, results)
    if not node.children:
        results.append(node)


def tokenize_code(tree, blob: str) -> List:
    tokens = []
    for node in tree.root_node.children:
        if node.type in ['function_definition', 'class_definition', 'import_statement']:
            traverse(node, tokens)
        elif node.type == 'decorated_definition':
            for child in node.children:
                if child.type in ['function_definition', 'class_definition']:
                    traverse(child, tokens)
    return [match_from_span(token, blob) for token in tokens]


class record():
    def __init__(self, textCN, textEN, code, api, sizeIn, sizeOut, id, ref_id, type):
        self.textCN = textCN
        self.textEN = textEN
        self.code = code
        self.api = api
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.id = id
        self.ref_id = ref_id
        self.type = type

    def __hash__(self):
        return hash(self.textCN + self.textEN + self.code)

    def __eq__(self, other):
        if self.textCN == other.textCN and self.textEN == other.textEN and self.code == other.code:
            return True
        else:
            return False

    def item(self):
        return [self.textCN, self.textEN, self.code, self.api, self.sizeIn, self.sizeOut, self.id, self.ref_id, self.type]


col_type = [str, str, str, eval, eval, eval, str, str, str]
col_name = ['textCN', 'textEN', 'code', 'api', 'sizeIn', 'sizeOut', 'id', 'ref_id', 'type']
json_list = []
with open(f'{DATA_PATH}data.v0.6-all.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        row = list(convert(value) for convert, value in zip(col_type, row))
        json_obj = {
            'idx': row[6],
            'code': row[2],
            'docstring': row[1],
            'code_tokens': tokenize_code(parser.parse(row[2].encode()), row[2]),
            'docstring_tokens': tokenize_docstring(row[1]),
            'type': row[8]
        }
        json_list.append(json_obj)


with jsonlines.open(f'{DATA_PATH}output.jsonl', mode='w') as writer:
    for json_obj in json_list:
        writer.write(json_obj)