import sys
import csv
import re
import json
import jsonlines
from fastHan import FastHan
from typing import List
from tree_sitter import Parser, Language

sys.path.append("../")
DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
PARSER_PATH = '../evaluator/CodeBLEU/'
DATA_PATH = '../data/'
PYTHON_LANGUAGE = Language(f'{PARSER_PATH}parser/my-languages.so', 'python')

KEY_WORDS = ['import', 'from', 'class', 'def', 'lambda', 'with', 'as', 'global', 'nonlocal', 'del', 'async']
CONDITION = ['for', 'while', 'if', 'elif', 'else', 'try', 'assert']
JUMP_TO = ['pass', 'break', 'await', 'raise', 'continue', 'yield', 'return', 'except', 'finally']
LOGIC = ['True', 'False', 'and', 'or', 'not', 'is', 'in', 'None']
OPERATOR = ['+', '-', '*', '**', '/', '//', '%', '@', '<<', '>>', '&', '|', '^',
            '~', ':=', '<', '>', '<=', '>=', '==', '!=']
DELIMITER = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '=', '->',
             '+=', '-=', '*=', '/=', '//=', '%=', '@=', '&=', '|=', '^=', '>>=', '<<=', '**=']
SPECIAL = ['\'', '"', '#', '\\']
label_dict = {
    'key': KEY_WORDS,
    'cond': CONDITION,
    'jump': JUMP_TO,
    'logic': LOGIC,
    'operator': OPERATOR,
    'delimiter': DELIMITER,
    'special': SPECIAL
}
token_dict = {token: k for k, v in label_dict.items() for token in v}


parser = Parser()
parser.set_language(PYTHON_LANGUAGE)
model = FastHan(model_type="large")
model.set_device('cuda:0')
# model.add_user_dict(["LSTM", ])
# model.set_user_dict_weight(0.05)


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


class Record:

    def __init__(self, textCN, textEN, code, api, sizeIn, sizeOut, id, ref_id, type, CN_token=None, EN_token=None, code_token=None, code_label=None):
        self.textCN = textCN
        self.textEN = textEN
        self.code = code
        self.api = api
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.id = id
        self.ref_id = ref_id
        self.type = type
        self.CN_token = CN_token
        self.EN_token = EN_token
        self.code_token = code_token
        self.code_label = code_label

    def __hash__(self):
        return hash(self.textCN + self.textEN + self.code)

    def __eq__(self, other):
        if self.textCN == other.textCN and self.textEN == other.textEN and self.code == other.code:
            return True
        else:
            return False

    def item(self):
        return [self.textCN, self.CN_token, self.textEN, self.EN_token, self.code, self.code_token, self.code_label,
                self.api, self.sizeIn, self.sizeOut, self.id, self.ref_id, self.type]


col_type = [str, str, str, eval, eval, eval, str, str, str]
col_name = ['textCN', 'CN_token', 'textEN', 'EN_token', 'code', 'code_token', 'code_label', 'api', 'sizeIn', 'sizeOut', 'id', 'ref_id', 'type']
obj_list = []
with open(f'{DATA_PATH}dataset/data.v0.6-all.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        row = list(convert(value) for convert, value in zip(col_type, row))
        record = Record(
            *row,
            CN_token=model(row[0], 'CWS'),
            EN_token=tokenize_docstring(row[1]),
            code_token=tokenize_code(parser.parse(row[2].encode()), row[2]),
            code_label=None
        )
        obj_list.append(record)


with jsonlines.open(f'{DATA_PATH}dataset.jsonl', mode='w') as writer:
    for obj in obj_list:
        writer.write(json.dumps(obj))

