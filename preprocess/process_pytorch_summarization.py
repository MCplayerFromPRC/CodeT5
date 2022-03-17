import sys
import csv
import re
import random
import jsonlines
from typing import List
from tree_sitter import Parser, Language

sys.path.append("../")
DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
PARSER_PATH = '../evaluator/CodeBLEU/'
DATA_PATH = '../data/'
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


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


class record():
    def __init__(self, textCN, textEN, code, api, sizeIn, sizeOut, id, ref_id):
        self.textCN = textCN
        self.textEN = textEN
        self.code = code
        self.api = api
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.id = id
        self.ref_id = ref_id

    def __hash__(self):
        return hash(self.textCN + self.textEN + self.code + self.id)

    def __eq__(self, other):
        if self.textCN == other.textCN and self.textEN == other.textEN and self.code == other.code and self.id == other.id:
            return True
        else:
            return False

    def item(self):
        return [self.textCN, self.textEN, self.code, self.api, self.sizeIn, self.sizeOut, self.id, self.ref_id]


col_type = [str, str, str, eval, eval, eval, str, str]
long_desc_dict, long_num = {}, 0
with open(f'{DATA_PATH}dataset/data.v0.6-long.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        row = list(convert(value) for convert, value in zip(col_type, row))
        if not row[2].startswith("error code | "):
            long_num += 1
            if row[7] in long_desc_dict:
                long_desc_dict[row[7]].append(record(*row))
            else:
                long_desc_dict[row[7]] = [record(*row)]


train_group_info, valid_group_info, test_group_info = [], [], []

long_desc_dict = random_dic(long_desc_dict)

print(long_num)
train_num, valid_num, test_num = 0, 0, 0
for k, v in list(long_desc_dict.items()):
    if k not in [data.id for data in v]:
        long_num -= len(v)
        del long_desc_dict[k]
        continue
print(long_num)

for k, v in long_desc_dict.items():
    if train_num < long_num * 0.8:
        train_num += len(v)
        train_group_info.extend(v)
    elif valid_num < long_num * 0.1:
        valid_num += len(v)
        valid_group_info.extend(v)
    else:
        test_num += len(v)
        test_group_info.extend(v)


with jsonlines.open(f'{DATA_PATH}summarize/pytorch/train.jsonl', mode='w') as writer:
    for train_data in train_group_info:
        json_obj = {
            'idx': train_data.id,
            'code': train_data.code,
            'docstring': train_data.textEN,
            'code_tokens': tokenize_code(parser.parse(train_data.code.encode()), train_data.code),
            'docstring_tokens': tokenize_docstring(train_data.textEN)
        }
        writer.write(json_obj)

with jsonlines.open(f'{DATA_PATH}summarize/pytorch/valid.jsonl', mode='w') as writer:
    for valid_data in valid_group_info:
        json_obj = {
            'idx': valid_data.id,
            'code': valid_data.code,
            'docstring': valid_data.textEN,
            'code_tokens': tokenize_code(parser.parse(valid_data.code.encode()), valid_data.code),
            'docstring_tokens': tokenize_docstring(valid_data.textEN)
        }
        writer.write(json_obj)

with jsonlines.open(f'{DATA_PATH}summarize/pytorch/test.jsonl', mode='w') as writer:
    for test_data in test_group_info:
        json_obj = {
            'idx': test_data.id,
            'code': test_data.code,
            'docstring': test_data.textEN,
            'code_tokens': tokenize_code(parser.parse(test_data.code.encode()), test_data.code),
            'docstring_tokens': tokenize_docstring(test_data.textEN)
        }
        writer.write(json_obj)

