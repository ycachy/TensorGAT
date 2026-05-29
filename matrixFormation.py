import copy
import random
import json
import sys
import time, itertools
import os, sys, os.path as osp
import argparse, torch
import numpy as np
from torch_geometric.data import Dataset, Data, DataLoader
import torch
'''

'>>=', 'continue_statement', '|', '(', '#else', 'preproc_def', 'continue', '<<', '-', '>', 'preproc_arg', 'abstract_pointer_declarator', 'parenthesized_expression', 'preproc_function_def', 'preproc_directive', 'for', 'field_declaration_list', 'inline', '#ifdef', 'parameter_list', 'return', 'type_descriptor', 'preproc_ifdef', '~', 'comma_expression', 'compound_literal_expression', 'parameter_declaration', '#endif', 'null', 'declaration', 'assignment_expression', 'return_statement', 'goto_statement', '!', ']', 'type_qualifier', '}', 'preproc_include', '+', '-=', 'enum', 'break_statement', 'string_literal', 'preproc_else', '#if', 'unsigned', 'system_lib_string', ',', 'else', "'", '%=', '>=', '&=', 'enumerator_list', 'parenthesized_declarator', 'typedef', 'call_expression', 'escape_sequence', 'signed', '<', 'initializer_list', 'if', '|=', '...', 'unary_expression', 'type_identifier', '#ifndef', 'subscript_expression', 'number_literal', 'cast_expression', 'primitive_type', '<<=', 'sized_type_specifier', 'default', 'register', '#define', 'false', 'if_statement', '++', 'goto', '->', '^=', 'break', 'conditional_expression', '*', 'update_expression', '==', '--', 'init_declarator', '=', '||', 'preproc_call', '&', 'switch_statement', '[', 'expression_statement', '+=', 'true', '{', 'type_definition', 'preproc_params', 'statement_identifier', 'static', 'preproc_if', 'array_declarator', 'do_statement', 'while_statement', ';', 'pointer_expression', 'ERROR', '\n', 'switch', '!=', 'macro_type_specifier', '.', 'sizeof_expression', 'abstract_array_declarator', '"', 'while', 'abstract_function_declarator', 'long', ':', ')', 'short', 'binary_expression', 'const', 'enumerator', '*=', '&&', '%', 'struct_specifier', 'argument_list', 'function_declarator', 'case_statement', 'concatenated_string', 'char_literal', '/=', '^', 'identifier', 'function_definition', 'field_declaration', 'field_identifier', '<=', 'for_statement', 'sizeof', 'case', '/', '>>', 'field_expression', '?', 'translation_unit', 'do', 'abstract_parenthesized_declarator', 'pointer_declarator', 'struct', 'labeled_statement', 'enum_specifier', 'storage_class_specifier', 'compound_statement', '#include'
'''
# from sklearn.metrics import f1_score, precision_score, recall_score
pyTypesOfStatements = ['case', '{{', '**=', '"', 'type_conversion', 'ellipsis', ':=', 'named_expression', 'comment', 'await', 'async', 'import_prefix', 'relative_import', 'lambda_parameters', '{', 'except', 'string', 'not_operator', 'assert', 'else_clause', 'continue', 'print', 'keyword_argument', 'in', 'try_statement', 'aliased_import', 'not', '>=', 'try', 'parameters', '%', 'subscript', '&', '}', 'list_splat_pattern', '<<', 'exec_statement', 'assignment', 'typed_default_parameter', 'import_from_statement', 'nonlocal', 'binary_operator', 'module', 'with', 'concatenated_string', '*=', 'pass', '>', '|=', 'ERROR', '//=', ',', 'unary_operator', '+', 'conditional_expression', 'break_statement', 'argument_list', '__future__', 'for_in_clause', '<=', 'identifier', '^=', 'nonlocal_statement', 'function_definition', 'return', 'tuple_pattern', ')', 'lambda', '>>=', 'for_statement', 'class', '=', '+=', '[', 'continue_statement', 'as_pattern', '<<=', 'default_parameter', 'while_statement', 'exec', 'future_import_statement', 'as_pattern_target', 'comparison_operator', '&=', 'false', 'as', 'if_statement', 'elif_clause', '|', 'finally_clause', '<>', '^', 'set_comprehension', '*', 'list_comprehension', 'assert_statement', 'keyword_separator', '!=', 'global', 'typed_parameter', 'call', 'true', ':', 'if', 'boolean_operator', 'list_splat', 'pattern_list', '@', 'if_clause', '->', 'attribute', 'except_clause', 'integer', 'decorated_definition', 'generator_expression', 'set', 'dictionary_splat_pattern', 'def', 'del', 'decorator', ';', 'delete_statement', 'parenthesized_expression', '==', 'dictionary', 'with_item', 'and', '.', '/=', 'list', 'dotted_name', 'print_statement', 'dictionary_splat', 'chevron', 'pair', 'elif', 'block', 'yield', 'with_statement', 'wildcard_import', 'import_statement', 'finally', 'expression_list', 'tuple', 'float', 'none', 'with_clause', '~', 'is', 'pass_statement', 'global_statement', 'raise_statement', 'augmented_assignment', 'expression_statement', '-', ']', 'or', 'class_definition', 'break', 'list_pattern', '(', 'return_statement', 'for', 'raise', 'slice', '%=', 'type', '-=', '<', 'dictionary_comprehension', '>>', 'else', 'from', 'while', '/', '//', 'import', '**']
javaTypeofStatements = ['yield_statement','yield','receiver_parameter','named_expression','strictfp','annotated_type','class_literal', 'transient', 'volatile','synchronized_statement','annotation_type_element_declaration', 'annotation_type_body', '@interface', 'annotation_type_declaration', 'element_value_pair', 'for', '->', 'throw', 'throws', 'interface_declaration', '<<=', '>=', 'private', '[', 'modifiers', 'final', 'resource', 'dimensions_expr', 'return_statement', '=', 'void_type', 'switch_label', 'interface', 'continue_statement', 'package_declaration', 'asterisk', 'protected', 'decimal_floating_point_literal', '!', 'inferred_parameters', '&=', 'element_value_array_initializer', 'assignment_expression', 'char', 'finally', 'argument_list', 'short', 'class', 'catch_clause', '--', 'package', 'do', ':', 'field_declaration', '>>', 'assert_statement', 'marker_annotation', 'enum_constant', '?', 'object_creation_expression', 'formal_parameter', 'false', '*', '::', '>', 'true', 'character_literal', 'hex_integer_literal', '{', 'public', 'if_statement', '%=', 'byte', ']', 'static', '/', 'floating_point_type', 'method_reference', 'while_statement', 'constant_declaration', 'throw_statement', '+=', 'enum', 'formal_parameters', 'do_statement', 'enum_body', 'null_literal', ',', 'type_parameter', 'binary_integer_literal', '*=', 'lambda_expression', 'field_access', 'long', 'parenthesized_expression', '^=', 'string_literal', '|=', 'enhanced_for_statement', 'array_creation_expression', 'implements', 'import_declaration', 'finally_clause', '~', 'program', 'catch_type', 'super_interfaces', 'dimensions', '>>=', 'generic_type', 'return', '++', '...', 'array_access', 'spread_parameter', 'while', 'constructor_declaration', 'scoped_type_identifier', 'assert', 'resource_specification', 'method_invocation', 'local_variable_declaration', 'integral_type', 'if', 'break', 'instanceof_expression', 'interface_body', '-=', '(', 'ERROR', '>>>', 'boolean_type', 'static_initializer', 'expression_statement', '!=', 'enum_body_declarations', 'ternary_expression', '|', 'else', '-', 'class_declaration', 'method_declaration', 'enum_declaration', 'identifier', 'type_bound', 'block', 'case', 'continue', 'type_identifier', '.', 'try_with_resources_statement', 'binary_expression', 'decimal_integer_literal', 'break_statement', '@', ')', 'array_initializer', 'annotation_argument_list', 'switch', 'default', 'try_statement', 'with', '^', '>>>=', 'constructor_body', 'super', 'to', 'wildcard', 'int', '&&', '/=', 'switch_expression', 'catch_formal_parameter', 'import', '<=', 'double', 'unary_expression', ';', 'type_parameters', 'synchronized', 'extends_interfaces', 'float', 'annotation', '%', 'catch', 'cast_expression', 'explicit_constructor_invocation', 'switch_block', '<<', 'superclass', 'new', 'labeled_statement', '||', 'scoped_identifier', 'instanceof', 'class_body', 'switch_block_statement_group', 'type_list', 'abstract', '==', 'this', 'extends', 'variable_declarator', '+', 'try', '}', 'update_expression', '<', '&', 'type_arguments', 'for_statement', 'array_type']

cTpyeofStatements = ['bitfield_clause', '::', 'field_designator', 'ms_pointer_modifier', '#elif', 'preproc_elif', 'restrict', 'ms_restrict_modifier', '>>=', 'continue_statement', '|', '(', '#else', 'preproc_def', 'continue', '<<', '-', '>', 'preproc_arg', 'abstract_pointer_declarator', 'parenthesized_expression', 'preproc_function_def', 'preproc_directive', 'for', 'field_declaration_list', 'inline', '#ifdef', 'parameter_list', 'return', 'type_descriptor', 'preproc_ifdef', '~', 'comma_expression', 'compound_literal_expression', 'parameter_declaration', '#endif', 'null', 'declaration', 'assignment_expression', 'return_statement', 'goto_statement', '!', ']', 'type_qualifier', '}', 'preproc_include', '+', '-=', 'enum', 'break_statement', 'string_literal', 'preproc_else', '#if', 'unsigned', 'system_lib_string', ',', 'else', "'", '%=', '>=', '&=', 'enumerator_list', 'parenthesized_declarator', 'typedef', 'call_expression', 'escape_sequence', 'signed', '<', 'initializer_list', 'if', '|=', '...', 'unary_expression', 'type_identifier', '#ifndef', 'subscript_expression', 'number_literal', 'cast_expression', 'primitive_type', '<<=', 'sized_type_specifier', 'default', 'register', '#define', 'false', 'if_statement', '++', 'goto', '->', '^=', 'break', 'conditional_expression', '*', 'update_expression', '==', '--', 'init_declarator', '=', '||', 'preproc_call', '&', 'switch_statement', '[', 'expression_statement', '+=', 'true', '{', 'type_definition', 'preproc_params', 'statement_identifier', 'static', 'preproc_if', 'array_declarator', 'do_statement', 'while_statement', ';', 'pointer_expression', 'ERROR', '\n', 'switch', '!=', 'macro_type_specifier', '.', 'sizeof_expression', 'abstract_array_declarator', '"', 'while', 'abstract_function_declarator', 'long', ':', ')', 'short', 'binary_expression', 'const', 'enumerator', '*=', '&&', '%', 'struct_specifier', 'argument_list', 'function_declarator', 'case_statement', 'concatenated_string', 'char_literal', '/=', '^', 'identifier', 'function_definition', 'field_declaration', 'field_identifier', '<=', 'for_statement', 'sizeof', 'case', '/', '>>', 'field_expression', '?', 'translation_unit', 'do', 'abstract_parenthesized_declarator', 'pointer_declarator', 'struct', 'labeled_statement', 'enum_specifier', 'storage_class_specifier', 'compound_statement', '#include', 'initializer_pair', 'volatile', 'attribute_specifier', 'extern', '__attribute__', 'subscript_designator', 'variadic_parameter', 'auto', 'preproc_defined', 'defined', 'union_specifier', 'union', ']]', 'attribute', '[[', 'attributed_declarator', 'attribute_declaration', 'comment','yield_statement']

cppTpyeofStatements = ['throw_specifier','.*','field_designator', 'initializer_pair', 'signed', 'new_declarator', 'class_specifier', '>=', 'expression_statement', ')', 'primitive_type', 'new', 'concatenated_string', 'attribute_declaration', 'long', '#include', 'break', '...', 'parameter_declaration', 'pointer_declarator', '\n', 'try_statement', '[', 'private', 'declaration_list', 'dependent_type', 'if', 'optional_type_parameter_declaration', 'false', ';', 'preproc_params', '?', 'variadic_type_parameter_declaration', 'lambda_expression', 'subscript_expression', 'string_literal', 'argument_list', '+', 'delete_method_clause', ':', 'template_parameter_list', 'preproc_def', '>>=', 'template_argument_list', 'raw_string_literal', 'enum_specifier', 'type_definition', 'namespace_definition', '&&', 'call_expression', 'update_expression', '++', 'do', 'type_qualifier', 'binary_expression', 'field_declaration_list', 'optional_parameter_declaration', 'friend_declaration', 'structured_binding_declarator', 'continue_statement', 'defined', 'if_statement', 'goto_statement', 'preproc_call', 'typedef', '--', "'", '.', 'decltype', '#ifdef', 'static', '/=', 'template_type', 'return', 'namespace', 'true', 'null', '{', 'statement_identifier', 'operator_cast', '+=', 'variadic_parameter_declaration', 'template_declaration', 'placeholder_type_specifier', 'trailing_return_type', '->', 'type_identifier', 'parenthesized_expression', 'init_declarator', 'pointer_expression', 'condition_clause', 'const', '|', 'constexpr', '"', '()', '^', 'public', 'type_parameter_declaration', 'sizeof_expression', 'literal_suffix', 'translation_unit', 'case', '#else', 'explicit', 'conditional_expression', 'enum', 'alias_declaration', 'base_class_clause', 'unsigned', 'requires', '::', 'catch_clause', 'switch_statement', '==', 'attribute', '*=', '=', 'field_expression', 'parameter_pack_expansion', 'reference_declarator', '>', 'preproc_arg', '%=', 'short', 'sizeof', ',', '__attribute__', '*', 'delete', 'enumerator', 'while', 'qualified_identifier', 'else', 'unary_expression', 'char_literal', '}', '<<', 'preproc_directive', 'do_statement', '-', '#if', 'new_expression', '!', 'struct_specifier', 'abstract_array_declarator', 'using_declaration', 'preproc_ifdef', 'default', 'ERROR', 'preproc_else', 'escape_sequence', 'compound_statement', 'field_initializer_list', '|=', 'comma_expression', 'sized_type_specifier', '||', 'explicit_function_specifier', '#define', '#ifndef', 'abstract_parenthesized_declarator', ']]', '<=', 'break_statement', 'operator', 'identifier', 'preproc_function_def', 'template_method', 'attribute_specifier', 'catch', 'class', 'abstract_pointer_declarator', 'declaration', 'struct', 'typename', '<<=', 'this', 'type_descriptor', 'field_initializer', 'variadic_declarator', 'try', 'lambda_default_capture', 'requires_clause', 'user_defined_literal', 'parenthesized_declarator', 'number_literal', 'delete_expression', 'namespace_identifier', '&=', '(', 'auto', 'namespace_alias_definition', 'abstract_reference_declarator', 'return_statement', '-=', 'case_statement', ']', '#endif', 'function_declarator', '<', 'switch', 'friend', 'for', 'preproc_if', '%', 'preproc_defined', 'cast_expression', 'abstract_function_declarator', 'compound_literal_expression', 'template_template_parameter_declaration', '/', 'array_declarator', 'enumerator_list', '!=', 'goto', 'while_statement', 'nullptr', '~', 'initializer_list', 'system_lib_string', 'lambda_capture_specifier', 'template', 'inline', 'labeled_statement', 'function_definition', 'template_function', 'storage_class_specifier', 'for_range_loop', 'destructor_name', 'preproc_include', 'for_statement', '&', '^=', 'field_declaration', '>>', 'access_specifier', 'parameter_list', 'operator_name', 'field_identifier', '[[', 'continue', 'using', 'assignment_expression', 'virtual_specifier', 'u"', 'mutable', '""', 'volatile', 'ms_restrict_modifier', 'override', 'ms_pointer_modifier', 'union_specifier', 'default_method_clause', 'comment', 'dependent_name', 'register', 'linkage_specification', '#elif', 'extern', 'static_assert_declaration', 'bitfield_clause', 'static_assert', 'preproc_elif', 'virtual', 'noexcept', 'throw', 'throw_statement', 'ref_qualifier', 'virtual_function_specifier', 'protected', '[]', 'union', 'subscript_designator', 'final', 'L"', 'restrict', 'thread_local', '->*', 'fold_expression']
# discuss req of padding??
# maxLength = max(len(pyTypesOfStatements), len(cTpyeofStatements))
maxLength=173
print(maxLength)
import json
import numpy as np


# will give an encoded matrix for setofStatements
def oneHotEncoder(setofStatements, langType):
    statements = json.loads(setofStatements)
    statements_dict = statements["nodeList"]
    # 根据语言选择语句类型列表
    if langType == "java":
        type_list = javaTypeofStatements
    elif langType == "python":
        type_list = pyTypesOfStatements
    elif langType == "c++":
        type_list = cppTpyeofStatements
    else:
        type_list = cTpyeofStatements

        # 初始化编码矩阵
    encodeMatrix = np.zeros((len(statements_dict), maxLength), dtype='int32')

    # 遍历字典中的节点
    for i in range(len(statements_dict)):
        idx = type_list.index(statements_dict[i]["type"])
        encodeMatrix[i][idx] = 1
        # for node_id, node_data in statements_dict.items():
    #     node_type = node_data['type']  # 获取节点的类型
    #
    #     # 找到节点类型在语句类型列表中的索引
    #     idx = type_list.index(node_type)
    #     encodeMatrix[int(node_id)][idx] = 1  # 更新对应的编码矩阵位置

    return encodeMatrix


# prepares the adjacency matrix
def adjacencyMatrixCreator(setofStatements):
    jsonArrayConverted = json.loads(setofStatements)
    num_nodes = len(jsonArrayConverted)
    srcArr = []
    desArr = []
    edge_type = []
    for i in range(len(jsonArrayConverted)):
        if ('children' in jsonArrayConverted[i]):
            listOfChildren = jsonArrayConverted[i]['children']
            parentId = int(jsonArrayConverted[i]['id'])

            # directed edges / undirected edges ??
            for child in listOfChildren:
                childId = int(child)
                srcArr.append(parentId)
                desArr.append(childId)
                edge_type.append(0)
    nsrcArr = np.array(srcArr).astype('int32')
    ndesArr = np.array(desArr).astype('int32')
    adjacencyMatrix = np.row_stack((nsrcArr, ndesArr, edge_type))
    return adjacencyMatrix, num_nodes


'''
  时间2024/11/11
  高维Tensor建立
'''


def build_edges_matrix(edges, edge_type):
    edges_matric = [[], [], []]  # 多增加一个维度用于保存边的类型
    for edge in edges:
        source_vertex_id = edge[0]
        target_vertex_id = edge[1]
        edges_matric[0].append(source_vertex_id)
        edges_matric[1].append(target_vertex_id)
        edges_matric[2].append(edge_type)  # 保存边的类型
    return edges_matric
def High_latitudestensor(ast, nodes_number_limit=600):
  astOfProgram = json.loads(ast)

  # 构建不同类型矩阵
  ast_matric = build_edges_matrix(astOfProgram.get('astEdges', []), edge_type=0)
  cfg_matric = build_edges_matrix(astOfProgram.get('cfgEdges', []), edge_type=1)
  ddg_matric = build_edges_matrix(astOfProgram.get('dfgEdges', []), edge_type=2)
  ncs_matric = build_edges_matrix(astOfProgram.get('ncsEdges', []), edge_type=3)

  # 如果相应的边缘列表为空，则使用astEdges构建矩阵
  if len(astOfProgram.get('cfgEdges', [])) == 0:
    cfg_matric = build_edges_matrix(astOfProgram.get('astEdges', []), edge_type=1)
  if len(astOfProgram.get('ddgEdges', [])) == 0:
    ddg_matric = build_edges_matrix(astOfProgram.get('astEdges', []), edge_type=2)
  if len(astOfProgram.get('ncsEdges', [])) == 0:
    ncs_matric = build_edges_matrix(astOfProgram.get('astEdges', []), edge_type=3)

  # 将矩阵打包成字典返回
  matrices = {
    'ast_matric': ast_matric,
    'cfg_matric': cfg_matric,
    'dfg_matric': ddg_matric,
    'ncs_matric': ncs_matric
  }
  return matrices
