# utils/safe_eval.py
"""
安全的数学表达式求值器，基于 AST 白名单。
仅支持基本算术运算和部分数学函数，禁止任何危险操作。
"""

import ast
import operator
import math
from typing import Any, Dict

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt,
    'abs': abs,
}

class SafeEvaluator:
    @classmethod
    def evaluate(cls, expr: str, variables: Dict[str, Any]) -> Any:
        cls._sanitize(expr)
        try:
            tree = ast.parse(expr, mode='eval')
            return cls._eval_node(tree.body, variables)
        except Exception as e:
            raise ValueError(f"公式求值失败: {e}")

    @classmethod
    def _sanitize(cls, expr: str) -> None:
        import re
        expr = re.sub(r'#.*', '', expr)
        expr = re.sub(r'"""[\s\S]*?"""', '', expr)
        expr = re.sub(r"'''[\s\S]*?'''", '', expr)
        if '__' in expr:
            raise ValueError("公式禁止使用双下划线 '__'")
        if re.search(r'\bimport\b', expr):
            raise ValueError("公式禁止使用 import")
        if re.search(r'\b(exec|eval|compile|open|input)\b', expr):
            raise ValueError("公式禁止调用危险内置函数")

    @classmethod
    def _eval_node(cls, node, variables: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                return variables[name]
            elif name in _ALLOWED_FUNCTIONS:
                return _ALLOWED_FUNCTIONS[name]
            else:
                raise NameError(f"未定义的变量或函数: {name}")
        elif isinstance(node, ast.Attribute):
            raise TypeError("属性访问（如 np.log10）不被支持，请直接使用函数名（如 log10）")
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left, variables)
            right = cls._eval_node(node.right, variables)
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise TypeError(f"不支持的运算符: {op_type}")
            return _ALLOWED_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand, variables)
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise TypeError(f"不支持的一元运算符: {op_type}")
            return _ALLOWED_OPERATORS[op_type](operand)
        elif isinstance(node, ast.Call):
            func_obj = cls._eval_node(node.func, variables)
            if not callable(func_obj) or func_obj not in _ALLOWED_FUNCTIONS.values():
                raise TypeError(f"禁止调用的函数: {node.func.id if hasattr(node.func, 'id') else node.func}")
            args = [cls._eval_node(arg, variables) for arg in node.args]
            return func_obj(*args)
        else:
            raise TypeError(f"不支持的表达式节点: {type(node).__name__}")