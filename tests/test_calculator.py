"""
CalculatorTool 单元测试

覆盖：基础运算、各运算符、一元运算、安全防护（非法字符/危险调用）、错误表达式
"""
import pytest
from core.tools import CalculatorTool


@pytest.fixture
def calc():
    return CalculatorTool()


class TestCalculatorBasic:
    """基础运算"""

    def test_addition(self, calc):
        r = calc.run(expression="1 + 2")
        assert r["result"] == 3

    def test_subtraction(self, calc):
        r = calc.run(expression="10 - 4")
        assert r["result"] == 6

    def test_multiplication(self, calc):
        r = calc.run(expression="3 * 5")
        assert r["result"] == 15

    def test_division(self, calc):
        r = calc.run(expression="20 / 4")
        assert r["result"] == 5.0

    def test_modulo(self, calc):
        r = calc.run(expression="17 % 5")
        assert r["result"] == 2

    def test_power(self, calc):
        r = calc.run(expression="2 ** 3")
        assert r["result"] == 8


class TestCalculatorComplex:
    """复合表达式与优先级"""

    def test_operator_precedence(self, calc):
        # 2 + 3 * 4 == 14，验证优先级
        r = calc.run(expression="2 + 3 * 4")
        assert r["result"] == 14

    def test_parentheses(self, calc):
        # (2 + 3) * 4 == 20，验证括号
        r = calc.run(expression="(2 + 3) * 4")
        assert r["result"] == 20

    def test_unary_minus(self, calc):
        r = calc.run(expression="-5")
        assert r["result"] == -5

    def test_unary_plus(self, calc):
        r = calc.run(expression="+5")
        assert r["result"] == 5

    def test_nested_expression(self, calc):
        # 复杂嵌套
        r = calc.run(expression="(1 + 2) * (3 + 4) - 5")
        assert r["result"] == 16

    def test_whitespace_tolerant(self, calc):
        # 多空格应被正确处理
        r = calc.run(expression="  1   +   2  ")
        assert r["result"] == 3


class TestCalculatorSecurity:
    """
    安全防护测试

    CalculatorTool 用 AST 安全解析器替代 eval，必须拒绝：
    - 非法字符
    - 函数调用（如 __import__）
    - 变量名
    """
    def test_rejects_letters(self, calc):
        r = calc.run(expression="abc")
        assert "error" in r

    def test_rejects_function_call(self, calc):
        # 经典 eval 注入尝试，必须被拒绝
        r = calc.run(expression="__import__('os').system('echo hacked')")
        assert "error" in r

    def test_rejects_variable(self, calc):
        r = calc.run(expression="x + 1")
        assert "error" in r

    def test_returns_expression_in_error(self, calc):
        # 错误时仍应回传原始表达式，便于排错
        r = calc.run(expression="1 / 0 + invalid")
        assert r["expression"] == "1 / 0 + invalid"

    def test_schema_valid(self, calc):
        """工具 schema 应符合 OpenAI function-calling 格式"""
        s = calc.to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "calculator"
        assert "expression" in s["function"]["parameters"]["properties"]
        assert "expression" in s["function"]["parameters"]["required"]
