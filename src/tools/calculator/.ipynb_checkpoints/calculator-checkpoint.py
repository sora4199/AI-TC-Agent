import ast
import operator


class Calculator:
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def calculate(self, expression: str):
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)

            return {
                "result": result
            }

        except ZeroDivisionError:
            return {
                "error": "Division by zero"
            }

        except (ValueError, SyntaxError):
            return {
                "error": f"Invalid expression: {expression}"
            }

    def _evaluate(self, node):

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numbers are allowed")

        # Binary operations: +, -, *, /
        if isinstance(node, ast.BinOp):

            if type(node.op) not in self.OPERATORS:
                raise ValueError(
                    "Only +, -, *, / are supported"
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return self.OPERATORS[type(node.op)](
                left,
                right
            )

        # Positive / negative numbers
        if isinstance(node, ast.UnaryOp):

            if isinstance(node.op, ast.USub):
                return -self._evaluate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return self._evaluate(node.operand)

        raise ValueError(
            "Unsupported expression"
        )


if __name__ == "__main__":

    calculator = Calculator()

    test_expressions = [
        "125 + 37",
        "125 - 37",
        "125 * 37",
        "125 / 37",
        "(125 + 37) * 2 / 4",
        "-10 + 5",
        "10 * (5 + 3)",
        "(100 - 20) / (5 + 3)",
    ]

    print("=== CALCULATOR TEST ===")

    for expression in test_expressions:
        result = calculator.calculate(expression)

        print(f"{expression}")
        print(f"-> {result}")
        print()