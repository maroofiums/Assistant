def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")

    if not expression or not set(expression) <= allowed:
        return "Invalid Expression"

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))

    except Exception:
        return "Calucation Error"
