def calculate(expression: str):
    try:
        return eval(expression)
    except Exception as e:
        return f"计算错误:{e}"