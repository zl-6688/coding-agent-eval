import inspect

import calc

src = inspect.getsource(calc)
assert "eval(" not in src and "exec(" not in src, "不允许使用 eval/exec"

from calc import evaluate

assert evaluate("2+3*4") == 14
assert evaluate("(2+3)*4") == 20
assert evaluate("10/4") == 2.5
assert evaluate("2*(3+4)-1") == 13
assert evaluate("100") == 100
assert evaluate("2 + 3 * (4 - 1)") == 11

print("ok")
