from roman import to_roman, from_roman

assert to_roman(4) == "IV"
assert to_roman(9) == "IX"
assert to_roman(58) == "LVIII"
assert to_roman(1994) == "MCMXCIV"
assert from_roman("IV") == 4
assert from_roman("LVIII") == 58
assert from_roman("MCMXCIV") == 1994
for n in (1, 49, 944, 2023, 3999):
    assert from_roman(to_roman(n)) == n

print("ok")
