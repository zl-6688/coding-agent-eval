from fizzbuzz import fizzbuzz

r = fizzbuzz(15)
assert len(r) == 15
assert r[0] == "1"
assert r[1] == "2"
assert r[2] == "Fizz"
assert r[4] == "Buzz"
assert r[14] == "FizzBuzz"

print("ok")
