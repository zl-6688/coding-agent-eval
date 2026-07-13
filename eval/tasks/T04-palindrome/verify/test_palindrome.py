from palindrome import is_palindrome

assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("hello") is False
assert is_palindrome("") is True
assert is_palindrome("Was it a car or a cat I saw?") is True
assert is_palindrome("ab") is False

print("ok")
