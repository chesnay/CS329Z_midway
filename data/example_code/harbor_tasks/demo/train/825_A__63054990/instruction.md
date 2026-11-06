# 825/A: Binary Protocol

Your Python 3 program is `/app/solution.py`. It reads standard input and writes the answer to standard output.

Codeforces judged submission 63054990 as WRONG_ANSWER. It passed 10 tests. Fix the program in place. Do not change the file path.

Time limit: 1.0 seconds. Memory limit: 256.0 MB.

## Statement

Polycarp has just invented a new binary protocol for data transmission. He is encoding positive integer decimal number to binary string using following algorithm:

- Each digit is represented with number of '1' characters equal to the value of that digit (for 0 it is zero ones).
- Digits are written one by one in order corresponding to number and separated by single '0' character.

Though Polycarp learnt how to encode the numbers, he has no idea how to decode them back. Help him calculate the decoded number.

## Input

The first line contains one integer number n (1 ≤ n ≤ 89) — length of the string s.

The second line contains string s — sequence of '0' and '1' characters, number in its encoded format. It is guaranteed that the number corresponding to the string is positive and doesn't exceed 10^{9}. The string always starts with '1'.

## Output

Print the decoded number.


## Samples

Sample 1 input:
3
111

Sample 1 output:
3

Sample 2 input:
9
110011101

Sample 2 output:
2031

## Teacher feedback

try again
