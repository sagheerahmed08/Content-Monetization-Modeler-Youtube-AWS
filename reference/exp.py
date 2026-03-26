n = int(input())
arr = list(map(int, input().split()))

result = 0
for num in arr:
    result ^= num

print(result)