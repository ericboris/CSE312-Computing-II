def nCr(n, r):
	return (fact(n) / (fact(r) * fact(n - r)))
	
def fact(n):
	res = 1
	for i in range(2, n+1):
		res = res * i
	return res
	
def form(n, p, k):
	return nCr(n, k) * (p ** k) * (1 - p) ** (n - k)
