# import pstats

# p = pstats.Stats('stats.txt')
# p.strip_dirs().sort_stats('cumulative').print_stats(20)\

# p.print_callers(20, '__getitem__')

even = [2, 4, 6, 8]
odd = [1, 3, 5, 7]
for even_i, odd_i in zip(even, odd):
    print(even_i, odd_i)