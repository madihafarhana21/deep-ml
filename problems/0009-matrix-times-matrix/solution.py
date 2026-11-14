def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    if len(a[0]) != len(b):
        return -1

    c = []
    for row in a:
        new_row = []
        for col in zip(*b):
            total = 0
            for x, y in zip(row, col):
                total += x*y
            new_row.append(total)
        c.append(new_row)
    return c