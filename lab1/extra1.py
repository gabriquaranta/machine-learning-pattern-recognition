class Book:
    def __init__(self) -> None:
        self.bought = 0
        self.sold = 0
        self.gain = 0
        self.loss = 0


# open file + read lines
with open("lab1/extra1_sales.txt") as inputfile:
    lines = inputfile.readlines()

# "MM/YY" : qty dic
mmyy_sales = {}
books = {}

# split entry into corresponding fields
for line in lines:
    fields = line.split(" ")
    # <ISBN> <BUY/SELL> <DATE> <#-OF-COPIES> <PRICE-PER-COPY>
    bookid = fields[0]
    action = fields[1]
    date = fields[2]
    mmyy = date[3:]
    qty = int(fields[3])
    price = float(fields[4])

    # update mmyy_sales
    if action == "S":
        if mmyy in mmyy_sales:
            mmyy_sales[mmyy] += qty
        else:
            mmyy_sales[mmyy] = qty

    # update books
    if bookid in books:
        if action == "S":
            books[bookid].sold += qty
            books[bookid].gain += price * qty
        else:
            books[bookid].bought += qty
            books[bookid].loss += price * qty
    else:
        books[bookid] = Book()
        if action == "S":
            books[bookid].sold += qty
            books[bookid].gain += price * qty
        else:
            books[bookid].bought += qty
            books[bookid].loss += price * qty

# pirint sale per mm/yy
print(mmyy_sales, "\n")
# print available per book
for i in books:
    print(i, books[i].bought - books[i].sold)
# print gain
print()
for i in books:
    b = books[i]
    print(i, b.gain - (b.loss / b.bought) * b.sold)
