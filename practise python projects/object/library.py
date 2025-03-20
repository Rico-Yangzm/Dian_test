import re

def borrowbook(book, days, name, borrow_list):
    name = book + "_" + name
    borrow_list[name] = days
    return borrow_list

def returnbook(name, borrow_list):
    del(borrow_list[name])
    return borrow_list

def time_pass(borrow_list):
    for book in borrow_list:
        borrow_list[book] -= 1
    return borrow_list

class Library:
    def __init__(self, file_path):
        book_dict = {}
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            pattern = r'(.+?)\s+(\d+)'

            books = re.findall(pattern, text)
            for book in books:
                name = book[0]
                numbers = int(book[1])
                book_dict[name] = numbers
        self.list = book_dict

    def borrow_book(self, name, days, borrow_list):
        if name in self.list:
            if self.list[name] > 0:
                self.list[name] -= 1
                user_name = input("please tell me your name to register:")
                print(f"you have borrowed {name} for {days} days")
                borrow_list = borrowbook(name, days, user_name, borrow_list)
            else:
                print(f"sorry, there's no '{name}' left in the library")
                print("maybe you can try borrow another one")
        else:
            print(f"sorry, the book {name} is not available in our library")
        return borrow_list,self

    def return_book(self, book, name, borrow_list):
        tag = book + "_" + name
        if tag in borrow_list:
            self.list[book] += 1
            if tag in borrow_list:
                if borrow_list[tag] >= 0:
                    print("thank you for your cooperation")
                    borrow_list = returnbook(tag, borrow_list)
                else:
                    print("sorry, you have return it a bit later")
                    print(f"you have to pay {0.5 * abs(borrow_list[tag])} dollars for that")
                    borrow_list = returnbook(tag, borrow_list)
            else:
                print("you haven't borrow that book from our library")
        else:
            print("the book is not in the library")
        return borrow_list, self

    def buy_book(self, book):
        if book in self.list:
            if self.list[book] > 0:
                self.list[book] -= 1
                print("you've successfully buy it")
            else:
                print(f"sorry, there's no '{book}' left in the library")


def main():
    print("welcome to yang's library!")
    print("may I help you?")
    library = None
    borrow_list = {}
    while True:
        print("1.input the book list")
        print("2.borrow a book")
        print("3.return a book")
        print("4.buy a book")
        print("5.the next day")
        print("6.exit")
        operation = int(input("enter the number to choose:"))
        if library is not None:
            match operation:
                case 1:
                    file_path = input("input the file path of the list：")
                    library = Library(file_path)
                    continue
                case 2:
                    book = input("the book you want to borrow:")
                    day = int(input("how many days do you want to borrow:"))
                    (borrow_list, library) = library.borrow_book(book, day, borrow_list)
                    continue
                case 3:
                    book = input("the book you want to return:")
                    name = input("your name please:")
                    (borrow_list, library) = library.return_book(book, name, borrow_list)
                    continue
                case 4:
                    book = input("the book you want to buy:")
                    library.buy_book(book)
                case 5:
                    time_pass(borrow_list)
                    print("tomorrow......")
                case 6:
                    print("thanks for using")
                    break
                case _:
                    print("unknown operation please retry")
                    continue
        else:
            print("please input the file path of the list first")
            file_path = input("input the file path of the list：")
            library = Library(file_path)
            continue
main()


