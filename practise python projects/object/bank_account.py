class Account:
    def __init__(self, name, money):
        self.name = name
        self.money = money

    def withdraw(self, amount):
        if self.money >= amount:
            self.money -= amount
            print(f"You've successfully withdraw {amount} dollars.")
            print(f"Congras!")
            print(f"There's {self.money} left in your account")
        else:
            print(f"You don't have enough money.")
            print(f"Maybe you can try to get a few less than {self.money}")

    def deposit(self, amount):
        print(f"You've successfully deposit {amount} dollars.")
        self.money += amount
        print(f"Congras!")
        print(f"There's {self.money} left in your account")


def main():
    names={"han":Account("han", 100000000)}
    while True:
        print(f"\nWelcome to bank Yang! May I help you?")
        print("1.withdraw")
        print("2.deposit")
        print("3.exit")
        operation = int(input("enter the number to choose:"))
        if operation == 3:
            print("quit")
            break
        name = input("your name:")
        if name in names:
            account = names[name]
            match operation:
                case 1:
                    amount = int(input("how much do you want to withdraw: "))
                    account.withdraw(amount)
                case 2:
                    amount = int(input("how much do you want to deposit:"))
                    account.deposit(amount)
                case _:
                    print("unknown operation, please retry")
        else:
            print("you haven't got an account yet, input your information to create.")
            money = int(input("how much money do you have already:"))
            names[name] = Account(name, money)
            continue
main()