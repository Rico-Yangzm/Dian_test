import random
guess=random.randint(1,100)
number=int(input("Guess a number between1 and 100: "))
time_tried=1
while number != guess:
    if number > guess:
        print("too high")
    else:
        print("too low")
    time_tried += 1
    number = int(input("Guess again: "))
print("congratulations! you have done it!")
print("using ",time_tried," times only.")