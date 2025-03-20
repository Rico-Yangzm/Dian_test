def manage_system():
    students={}
    while True:
        print("\nplease choose your operation:")
        print("1.add student's score")
        print("2.change student's score")
        print("3.delete student's score")
        print("4.search student's score")
        print("5.exit")
        operation = int(input("enter the number to choose:"))
        match operation:
            case 1:
                name = input("student's name:")
                score=int(input("score:"))
                if name not in students:
                    students[name] = score
                    print(f"student {name}'s score is loaded")
                else:
                    print(f"student {name}'s score is already loaded in the system before")
                    continue
            case 2:
                name = input("student's name:")
                score = input("new score:")
                if name in students:
                    students[name] = score
                    print(f"student {name}'s score is changed")
                else:
                    print(f"student {name}'s score is not in the system.")
                    print(f"do you want to load it?")
                    print(f"1.load")
                    print(f"2.don't load")
                    load = int(input("enter the number to choose:"))
                    match load:
                        case 1:
                            students[name] = score
                            print(f"student {name}'s score is loaded")
                        case 2:
                            continue
            case 3:
                name = input("student's name:")
                if name in students:
                    del(students[name])
                    print("student {name}'s score is deleted")
                else:
                    print(f"student {name}'s score is not in the system.")
            case 4:
                name = input("student's name:")
                if name in students:
                    print(f"student {name}'s score is {students[name]}")
                else:
                    print(f"student {name}'s score is not in the system.")
            case 5:
                print("quit system")
                break
            case _:
                print("Invalid operation.")


manage_system()