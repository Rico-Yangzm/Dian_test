def main():
    a=float(input("enter the first  number:"))
    calculate=str(input("+ or - or * or / or mod :"))
    b=float(input("enter the second number:"))
    tag=1
    c=None

    match calculate:
        case "+": c=a+b
        case "-": c=a-b
        case "*": c=a*b
        case "/":

                if b==0:
                    print("error")
                    tag=0
                else: c=a/b
        case "mod": c=int(a%b)
    if c is not None:
        return tag,c
    else:
        return tag,None

(t,result)=main()
if t :
    print("the result is:", result)
else:
    pass