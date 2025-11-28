take = str(input("Enter string :"))
volwels = take.upper()
count = 0
volwels_check = ["A","E","I","O","U"]
for i in range(0,len(volwels_check)):
    for j in range(0,len(volwels)):
        if volwels_check[i] == volwels[j]:
            count +=1
print(count)

