def argchecker(kwargs, ArgCheckList):
    for arg in ArgCheckList:
        if not arg in kwargs:
            raise Exception ('You omit an essential argument registered in :', ArgCheckList)


def remove_last_zero_points(strings):
    num = len(strings)
    for idx in range(1, len(strings)):
        if strings[-idx] == '0':
            num -= 1
        elif strings[-idx] == '.':
            num -= 1
            break
        else:
            break
    return strings[:num]
