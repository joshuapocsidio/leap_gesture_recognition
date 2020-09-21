# Prints out a simple list
def print_list(str_list):
    for item in str_list:
        print(item)


# Prints out a numbered list
def print_numbered_list(str_list, starting_index=1):
    i = starting_index
    for item in str_list:
        print(str(i) + ' - ' + str(item))
        i += 1
