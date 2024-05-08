import sleuthbuilder as sb

# Prompt the user for a binary string
user_input = input("Please enter a binary string: ")

# Ensure the input is a valid binary string
if all(c in '01' for c in user_input):
    p_value = sb.get_p_value_for_string(user_input)
    print("P-value:", p_value)
else:
    print("Invalid input. Please make sure to enter a binary string containing only 0s and 1s.")
