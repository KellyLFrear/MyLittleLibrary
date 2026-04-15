# Function To load A Word List From A Text File And Return It As A Set Of Lowercase Words (Stripping Whitespace)
def load_word_list(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


# Function for Main Execution To Load The Beginner, Intermediate, And Advanced Word Lists And Print Their Lengths To Verify They Were Loaded Correctly
def main():
    # Load The Word Lists From The Specified Text Files
    beginner = load_word_list("data/vocab/beginner_1000.txt")
    intermediate = load_word_list("data/vocab/intermediate_3000.txt")
    advanced = load_word_list("data/vocab/advanced_6000.txt")

    # Print The Number Of Words Loaded For Each Level To Verify Correct Loading
    print("Beginner Size: ", len(beginner))
    print("Intermediate Size: ", len(intermediate))
    print("Advanced Size: ", len(advanced))


# Main
if __name__ == "__main__":
    main() # Call The Main Function To Execute The Code And Load The Word Lists