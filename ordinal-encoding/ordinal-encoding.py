def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Create a mapping dictionary where keys are the categories 
    # and values are their corresponding 0-indexed positions.
    mapping = {category: index for index, category in enumerate(ordering)}
    
    # Replace each value in the input list with its mapped integer.
    return [mapping[value] for value in values]