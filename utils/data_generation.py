import pandas as pd

def generate_sample_data(desc):
    """
    Generate data based on the description.

    Parameters:
        desc: description of the data
    
    Returns:
        data: generated data
    """
    
    if "sales" in desc.lower():
        data = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "Sales": [1000, 1500, 1200, 1700, 1300, 1600]
        })
    else:
        data = pd.DataFrame({
            "X": range(10),
            "Y": [x**2 for x in range(10)]
        })
    
    return data