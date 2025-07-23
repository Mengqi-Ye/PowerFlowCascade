def add_lineID_clone_ways(data, country_code='VN'):
    """
    Creates a unique 'LineID' for each way element in the dataset.
    If a way needs to be cloned (has more than one system), it will be duplicated, tripled, or quadrupled.
    
    Parameters:
    - data (DataFrame): Input dataset containing way elements.
    - country_code (str): Two-letter country code.
    
    Returns:
    - DataFrame: New dataset with cloned ways and 'LineID' column.
    """
    start_time = time.time()
    print('Start adding "LineID" and cloning ways...')
    
    # Fill NaN values with 1 (indicating no clone) and convert to int
    # data['systems'] = data['systems'].fillna(1).astype(int)

    # Create unique LineID prefix
    lineID_prefix = f'LINE{country_code}'
    
    # Initialize list for new data
    data_new = []
    
    # Process each row in data
    for i, row in data.iterrows():
        num_clones = int(row['circuits']) # Determine number of clones based on 'circuits' value
        base_lineID = f"{lineID_prefix}{str(i+1).zfill(4)}"  # Base LineID with four digits
        
        if num_clones == 1:
            # For rows where circuits = 1, add only the base LineID
            row['LineID'] = base_lineID
            data_new.append(row)  # Add the original row to data_new
       
        else:
            # For rows where circuits > 1, create clones as per 'circuits' 
            # and add suffixes 'a', 'b', 'c', 'd' as needed
            clones = [row.copy() for _ in range(num_clones)]
            for j, clone in enumerate(clones):
                clone['LineID'] = f"{base_lineID}{chr(97 + j)}"  # Append 'a', 'b', 'c', 'd'
                data_new.append(clone)
    
    # Convert list of expanded data back to a DataFrame
    data_new = pd.DataFrame(data_new).reset_index(drop=True)

    # Print cloning summary
    print(f"   ... {sum(row['circuits'] == 2 for _, row in data.iterrows())} ways doubled, "
          f"{sum(row['circuits'] == 3 for _, row in data.iterrows())} tripled, "
          f"{sum(row['circuits'] == 4 for _, row in data.iterrows())} quadrupled.")
    print(f'   ... finished! ({time.time() - start_time:.3f} seconds) \n')
    
    return gpd.GeoDataFrame(data_new, geometry='geometry')