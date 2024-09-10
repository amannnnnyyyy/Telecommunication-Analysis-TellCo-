def calc_Total(data):
    data['Social Media Total'] = data['Social Media DL (Bytes)'] + data['Social Media UL (Bytes)']
    data['Email Total'] = data['Email UL (Bytes)']+data['Email DL (Bytes)']
    data['Google Total'] = data['Google DL (Bytes)'] + data['Google UL (Bytes)']
    data['Youtube Total'] = data['Youtube DL (Bytes)'] + data['Youtube UL (Bytes)'] 
    data['Netflix Total'] = data['Netflix DL (Bytes)'] + data['Netflix UL (Bytes)']
    data['Gaming Total'] = data['Gaming DL (Bytes)']+data['Gaming UL (Bytes)']
    data['Other Total'] = data['Other DL (Bytes)'] + data['Other UL (Bytes)']
    return data