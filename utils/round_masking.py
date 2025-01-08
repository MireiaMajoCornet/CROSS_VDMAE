def round_masking(masking_ratio):
    ''' Round the masking ratio to the closest valid one '''
    valid_ratios = [i/196 for i in range(1, 197)]
    #hek with is the closest to the masking ratio
    closest_ratio = min(valid_ratios, key=lambda x: abs(x - masking_ratio))
    return closest_ratio
