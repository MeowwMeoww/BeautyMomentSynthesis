def check_ids_equal(list_A, list_B):
    flatten_A = [x for x in list_A]
    flatten_B = [x for x in list_B]

    return sorted(flatten_A) == sorted(flatten_B)