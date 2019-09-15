from collections import defaultdict


def group_by_field(data, fields):
    d = defaultdict(list)
    for item in data:
        if isinstance(fields, (tuple, list)):
            k = []
            for field in fields:
                k.append(item[field])
            if len(k) > 1:
                k = tuple(k)
            else:
                k = k[0]
        else:
            k = item[fields]
        d[k].append(item)
    return d


groups = group_by_field(scripts, ('bnf_name',))
test_max_item = get_max_item(groups=groups)
