def _get_class_names(path):
    '''
    Get the Class Map.from dir.
    :param path:
    :return:
    '''
    classes = dict()
    class_f = open(path, 'r')
    for line in class_f.readlines():
        tmp = line.strip().split(',')
        try:
            tmp[1]
        except:  # if there is no tmp[1], then the class map is Doncare.
            classes[tmp[0]] = 'DontCare'
        else:  # else tmp[1]
            classes[tmp[0]] = tmp[1]
    return classes
