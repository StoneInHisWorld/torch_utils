from functools import wraps


def unpack_kwargs(allow_args: dict[str, tuple]):
    """
    为函数拆解输入的关键字参数。
    若调用时对某字符串参数进行赋值，则会检查该参数是否在提供的参数范围内。
    若未检测到输入值，则会指定默认值。
    会忽略不识别的参数值。
    需要被装饰函数含有parameters参数，设定其值为None，用于接受拆解过后的关键词参数。
    所有关键字参数均被视为用于拆解的参数，被装饰函数不得拥有关键字参数。
    :param allow_args: 函数允许的关键词参数，字典key为参数关键字，value为二元组。
        二元组0号位为默认值，1号位为允许范围。1号位仅当参数类型为字符串时有效，否则为空列表。
    :return: 装饰过的函数
    """

    def unpack_decorator(train_func):
        """
        装饰器
        :param train_func: 需要参数的函数
        :return 装饰过的函数
        """

        @wraps(train_func)
        def wrapper(*args, **kwargs):
            """
            按允许输入参数排列顺序拆解输入参数，或者赋值为其默认值
            :param args: train_func的位置参数
            :param kwargs: train_func的关键字参数
            :return: 包装函数
            """
            parameters = ()
            for k in allow_args.keys():
                default = allow_args[k][0]
                allow_range = allow_args[k][1]
                input_arg = kwargs.pop(k, default)
                if isinstance(default, str):
                    assert input_arg in allow_range, \
                        f'输入参数{k}:{input_arg}不在允许范围内，允许的值为{allow_range}'
                parameters = *parameters, input_arg
            return train_func(*args, parameters=parameters)

        return wrapper

    return unpack_decorator
