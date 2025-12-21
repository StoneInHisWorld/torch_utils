class PredictionWrapper:

    def __init__(self, module_type: type, raw_ds=None):
        """预测结果包装器
        利用本包装器需要编写子类，并在子类中编写对应网络的包装函数，格式为：
            {module_type.__name__}_wrap_fn(**kwargs)
        在这个类中可以直接通过属性的方式访问所有预测结果，支持的预测数据包括：
            self.predictions
            if ret_ds:
                self.inputs, self.labels
            if ret_ls_metric:
                self.metrics, self.ls_es, self.critera_names, self.ls_names
            如果调用wrap()时给定的预测结果不包含上述内容，访问的时候将会抛出AttributeError

        Args:
            module_type: 包装器对象服务的网络类型
        """
        self.module_type = module_type
        self.raw_ds = raw_ds

    def wrap(self, prediciton_results,
             ret_ds: bool = True, ret_ls_metric: bool = True, **kwargs):
        """对预测结果进行提取，根据网络类型调用包装后返回
        预测结果根据ret_ds, ret_ls_metric不同而有不同的结果，可以通过属性进行访问

        Args:
            prediciton_results: 预测结果，将根据ret_ds, ret_ls_metric的值进行结果解包
            ret_ds: 是否返回数据集
            ret_ls_metric: 是否返回损失值和评价指标
            **kwargs: 包装函数所用关键字参数

        Returns:
            包装好的结果
        """
        self.predictions = prediciton_results.pop(0)
        if ret_ds:
            self.inputs, self.labels = prediciton_results.pop(0), prediciton_results.pop(0)
        if ret_ls_metric:
            (self.metrics, self.ls_es, self.criteria_fns, self.criteria_names,
             self.ls_fns, self.ls_names) = [prediciton_results.pop(0) for _ in range(6)]
        try:
            wrapped_results = getattr(self, f'{self.module_type.__name__}_wrap_fn')(**kwargs)
        except AttributeError:
            raise NotImplementedError(f"没有为{self.module_type.__name__}定制结果包装程序！"
                                      f"请在{self.__class__.__name__}类中创建{self.module_type.__name__}_wrap_fn()方法！")
        del self.predictions
        if ret_ds:
            del self.inputs, self.labels
        if ret_ls_metric:
            del self.metrics, self.ls_es, self.criteria_fns, self.criteria_names,\
             self.ls_fns, self.ls_names
        return wrapped_results