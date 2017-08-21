
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'vigan':
        from .VIGAN_model import VIGANModel
        assert(opt.align_data == False)
        model = VIGANModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
