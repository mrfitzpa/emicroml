# -*- coding: utf-8 -*-
# Copyright 2025 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
"""For creating wrappers to PyTorch optimizer classes.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For accessing attributes of functions.
import inspect

# For randomly selecting items in dictionaries.
import random



# For validating and converting objects.
import czekitout.check
import czekitout.convert

# For defining classes that support enforced validation, updatability,
# pre-serialization, and de-serialization.
import fancytypes

# For constructing machine learning optimizers.
import torch



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["AdamW",
           "Generic"]



class _Base(fancytypes.PreSerializableAndUpdatable):
    def __init__(self, ctor_params):
        if type(self) is BaseShape:
            self._generate_torch_ml_optimizer(ml_model_param_group=None)
        else:
            kwargs = ctor_params
            kwargs["skip_cls_tests"] = True
            fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

            self.execute_post_core_attrs_update_actions()

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self._clear_torch_ml_optimizer()

        return None



    def _clear_torch_ml_optimizer(self):
        self._torch_ml_optimizer = None

        return None



    def _generate_and_store_torch_ml_optimizer(self, ml_model_param_group):
        kwargs = {"ml_model_param_group": ml_model_param_group}
        self._torch_ml_optimizer = self._generate_torch_ml_optimizer(**kwargs)

        return None



    def _generate_torch_ml_optimizer(self, ml_model_param_group):
        raise NotImplementedError(_base_err_msg_1)



    def _step(self):
        self._torch_ml_optimizer.step()

        return None



def _check_and_convert_base_lr(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    base_lr = czekitout.convert.to_nonnegative_float(**kwargs)

    return base_lr



def _pre_serialize_base_lr(base_lr):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_base_lr(serializable_rep):
    base_lr = serializable_rep

    return base_lr



def _check_and_convert_weight_decay(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    weight_decay = czekitout.convert.to_nonnegative_float(**kwargs)

    return weight_decay



def _pre_serialize_weight_decay(weight_decay):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_weight_decay(serializable_rep):
    weight_decay = serializable_rep

    return weight_decay



def _check_and_convert_beta_1(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    beta_1 = czekitout.convert.to_nonnegative_float(**kwargs)

    return beta_1



def _pre_serialize_beta_1(beta_1):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_beta_1(serializable_rep):
    beta_1 = serializable_rep

    return beta_1



def _check_and_convert_beta_2(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    beta_2 = czekitout.convert.to_nonnegative_float(**kwargs)

    return beta_2



def _pre_serialize_beta_2(beta_2):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_beta_2(serializable_rep):
    beta_2 = serializable_rep

    return beta_2



def _check_and_convert_epsilon(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    epsilon = czekitout.convert.to_nonnegative_float(**kwargs)

    return epsilon



def _pre_serialize_epsilon(epsilon):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_epsilon(serializable_rep):
    epsilon = serializable_rep

    return epsilon



def _check_and_convert_use_amsgrad_variant(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    use_amsgrad_variant = czekitout.convert.to_bool(**kwargs)

    return use_amsgrad_variant



def _pre_serialize_use_amsgrad_variant(use_amsgrad_variant):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_use_amsgrad_variant(serializable_rep):
    use_amsgrad_variant = serializable_rep

    return use_amsgrad_variant



def _check_and_convert_skip_validation_and_conversion(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    skip_validation_and_conversion = czekitout.convert.to_bool(**kwargs)

    return skip_validation_and_conversion



_default_base_lr = 1e-3
_default_weight_decay = 1e-2
_default_beta_1 = 0.9
_default_beta_2 = 0.999
_default_epsilon = 1e-8
_default_use_amsgrad_variant = False
_default_skip_validation_and_conversion = False


class AdamW(_Base):
    r"""A wrapper to the PyTorch optimizer class :class:`torch.optim.AdamW`.

    The current class is a subclass of
    :class:`fancytypes.PreSerializableAndUpdatable`.

    An optimizer represented by an instance of the current class can only be
    associated with one machine learning model fitting parameter group, which is
    specified elsewhere.

    Parameters
    ----------
    base_lr : `float`, optional
        Same as the parameter ``lr`` for the class :class:`torch.optim.AdamW`.
    weight_decay : `float`, optional
        Same as the parameter ``weight_decay`` for the class 
        :class:`torch.optim.AdamW`.
    beta_1 : `float`, optional
        Same as the coefficient ``betas[0]`` of the parameter ``betas`` for the 
        class :class:`torch.optim.AdamW`.
    beta_2 : `float`, optional
        Same as the coefficient ``betas[1]`` of the parameter ``betas`` for the 
        class :class:`torch.optim.AdamW`.
    epsilon : `float`, optional
        Same as the parameter ``epsilon`` for the class 
        :class:`torch.optim.AdamW`.
    use_amsgrad_variant : `bool`, optional
        Same as the parameter ``amsgrad`` for the class 
        :class:`torch.optim.AdamW`.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("base_lr",
                        "weight_decay",
                        "beta_1",
                        "beta_2",
                        "epsilon",
                        "use_amsgrad_variant")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}

    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 base_lr=\
                 _default_base_lr,
                 weight_decay=\
                 _default_weight_decay,
                 beta_1=\
                 _default_beta_1,
                 beta_2=\
                 _default_beta_2,
                 epsilon=\
                 _default_epsilon,
                 use_amsgrad_variant=\
                 _default_use_amsgrad_variant,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _Base.__init__(self, ctor_params)

        return None



    def _generate_torch_ml_optimizer(self, ml_model_param_group):
        self_core_attrs = self.get_core_attrs(deep_copy=False)

        betas = (self_core_attrs["beta_1"], self_core_attrs["beta_2"])

        module_alias = torch.optim
        kwargs = {"params": ml_model_param_group,
                  "lr": self_core_attrs["base_lr"],
                  "weight_decay": self_core_attrs["weight_decay"],
                  "betas": betas,
                  "eps": self_core_attrs["epsilon"],
                  "amsgrad": self_core_attrs["use_amsgrad_variant"]}
        torch_ml_optimizer = module_alias.AdamW(**kwargs)

        return torch_ml_optimizer



_generic_ml_optimizer_name_to_cls_map = \
    {"adam_w": AdamW}



def _check_and_convert_ml_optimizer_name(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    obj = params[obj_name]

    accepted_strings = _generic_ml_optimizer_name_to_cls_map.keys()

    kwargs = {"obj": obj,
              "obj_name": obj_name,
              "accepted_strings": accepted_strings}
    czekitout.check.if_one_of_any_accepted_strings(**kwargs)

    ml_optimizer_name = obj

    return ml_optimizer_name



def _pre_serialize_ml_optimizer_name(ml_optimizer_name):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_ml_optimizer_name(serializable_rep):
    ml_optimizer_name = serializable_rep

    return ml_optimizer_name



def _check_and_convert_ml_optimizer_params(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    obj = params[obj_name]

    ml_optimizer_name = _check_and_convert_ml_optimizer_name(params)
    ml_optimizer_cls = _generic_ml_optimizer_name_to_cls_map[ml_optimizer_name]

    accepted_types = (dict, type(None))

    if isinstance(obj, accepted_types[-1]):
        ml_optimizer_params = ml_optimizer_cls().get_core_attrs(deep_copy=False)
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)

        try:
            ml_optimizer_cls(**obj)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise ValueError(err_msg)

        ml_optimizer_params = obj

    return ml_optimizer_params



def _pre_serialize_ml_optimizer_params(ml_optimizer_params):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_ml_optimizer_params(serializable_rep):
    ml_optimizer_params = serializable_rep

    return ml_optimizer_params



_default_ml_optimizer_name = "adam_w"
_default_ml_optimizer_params = None



class Generic(_Base):
    r"""A generic wrapper to a PyTorch optimizer class.

    The current class is a subclass of
    :class:`fancytypes.PreSerializableAndUpdatable`.

    The parameters of the current class specify an instance of one of the other
    public classes defined in the module :mod:`emicroml.modelling.optimizers`,
    with the other public class being a wrapper to a specific PyTorch optimizer
    class. Let ``specific_wrapper_cls`` be the other public class.

    An optimizer represented by an instance of the current class can only be
    associated with one machine learning model fitting parameter group, which is
    specified elsewhere.

    Parameters
    ----------
    ml_optimizer_name : "adam_w", optional
        The name associated with
        ``specific_wrapper_cls``. ``specific_wrapper_cls`` is determined by the
        value of ``ml_optimizer_name``:

        * If ``ml_optimizer_name`` is set to ``"adam_w"``, then
          ``specific_wrapper_cls`` is the class
          :class:`emicroml.modelling.optimizers.AdamW`.

    ml_optimizer_params : `dict` | `None`, optional
        ``ml_optimizer_params`` specifies the parameters of 
        ``specific_wrapper_cls`` used to construct an instance of itself. If
        ``ml_optimizer_params`` is set to ``None``, then the instance is 
        constructed by ``specific_wrapper_cls()``. Otherwise, the instance is
        constructed by ``specific_wrapper_cls(**ml_optimizer_params)``.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("ml_optimizer_name",
                        "ml_optimizer_params")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}

    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 ml_optimizer_name=\
                 _default_ml_optimizer_name,
                 ml_optimizer_params=\
                 _default_ml_optimizer_params,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _Base.__init__(self, ctor_params)

        return None



    def _generate_torch_ml_optimizer(self, ml_model_param_group):
        self_core_attrs = self.get_core_attrs(deep_copy=False)

        map_alias = _generic_ml_optimizer_name_to_cls_map
        key = self_core_attrs["ml_optimizer_name"]
        ml_optimizer_cls = map_alias[key]

        kwargs = self_core_attrs["ml_optimizer_params"]
        ml_optimizer = ml_optimizer_cls(**kwargs)

        kwargs = {"ml_model_param_group": ml_model_param_group}
        torch_ml_optimizer = ml_optimizer._generate_torch_ml_optimizer(**kwargs)

        return torch_ml_optimizer



def _check_and_convert_ml_optimizer(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    obj = params[obj_name]

    accepted_types = (Generic, type(None))
    
    if isinstance(obj, accepted_types[-1]):
        ml_optimizer = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)

        kwargs = obj.get_core_attrs(deep_copy=False)
        ml_optimizer = accepted_types[0](**kwargs)
        ml_optimizer._torch_ml_optimizer = obj._torch_ml_optimizer

    return ml_optimizer



def _pre_serialize_ml_optimizer(ml_optimizer):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_ml_optimizer(serializable_rep):
    ml_optimizer = Generic.de_pre_serialize(serializable_rep)
    
    return ml_optimizer



_default_ml_optimizer = None



###########################
## Define error messages ##
###########################

_base_err_msg_1 = \
    ("Cannot construct instances of the class "
     "`emicroml.modelling.optimizers._Base`, only subclasses of itself defined "
     "in the `emicroml.modelling.optimizers` module.")

_check_and_convert_ml_optimizer_params_err_msg_1 = \
    ("The object ``ml_optimizer_params`` does not specify a valid set of "
     "machine learning optimizer parameters: see traceback for details.")
