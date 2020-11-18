#!usr/bin/python
# -*- coding: utf-8 -*-

"""
CAM
"""

import math
import torch
import torch.nn.functional as F
from torchcammaster.torchcam.utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer, find_layer, find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer

__all__ = ['CAM', 'ScoreCAM', 'IntersectionSamCAM', 'SamCAM', 'SamCAM2', 'SamCAM3', 'SamCAM4', 'DropCAM', 'SSCAM',
           'ISSCAM']


class _CAM(object):
    """Implements a class activation map extractor

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer) -> None:

        # if not hasattr(model, conv_layer):
        #    raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model.arch

        # Forward hook
        model_type = model.name
        layer_name = model.layer
        if 'vgg' in model_type.lower():
            self.conv_layer = find_vgg_layer(self.model, layer_name)
        elif 'resnet' in model_type.lower():
            self.conv_layer = find_resnet_layer(self.model, layer_name)
        elif 'densenet' in model_type.lower():
            self.conv_layer = find_densenet_layer(self.model, layer_name)
        elif 'alexnet' in model_type.lower():
            self.conv_layer = find_alexnet_layer(self.model, layer_name)
        elif 'squeezenet' in model_type.lower():
            self.conv_layer = find_squeezenet_layer(self.model, layer_name)
        elif 'googlenet' in model_type.lower():
            self.conv_layer = find_googlenet_layer(self.model, layer_name)
        elif 'shufflenet' in model_type.lower():
            self.conv_layer = find_shufflenet_layer(self.model, layer_name)
        elif 'mobilenet' in model_type.lower():
            self.conv_layer = find_mobilenet_layer(self.model, layer_name)
        else:
            self.conv_layer = find_layer(self.model, layer_name)
        # print(self.model,self.model._modules.get(conv_layer),conv_layer)

        # self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        self.hook_handles.append(self.conv_layer.register_forward_hook(self._hook_a))
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module, input, output):
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def clear_hooks(self):
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def _normalize(cams):
        """CAM normalization"""
        cams -= cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
        cams_flatten_max_nonzero = cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
        cams_flatten_max_nonzero = torch.where(cams_flatten_max_nonzero != 0, cams_flatten_max_nonzero,
                                               torch.tensor(10e-8).to(device=cams_flatten_max_nonzero.device))
        cams /= cams_flatten_max_nonzero

        return cams

    def _get_weights(self, class_idx, scores=None):

        raise NotImplementedError

    def _precheck(self, class_idx, scores):
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if self.hook_a is None:
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")

        # Check class_idx value
        if class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        #  Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, class_idx, inp, scores=None, normalized=True):

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_cams(class_idx, inp, scores, normalized)

    def compute_cams(self, class_idx, inp, scores=None, normalized=True):
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the hooked model
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """
        '''
        _, _, h, w = inp.shape
        tmp=torch.zeros((self.hook_a.shape[1],h,w))
        for i in range(self.hook_a.shape[1]):
            print(self.hook_a.shape[1],self.hook_a[:,i,:,:].unsqueeze(0).shape)
            tmp[i,:]=F.interpolate(self.hook_a[:,i,:,:].unsqueeze(0), size=(h, w), mode='bilinear',
                                   align_corners=False)

        self.hook_a=tmp
        '''
        # Get map weight
        weights = self._get_weights(class_idx, scores)

        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a).to(device=weights.device)

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, inp.shape[-2:], mode='bilinear', align_corners=False)

        # Perform the weighted combination to get the CAM
        batch_cams = (weights.unsqueeze(-1).unsqueeze(-1) * upsampled_a.squeeze(0)).sum(dim=0)

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        self.clear_hooks()

        return batch_cams

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CAM(_CAM):
    """Implements a class activation map extractor as described in `"Learning Deep Features for Discriminative
    Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for unit :math:`k` in the fully
    connected layer..

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import CAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = CAM(model, 'layer4', 'fc')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        fc_layer (str): name of the fully convolutional layer
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, fc_layer):
        print(fc_layer, type(fc_layer))

        super().__init__(model, conv_layer)
        # Softmax weight
        if model.name == 'vgg16':
            self._fc_weights = self.model.classifier[6].weight.data
        else:
            self._fc_weights = self.model._modules.get(fc_layer).weight.data
            print(fc_layer, self.model._modules.get(fc_layer))

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Take the FC weights of the target class
        return self._fc_weights[class_idx, :]


class _ScoreCAM(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, batch_size=32):

        super().__init__(model, conv_layer)

        # Input hook
        self.hook_handles.append(self.model._modules.get(input_layer).register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, module, input):
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        self.upsampled_a = self._normalize(self.hook_a)

        #  Upsample it to input_size
        # 1 * O * M * N
        self.upsampled_a = F.interpolate(self.upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        masked_input = self.upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        weights = torch.zeros(masked_input.shape[0], dtype=masked_input.dtype).to(device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(weights.shape[0] / self.bs)):
            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                #  Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class ScoreCAM(_ScoreCAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, batch_size=32):
        super().__init__(model, conv_layer, input_layer, batch_size)

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        return super()._get_weights(class_idx, scores)

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class IntersectionSamCAM(_ScoreCAM):
    """Implements a class activation map extractor as described in `"SS-CAM: Smoothed Score-CAM for
    Sharper Visual Feature Localization" <https://arxiv.org/pdf/2006.14255.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{SS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{N} \\sum\\limits_1^N softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U` is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import SSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SSCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
        num_samples (int, optional): number of noisy samples used for weight computation
        std (float, optional): standard deviation of the noise added to the normalized activation
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, k=10, batch_size=32):
        super().__init__(model, conv_layer, input_layer, batch_size)
        self.k = k

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""
        '''
        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a)

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        masked_input = upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        sc = torch.zeros(masked_input.shape[0], dtype=masked_input.dtype).to(device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(sc.shape[0] / self.bs)):
            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, sc.shape[0]))
            with torch.no_grad():
                #  Get the softmax probabilities of the target class
                sc[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]
        '''
        sc=super()._get_weights()
        max_idx = sc.topk(self.k)[1]

        a = self.upsampled_a.squeeze(0).view(self.upsampled_a.shape[1], 1, -1).squeeze(1)
        b = self.upsampled_a[:, max_idx].sum(dim=1).view(1, 1, -1).squeeze(1)

        print(sc.shape, max_idx.shape, a.shape, b.shape)
        weights = torch.norm(a * b, dim=1)
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k},batch_size={self.bs})"


class DropCAM(_ScoreCAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, max_iter=600, prob=0.8, batch_size=32):

        super().__init__(model, conv_layer, input_layer)

        # Input hook
        self.hook_handles.append(self.model._modules.get(input_layer).register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True
        self.max_iter = max_iter
        self.prob = prob

    def _store_input(self, module, input):
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a).cpu()

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        dropout_weights = torch.ones(self.max_iter, upsampled_a.shape[1]).to(device=upsampled_a.device)

        for i in range(self.max_iter):
            zero_idx = torch.randperm(upsampled_a.shape[1])[:int(upsampled_a.shape[1] * self.prob)]
            dropout_weights[i, zero_idx] = 0.

        sc = torch.zeros(self.max_iter, upsampled_a.shape[1]).to(device=self._input.device)

        masked_input = (upsampled_a.squeeze(0) * dropout_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        masked_input = (masked_input - masked_input.min()) / (masked_input.max() - masked_input.min())

        # Disable hook updates
        self._hooks_enabled = False
        upsampled_a, masked_input = upsampled_a.cuda(), masked_input.cuda()
        masked_input = masked_input.unsqueeze(1) * self._input

        #  Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(self.max_iter / self.bs)):

            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, self.max_iter))
            one_idx = dropout_weights[selection_slice, :].nonzero()
            with torch.no_grad():

                out = F.softmax(self.model(masked_input[selection_slice, :, :, :]), dim=1)[:, class_idx]
                for i in range(one_idx.shape[0]):
                    sc[one_idx[i, 0], one_idx[i, 1]] = out[one_idx[i, 0]]
        '''
        for iter in range(self.max_iter):

            masked_input=(upsampled_a.squeeze(0)*dropout_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
            masked_input=(masked_input-masked_input.min())/(masked_input.max()-masked_input.min())
            masked_input = masked_input * self._input

            with torch.no_grad():
                sc[one_idx, iter] = F.softmax(self.model(masked_input), dim=1)[:, class_idx]
        '''
        # Reenable hook updates
        self._hooks_enabled = True

        mask = sc != 0
        weights = (sc * mask).sum(dim=0) / (mask.sum(dim=0) + 1e-8)

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SamCAM(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, batch_size=32):
        super().__init__(model, conv_layer)

        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self.hook_a  # self._normalize(self.hook_a)
        # upsampled_a=(upsampled_a-upsampled_a.min())/(upsampled_a.max()-upsampled_a.min())
        #  Upsample it to input_size
        # 1 * O * M * N
        # upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0], dtype=upsampled_a.dtype).to(device=upsampled_a.device)

        max_tensor = upsampled_a.view(1, upsampled_a.shape[1], -1, 1).max(dim=2)
        max_sum = max_tensor[0].sum(dim=1)

        weights = max_tensor[0] / max_sum
        weights = weights.squeeze(0).squeeze(-1)
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # print(weights,max_tensor[0])
        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)

        '''
        for idx in range(math.ceil(weights.shape[0] / self.bs)):

            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                # Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]
        '''
        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SamCAM2(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, p=10, batch_size=32):
        super().__init__(model, conv_layer)

        self.pow, self.bs = p, batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self.hook_a  # self._normalize(self.hook_a)
        # upsampled_a=(upsampled_a-upsampled_a.min())/(upsampled_a.max()-upsampled_a.min())
        #  Upsample it to input_size
        # 1 * O * M * N
        # upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0], dtype=upsampled_a.dtype).to(device=upsampled_a.device)

        max_tensor = upsampled_a.view(1, upsampled_a.shape[1], -1, 1).max(dim=2)

        max_sum = max_tensor[0].sum(dim=1)

        weights = max_tensor[0] / max_sum
        weights = weights.squeeze(0).squeeze(-1)

        weights = (weights - weights.min()) / (weights.max() - weights.min())
        weights = weights ** self.pow
        # print(weights,max_tensor[0])
        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)

        '''
        for idx in range(math.ceil(weights.shape[0] / self.bs)):

            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                # Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]
        '''
        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SamCAM3(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, p=10, batch_size=32):
        super().__init__(model, conv_layer)

        self.pow, self.bs = p, batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self.hook_a  # self._normalize(self.hook_a)
        upsampled_a = (upsampled_a - upsampled_a.min()) / (upsampled_a.max() - upsampled_a.min())
        #  Upsample it to input_size
        # 1 * O * M * N
        # upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W

        # Initialize weights
        # weights = torch.zeros(upsampled_a.shape[0], dtype=upsampled_a.dtype).to(device=upsampled_a.device)

        import torch.linalg as LA

        norm = LA.norm(upsampled_a.view(*upsampled_a.shape[:-2], -1, 1), 1, dim=2)
        norm = (norm - norm.min()) / (norm.max() - norm.min())

        max_tensor = upsampled_a.view(1, upsampled_a.shape[1], -1, 1).max(dim=2)
        print(max_tensor[0])

        max_sum = max_tensor[0].sum(dim=1)

        weights = max_tensor[0] / max_sum
        weights = weights.squeeze(0).squeeze(-1)
        norm = norm.squeeze(0).squeeze(-1)

        weights = (weights - weights.min()) / (weights.max() - weights.min())
        weights *= (1 - norm)
        weights = weights ** self.pow
        # print(weights,max_tensor[0])
        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)

        '''
        for idx in range(math.ceil(weights.shape[0] / self.bs)):

            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                # Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]
        '''
        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SamCAM4(_ScoreCAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, batch_size=32):

        super().__init__(model, conv_layer, input_layer)

        # Input hook
        self.hook_handles.append(self.model._modules.get(input_layer).register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, module, input):
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        myHa = (self.hook_a - self.hook_a.min()) / (self.hook_a.max() - self.hook_a.min())
        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a)

        max_tensor = myHa.view(1, myHa.shape[1], 1, -1).max(dim=3)
        myW = max_tensor[0].squeeze(0).squeeze(-1)

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        masked_input = upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        weights = torch.zeros(masked_input.shape[0], dtype=masked_input.dtype).to(device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        #  Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(weights.shape[0] / self.bs)):
            selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                #  Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True

        return weights * myW

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SSCAM(_ScoreCAM):
    """Implements a class activation map extractor as described in `"SS-CAM: Smoothed Score-CAM for
    Sharper Visual Feature Localization" <https://arxiv.org/pdf/2006.14255.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{SS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{N} \\sum\\limits_1^N softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U` is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import SSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SSCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
        num_samples (int, optional): number of noisy samples used for weight computation
        std (float, optional): standard deviation of the noise added to the normalized activation
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, batch_size=32, num_samples=35, std=2.0):

        super().__init__(model, conv_layer, input_layer, batch_size)

        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a)

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        upsampled_a = upsampled_a.squeeze(0).unsqueeze(1)

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0], dtype=upsampled_a.dtype).to(device=upsampled_a.device)

        # Disable hook updates
        self._hooks_enabled = False

        for _idx in range(self.num_samples):
            noisy_m = self._input * (upsampled_a +
                                     self._distrib.sample(self._input.size()).to(device=self._input.device))

            #  Process by chunk (GPU RAM limitation)
            for idx in range(math.ceil(weights.shape[0] / self.bs)):
                selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))

                with torch.no_grad():
                    #  Get the softmax probabilities of the target class
                    sc = F.softmax(self.model(noisy_m[selection_slice]), dim=1)[:, class_idx]
                    # print(weights[selection_slice],sc)
                    weights[selection_slice] = weights[selection_slice].where(weights[selection_slice] > sc, sc)
                    # weights[selection_slice] += F.softmax(self.model(noisy_m[selection_slice]), dim=1)[:, class_idx]

        # weights /= self.num_samples

        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.bs}, num_samples={self.num_samples}, std={self.std})"


class ISSCAM(_ScoreCAM):
    """Implements a variant of Score-CAM, based on Rakshit Naidu's `work
    <https://github.com/r0cketr1kky/ISS-CAM_resources>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{ISS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^N \\frac{i}{N} softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U` is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ISSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ISSCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        input_layer (str): name of the first layer
        batch_size (int, optional): batch size used to forward masked inputs
        num_samples (int, optional): number of noisy samples used for weight computation
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, batch_size=32, num_samples=10):

        super().__init__(model, conv_layer, input_layer, batch_size)

        self.num_samples = num_samples

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        upsampled_a = self._normalize(self.hook_a)

        #  Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        upsampled_a = upsampled_a.squeeze(0).unsqueeze(1)

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0], dtype=upsampled_a.dtype).to(device=upsampled_a.device)

        # Disable hook updates
        self._hooks_enabled = False
        fmap = 0

        for _idx in range(self.num_samples):
            fmap += (_idx + 1) / self.num_samples * self._input * upsampled_a

            #  Process by chunk (GPU RAM limitation)
            for idx in range(math.ceil(weights.shape[0] / self.bs)):
                selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs, weights.shape[0]))
                with torch.no_grad():
                    #  Get the softmax probabilities of the target class
                    weights[selection_slice] += F.softmax(self.model(fmap[selection_slice]), dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True

        return weights
