# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Chameleon.
"""
import torch
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class ChameleonProcessor(ProcessorMixin):
    r"""
    Constructs a Chameleon processor which wraps a Chameleon image processor and a Chameleon tokenizer into a single
    processor.

    [`ChameleonProcessor`] offers all the functionalities of [`ChameleonImageProcessor`] and [`LlamaTokenizerFast`].
    See the [`~ChameleonProcessor.__call__`] and [`~ChameleonProcessor.decode`] for more information.

    Args:
        image_processor ([`ChameleonImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    image_processor_class = "ChameleonImageProcessor"

    def __init__(self, image_processor, tokenizer, image_seq_length: int = 1024, image_token: str = "<image>"):
        self.image_seq_length = image_seq_length
        self.image_token = image_token
        self.image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
        self.image_end_token = "<eoss>"
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        return_for_text_completion: bool = False,
        return_image_masks: bool = False
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        # Replace the image token with the expanded image token sequence
        prompt_strings = []
        one_img_tokens = self.image_start_token + (self.image_token * self.image_seq_length) + self.image_end_token
        for sample in text:
            sample = sample.replace(self.image_token, one_img_tokens)
            if not return_for_text_completion:
                sample += self.tokenizer.sep_token  # special Chameleon treatment to add sep for chat mode
            prompt_strings.append(sample)

        import re
        pattern = r'<image\d+-\d+-\d+-\d+>'
        matches = []
        for sample in text:
            matches.append(re.findall(pattern, sample))
        
        sub_image_masks = torch.zeros(32, 32).bool()
      
        for j, ms, t in zip(range(len(text)), matches, prompt_strings):
            if ms == []:
                continue
            for m in ms:
                p2 = r'\d+'
                ranges = [int(i) for i in re.findall(p2, m)]
                assert True not in sub_image_masks[ranges[0]: ranges[1], ranges[2]: ranges[3]]
                sub_image_masks[ranges[0]: ranges[1], ranges[2]: ranges[3]] = True
                num_holders = (ranges[1]- ranges[0] ) * (ranges[3]- ranges[2])
                t = t.replace(m, self.image_start_token + self.image_token * num_holders + self.image_end_token )
            prompt_strings[j] = t

        data = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
            data["pixel_values"] = pixel_values
            
        if return_image_masks:
            return BatchFeature(data=data, tensor_type=return_tensors), sub_image_masks
        else:
            return BatchFeature(data=data, tensor_type=return_tensors)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

