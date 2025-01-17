""" CV model for current segmentation task
"""
import torch
import torch.nn as nn


class encoderBlock(nn.Module):
    def __init__(
            self,
            chanels_in: int, 
            chanels_out: int,
            kernal_size: tuple[int],
            num_conv_layers: int,
            dropout_p: float,
            leaky_relu_slope: float
    ) -> None:
        """Encoder block for the UNet-like architecture.
           Applies several conv layers, utilizes channels dropout, downsample in the end.
           Uses layer norm for the block output.
        """
        super().__init__()

        pad_size = [(conv_size_el - 1) // 2 for conv_size_el in kernal_size]
        conv_seq = []
        # first conv layer
        conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_out,
                        kernal_size,
                        padding=pad_size,
                        padding_mode="reflect"
        ))
        conv_seq.append(
            nn.LeakyReLU(leaky_relu_slope)
        )
        # rest conv layers
        for _ in range(num_conv_layers - 1):
            conv_seq.append(nn.Conv2d(
                        chanels_out,
                        chanels_out,
                        kernal_size,
                        padding=pad_size,
                        padding_mode="reflect"
            ))
            conv_seq.append(
                nn.LeakyReLU(leaky_relu_slope)
            )
        self._conv_seq = nn.Sequential(*conv_seq)
        
        POOL_SIZE = (2, 2)
        self._pooling = nn.MaxPool2d(POOL_SIZE)

        self._dropout = nn.Dropout2d(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # non-pooled feature map is used by decoder
        output = self._conv_seq(x)
        # layer norm on channels
        output = (output - torch.mean(output, dim=(2, 3), keepdim=True)) / torch.std(output, dim=(2, 3), keepdim=True)
        x = self._dropout(self._pooling(torch.clone(output)))

        return x, output

class decoderBlock(nn.Module):
    def __init__(
            self,
            chanels_in: int, 
            chanels_out: int,
            kernal_size: tuple[int],
            num_conv_layers: int,
            leaky_relu_slope: float
    ) -> None:
        """Decoder block for the UNet-like architecture.
           Applies upsampling first, then conv with the encoder output of the same level. Then several conv layers.
           Uses layer norm for the block output
        """
        super().__init__()

        UPSAMPLE_SCALE = (2, 2)
        self._upsampling = nn.Upsample(
            scale_factor=UPSAMPLE_SCALE,
            mode="bilinear"
        )

        pad_size = [(conv_size_el - 1) // 2  for conv_size_el in kernal_size]
        conv_seq = []
        # first conv layer counts encoder output
        conv_seq.append(nn.Conv2d(
                        2 * chanels_in,
                        chanels_in,
                        kernal_size,
                        padding=pad_size,
                        padding_mode="zeros"
        ))
        conv_seq.append(
            nn.LeakyReLU(leaky_relu_slope)
        )
        # middle conv layers
        for _ in range(num_conv_layers - 2):
            conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_in,
                        kernal_size,
                        padding=pad_size,
                        padding_mode="zeros"
            ))
            conv_seq.append(
                nn.LeakyReLU(leaky_relu_slope)
            )
        # last conv layer with channels reduction
        conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_out,
                        kernal_size,
                        padding=pad_size,
                        padding_mode="zeros"
        ))
        conv_seq.append(
            nn.LeakyReLU(leaky_relu_slope)
        )
        self._conv_seq = nn.Sequential(*conv_seq)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._upsampling(x)
        x = self._conv_seq(torch.concat([encoder_output, x], dim=1))
        # layer norm on channels
        x = (x - torch.mean(x, dim=(2, 3), keepdim=True)) / torch.std(x, dim=(2, 3), keepdim=True)

        return x


class directSegmentator(nn.Module):
    def __init__(
            self,
            num_levels: int,
            num_classes: int,
            kernal_size: tuple[int],
            num_conv_layers: int,
            encoder_dropout_p: float,
            leaky_relu_slope: float
    ) -> None:
        """ Implementing UNet-like architecture for specific classes segmentation on the image.
            Resolution decreases at 2 each level, num of channels is doubling with each level.
            Logits are an output of the model.

        Args:
            num_levels (int): depth of the encoder-decoder
            num_classes (int): num of final classes
            rest params for the encoder/decoder blocks
        """
        super().__init__()

        self._encoder_list = nn.ModuleList()
        self._decoder_list = nn.ModuleList()

        # number of channels is doubling with each level
        CHANNELS_START = 16
        # first encoder-decoder layer
        self._encoder_list.append(encoderBlock(3, CHANNELS_START, kernal_size, num_conv_layers, encoder_dropout_p, leaky_relu_slope))
        self._decoder_list.append(decoderBlock(CHANNELS_START, num_classes, kernal_size, num_conv_layers, leaky_relu_slope))
        for i in range(num_levels - 1):
            self._encoder_list.append(encoderBlock(
                CHANNELS_START * (2 ** i), 
                CHANNELS_START * (2 ** (i + 1)), 
                kernal_size, 
                num_conv_layers,
                encoder_dropout_p, 
                leaky_relu_slope))
            self._decoder_list.append(decoderBlock(
                CHANNELS_START * (2 ** (i + 1)), 
                CHANNELS_START * (2 ** i), 
                kernal_size, 
                num_conv_layers,
                leaky_relu_slope))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        # encoder stack
        for i in range(len(self._encoder_list)):
            x, encoder_out = self._encoder_list[i](x)
            encoder_outputs.append(encoder_out)
        # decoder stack
        for i in range(len(self._decoder_list) - 1, 0 - 1, -1):
            x = self._decoder_list[i](x, encoder_outputs[i])
            
        return x
        


